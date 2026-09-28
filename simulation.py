#!/usr/bin/env python3
"""
simulation.py

Reproduces every number, table entry and figure of the manuscript
"When Is a Variable-Order Fractional Capacitor Passive? Energy Identity, Rate Condition,
and a Certified Counterexample".
Repository: https://github.com/emailnaifar-ai/Fractional-Capacitor (MIT License).

Model (time measured in units of the reference time t_r = 1 s):
    iota(t) = C_v * D^{alpha(t)} v(t),
    D^{alpha(t)} v(t) = 1/Gamma(1-alpha(t)) * int_0^t (t-tau)^(-alpha(t)) v'(tau) dtau.

Sections
    A  L1 matrix H_{h,alpha}, circuit solver (Method A), Grunwald-Letnikov solver (Method B)
    B  certified counterexample (interval arithmetic, mpmath.iv), two-sided enclosures
    C  rate bound beta_*(T0) of the corollary and guaranteed horizons
    D  supercapacitor circuit on three grids, energies, Richardson extrapolation, timings
    E  discrete passivity test lambda_min(S_h), mass-matrix pencil mu_h, discrete horizon,
       port (terminal) energy form, fixed test voltage
    F  verification of the identities used in the paper
    G  figures (PDF and PNG) and CSV files
    H  direct evaluation of the kernel condition p_T >= 0 on the triangle 0 < tau < T <= T0
    I  mass matrix, exact continuous energies of piecewise-linear voltages (closed-form current),
       exact P1 (Galerkin) energy matrix, port form; criteria sweep alpha_w = 0.7 + 0.2 sin(w t)
    J  exact P1 energies for alpha_r: minimum Rayleigh quotient and horizon scan

Numerical settings: direct solvers (forward substitution, LAPACK eigh/Cholesky); adaptive quadrature
epsrel = 1e-11 (1e-12 in the N_hat closed-form check), epsabs = 1e-13 (verification only); Nelder-Mead polish of min p_T with xatol = 1e-10,
fatol = 1e-14; fixed-count bisections as documented in the code; interval arithmetic with 30 digits.

Outputs are written to ./figures and ./results.  All quantities except wall-clock
timings are deterministic (no random numbers except a seeded check in Section F).
Run:  python simulation.py
"""
import csv
import os
import platform
import time

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.linalg import eigh, solve_triangular
from scipy.special import digamma, gamma
import mpmath
from mpmath import iv, mp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# ----------------------------------------------------------------------------
# Fixed parameters
# ----------------------------------------------------------------------------
RS = 0.1          # series resistance [Ohm]
RP = 100.0        # parallel (leakage) resistance [Ohm]
CV = 10.0         # reference-time-normalized coefficient C_v (t_r = 1 s)
T_END = 15.0      # simulation horizon (units of t_r)
GRIDS = [0.01, 0.005, 0.0025]   # three time resolutions
U_ROUND = 2.0**-53              # unit roundoff of IEEE double precision

ORDERS = {
    "constant": dict(f=lambda t: 0.70 + 0.0 * np.asarray(t, float),
                     df=lambda t: 0.0 * np.asarray(t, float),
                     lo=0.70, hi=0.70, dmax=0.0),
    "slow": dict(f=lambda t: 0.70 + 0.03 * np.sin(0.2 * np.asarray(t, float)),
                 df=lambda t: 0.006 * np.cos(0.2 * np.asarray(t, float)),
                 # on [0,15], 0.2 t lies in [0,3] and sin >= 0, so alpha_s in [0.70,0.73]
                 lo=0.70, hi=0.73, dmax=0.006),
    "rapid": dict(f=lambda t: 0.70 + 0.20 * np.sin(8.0 * np.asarray(t, float)),
                  df=lambda t: 1.6 * np.cos(8.0 * np.asarray(t, float)),
                  lo=0.50, hi=0.90, dmax=1.6),
}
CASES = list(ORDERS)

# Colours: first three slots of the validated categorical palette (all-pairs safe),
# with line styles as secondary encoding for print and colour-vision deficiency.
STYLE = {
    "constant": dict(color="#2a78d6", ls="-", label=r"$\alpha_{\mathrm{c}}$ (constant)"),
    "slow": dict(color="#eb6834", ls="--", label=r"$\alpha_{\mathrm{s}}$ (slow)"),
    "rapid": dict(color="#1baf7a", ls="-.", label=r"$\alpha_{\mathrm{r}}$ (rapid)"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "legend.frameon": False, "savefig.bbox": "tight",
    "mathtext.fontset": "cm",
})


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)


def write_csv(name, header, rows):
    with open(os.path.join(RES, name), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def iapp(t):
    """Applied current profile [A]; values on the open switching intervals."""
    t = np.asarray(t, float)
    return np.where(t < 5, 2.0, np.where(t < 8, 0.0, np.where(t < 12, -1.0, 0.0)))


# ----------------------------------------------------------------------------
# A. Discretizations
# ----------------------------------------------------------------------------
def build_H(alpha, h, N, cv):
    """L1 matrix with the order frozen at each node t_n = n h, n = 1..N.

    (H v)_n = cv * sum_{j=1}^n a_{n,j} (v_j - v_{j-1}),  v_0 = 0,
    a_{n,j} = (1/h) int_{t_{j-1}}^{t_j} kappa(t_n, tau) dtau
            = h^{-a_n} [(n-j+1)^{1-a_n} - (n-j)^{1-a_n}] / Gamma(2-a_n).
    Returned H is N x N lower triangular with H[n-1, j-1] = cv (a_{n,j} - a_{n,j+1}).
    """
    t = h * np.arange(1, N + 1)
    a = np.asarray(alpha(t), float)
    l = np.arange(0, N + 1, dtype=float)
    logl = np.log(np.where(l > 0, l, 1.0))
    H = np.zeros((N, N))
    for n in range(1, N + 1):
        e = 1.0 - a[n - 1]
        P = np.exp(e * logl[: n + 1])
        P[0] = 0.0
        b = P[1:] - P[:-1]                 # b_l = (l+1)^e - l^e, l = 0..n-1
        c = cv * h ** (-a[n - 1]) / gamma(2.0 - a[n - 1])
        bl = b[::-1].copy()                # entry j-1 holds b_{n-j}
        row = bl.copy()
        row[:-1] -= bl[1:]                 # b_{n-j} - b_{n-j-1}
        H[n - 1, :n] = c * row
    return H


def energy_matrix(H, h):
    """S_{h,alpha} = (W_h H + H^T W_h)/2 with W_h = h I."""
    return 0.5 * h * (H + H.T)


def lam_min(S):
    return float(eigh(S, eigvals_only=True, subset_by_index=[0, 0])[0])


def circuit_L1(alpha, h, H=None):
    """Method A: L1 operator, right-endpoint energy quadrature (W_h = h I)."""
    N = int(round(T_END / h))
    t = h * np.arange(1, N + 1)
    if H is None:
        H = build_H(alpha, h, N, CV)
    i = iapp(t - 0.5 * h)                 # current on (t_{n-1}, t_n]
    v = solve_triangular(H + np.eye(N) / RP, i, lower=True)
    ifr = H @ v
    u = RS * i + v
    E = dict(sup=h * np.cumsum(u * i), Rs=h * np.cumsum(RS * i ** 2),
             Rp=h * np.cumsum(v ** 2 / RP), frac=h * np.cumsum(v * ifr))
    return dict(t=t, v=v, i=i, u=u, ifrac=ifr, p=u * i, pfrac=v * ifr, E=E, H=H)


def circuit_GL(alpha, h):
    """Method B: Grunwald-Letnikov operator with frozen order (v(0) = 0, so the
    Caputo and Riemann-Liouville derivatives of each frozen order coincide) and
    trapezoidal energy quadrature with exact treatment of the current jumps."""
    N = int(round(T_END / h))
    t = h * np.arange(0, N + 1)
    a = np.asarray(alpha(t), float)
    v = np.zeros(N + 1)
    i = iapp(t[1:] - 0.5 * h)
    jj = np.arange(1, N + 1, dtype=float)
    for n in range(1, N + 1):
        g = np.cumprod(1.0 - (a[n] + 1.0) / jj[:n])       # g_1..g_n
        hist = g @ v[n - 1::-1][:n]                       # sum_j g_j v_{n-j}
        c = CV * h ** (-a[n])
        v[n] = (i[n - 1] - c * hist) / (c + 1.0 / RP)
    ERs = RS * (2.0 ** 2 * 5 + 1.0 ** 2 * 4)             # exact
    ERp = np.trapezoid(v ** 2, t) / RP
    vi = 0.0
    for (ta, tb, I) in [(0, 5, 2.0), (5, 8, 0.0), (8, 12, -1.0), (12, 15, 0.0)]:
        na, nb = int(round(ta / h)), int(round(tb / h))
        vi += I * np.trapezoid(v[na:nb + 1], t[na:nb + 1])
    return dict(t=t, v=v, E=dict(sup=ERs + vi, Rs=ERs, Rp=ERp, frac=vi - ERp))


def richardson(e1, e2, e3):
    """Observed order and extrapolated value from grids h, h/2, h/4."""
    d1, d2 = e1 - e2, e2 - e3
    p = np.log2(d1 / d2)
    return p, e3 + (e3 - e2) / (2.0 ** p - 1.0)


# ----------------------------------------------------------------------------
# B. Certified counterexample
# ----------------------------------------------------------------------------
CEX = dict(a1="0.1", a2="0.9", eps="0.01", T="0.1", P=2000, dps=30)   # P: hold subintervals (paper: P)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return 3 * x ** 2 - 2 * x ** 3


def alpha_cex(t):
    a1, a2, eps, T = (float(CEX[k]) for k in ("a1", "a2", "eps", "T"))
    return a1 + (a2 - a1) * smoothstep((np.asarray(t, float) - eps) / (T - 2 * eps))


def v_cex(t):
    eps, T = float(CEX["eps"]), float(CEX["T"])
    t = np.asarray(t, float)
    return np.clip(np.minimum(t / eps, (T - t) / eps), 0.0, 1.0)


def certify_counterexample():
    iv.dps = CEX["dps"]
    a1, a2, eps, T = (iv.mpf(CEX[k]) for k in ("a1", "a2", "eps", "T"))
    M = CEX["P"]
    E_up = eps ** (1 - a1) / ((3 - a1) * iv.gamma(2 - a1))
    E_dn = -(eps ** (1 - a2)) / iv.gamma(4 - a2)
    p = 1 - a2

    def F1(t):   # int (T-t) t^p dt
        return T * t ** (p + 1) / (p + 1) - t ** (p + 2) / (p + 2)

    def F2(t):   # int (T-t) (t-eps)^p dt
        s = t - eps
        return (T - eps) * s ** (p + 1) / (p + 1) - s ** (p + 2) / (p + 2)

    E_cr = ((F1(T) - F1(T - eps)) - (F2(T) - F2(T - eps))) / (eps ** 2 * iv.gamma(2 - a2))
    t1, t2 = eps, T - eps

    def alpha_iv(t):
        x = (t - t1) / (t2 - t1)
        return a1 + (a2 - a1) * (3 * x ** 2 - 2 * x ** 3)

    # hold interval: for t in [s_j, s_{j+1}] and tau in [0, eps), 0 < t - tau < 1 and
    # alpha(t) in [a-, a+], hence (t-tau)^{-a-}/Gamma(1-a+) <= kappa(t,tau) <= (t-tau)^{-a+}/Gamma(1-a-)
    def G(t, q, first):
        return (t ** q) / q if first else (t ** q - (t - eps) ** q) / q

    Eh, Eh_lo = iv.mpf(0), iv.mpf(0)
    for j in range(M):
        sj = t1 + (t2 - t1) * iv.mpf(j) / M
        sj1 = t1 + (t2 - t1) * iv.mpf(j + 1) / M
        am = iv.mpf(alpha_iv(sj).a)      # lower bound of alpha on [s_j, s_{j+1}]
        ap = iv.mpf(alpha_iv(sj1).b)     # upper bound of alpha on [s_j, s_{j+1}]
        q = 2 - ap
        Eh += (G(sj1, q, False) - G(sj, q, j == 0)) / (eps * (1 - ap) * iv.gamma(1 - am))
        q = 2 - am
        Eh_lo += (G(sj1, q, False) - G(sj, q, j == 0)) / (eps * (1 - am) * iv.gamma(1 - ap))
    total = E_up + E_cr + E_dn + Eh
    total_lo = E_up + E_cr + E_dn + Eh_lo
    # accurate (non-certified) value of the hold energy, for comparison only
    mp.dps = 20
    a1f, a2f, epsf, Tf = (mp.mpf(CEX[k]) for k in ("a1", "a2", "eps", "T"))

    def al(t):
        x = min(max((t - epsf) / (Tf - 2 * epsf), 0), 1)
        return a1f + (a2f - a1f) * (3 * x ** 2 - 2 * x ** 3)

    def cur(t):
        a = al(t)
        return (t ** (1 - a) - (t - epsf) ** (1 - a)) / (epsf * mp.gamma(2 - a))

    Eh_acc = mp.quad(cur, mp.linspace(epsf, Tf - epsf, 41))
    out = dict(E1_up_ramp=(E_up.a, E_up.b), E3plus_cross=(E_cr.a, E_cr.b), E3minus_down_ramp=(E_dn.a, E_dn.b),
               E2_hold=(Eh_lo.a, Eh.b), E_total=(total_lo.a, total.b))
    return out, float(Eh_acc), float(mpmath.mpf(E_up.mid) + mpmath.mpf(E_cr.mid)
                                      + mpmath.mpf(E_dn.mid) + Eh_acc)


def iv_endpoint(x):
    """mpf value of a degenerate interval endpoint returned by mpmath.iv (x.a or x.b)."""
    mp.dps = 50
    return mpmath.mpf(x._mpi_[0])


def round_dir(x, digits, up):
    """exact decimal string of x rounded to `digits` decimals towards +inf (up) or -inf."""
    from decimal import Decimal
    mp.dps = 50
    s = mpmath.mpf(10) ** digits
    n = int(mpmath.ceil(x * s) if up else mpmath.floor(x * s))
    return str(Decimal(n).scaleb(-digits))


def discrete_counterexample(h):
    T = float(CEX["T"])
    N = int(round(T / h))
    t = h * np.arange(1, N + 1)
    H = build_H(alpha_cex, h, N, 1.0)
    v = v_cex(t)
    i = H @ v
    E = h * np.cumsum(v * i)
    return t, v, i, E, lam_min(energy_matrix(H, h))


# ----------------------------------------------------------------------------
# C. Rate bound of the corollary
# ----------------------------------------------------------------------------
def rate_bound_funcs(alo, ahi):
    psibar = -digamma(1 - ahi)
    Glo, Ghi = gamma(1 - alo), gamma(1 - ahi)

    def ghat(s):
        s = np.asarray(s, float)
        with np.errstate(divide="ignore"):
            return np.where(s <= 1, s ** (-alo), s ** (-ahi)) / Ghi

    def Nhat(l):
        l = np.asarray(l, float)
        a, b = ahi, alo
        ls = np.minimum(l, 1.0)
        n1 = psibar * ls ** (1 - a) / (1 - a) + ls ** (1 - a) * (1 / (1 - a) ** 2 - np.log(ls) / (1 - a))
        lb = np.maximum(l, 1.0)
        n2 = (psibar * (lb ** (1 - b) - 1) / (1 - b)
              + lb ** (1 - b) * (np.log(lb) / (1 - b) - 1 / (1 - b) ** 2) + 1 / (1 - b) ** 2)
        return (n1 + n2) / Glo

    def Ghat(s):
        s = np.asarray(s, float)
        m = np.where(s <= 1, s ** (-ahi), s ** (-alo))
        return m * (psibar + np.abs(np.log(s))) / Glo

    return ghat, Nhat, Ghat


def partition(T0, J):
    l = np.concatenate([[0.0], T0 * np.logspace(-12, 0, J // 2), np.linspace(0, T0, J // 2)])
    return np.unique(l[(l >= 0) & (l <= T0)])


def beta_star(T0, alo, ahi, J=200000):
    """Lower bound of inf_{0<l<T0} [ghat(l)+ghat(T0-l)]/Nhat(l) via monotonicity on a partition."""
    ghat, Nhat, _ = rate_bound_funcs(alo, ahi)
    l = partition(T0, J)
    lj, lj1 = l[:-1], l[1:]
    val = (ghat(lj1) + ghat(T0 - lj)) / Nhat(lj1)
    return float(val.min())


def beta_star_iv(T0, alo, ahi, J=4000):
    """Interval-arithmetic version of beta_star (certified lower bound)."""
    iv.dps = 30
    mp.dps = 40
    psibar = -mp.digamma(1 - mp.mpf(str(ahi)))          # from the decimal value, not the binary double
    psib = iv.mpf([psibar - mp.mpf("1e-30"), psibar + mp.mpf("1e-30")])
    A, B = iv.mpf(str(ahi)), iv.mpf(str(alo))
    Glo, Ghi = iv.gamma(1 - B), iv.gamma(1 - A)
    one = iv.mpf(1)

    def ghat(s):   # s is a point interval
        return (s ** (-B) if s.b <= 1 else s ** (-A)) / Ghi

    def Nhat(l):
        a, b = A, B
        if l.b <= 1:
            n = psib * l ** (1 - a) / (1 - a) + l ** (1 - a) * (1 / (1 - a) ** 2 - iv.log(l) / (1 - a))
        else:
            n1 = psib / (1 - a) + 1 / (1 - a) ** 2
            n2 = (psib * (l ** (1 - b) - 1) / (1 - b)
                  + l ** (1 - b) * (iv.log(l) / (1 - b) - 1 / (1 - b) ** 2) + 1 / (1 - b) ** 2)
            n = n1 + n2
        return n / Glo

    lpts = partition(float(T0), J)
    T0i = iv.mpf(str(T0))
    best = None
    for k in range(len(lpts) - 1):
        lj1 = iv.mpf(float(lpts[k + 1]))
        rem = T0i - iv.mpf(float(lpts[k]))
        if not rem.a > 0:
            continue
        # ghat is nonincreasing: a lower bound of ghat(T0 - l_k) is ghat at the UPPER endpoint
        num = ghat(lj1) + ghat(iv.mpf(rem.b))
        r = iv_endpoint((num / Nhat(lj1)).a)
        best = r if best is None or r < best else best
    f = float(best)
    if mpmath.mpf(f) > best:          # round the certified lower bound downwards
        f = float(np.nextafter(f, -np.inf))
    return f


def order_range(alpha, dalpha, T0, n=20001):
    """conservative range and rate of alpha on [0,T0] from n samples: the sampled min/max are widened by
    rate*delta/2 (delta = sample spacing), which bounds the deviation between samples.  For the orders used
    here |alpha'| attains its maximum at t = 0, which is a sample point."""
    ts = np.linspace(0, T0, n)
    rate = float(np.abs(dalpha(ts)).max())
    pad = rate * (T0 / (n - 1)) / 2
    return float(alpha(ts).min()) - pad, float(alpha(ts).max()) + pad, rate


def corollary_horizon(alpha, dalpha, Tmax=15.0, local=True, n=241):
    """Largest T0 (grid search plus bisection) such that Corollary 3.4 applies on [0,T0].
    local=True: order range and max|alpha'| taken on [0,T0]; local=False: taken on [0,Tmax]."""
    glo, ghi, grate = order_range(alpha, dalpha, Tmax, 200001)

    def ok(T0):
        lo, hi, rate = order_range(alpha, dalpha, T0) if local else (glo, ghi, grate)
        return beta_star(T0, lo, max(hi, lo + 1e-15), J=20000) >= rate

    grid = np.logspace(-7, np.log10(Tmax), n)
    grid[-1] = Tmax
    good = [i for i, T0 in enumerate(grid) if ok(T0)]
    if not good:
        return 0.0
    i = max(good)
    if i == len(grid) - 1:
        return Tmax
    a, b = grid[i], grid[i + 1]
    for _ in range(30):                      # refine the crossing by bisection in log T0
        m = np.sqrt(a * b)
        a, b = (m, b) if ok(m) else (a, m)
    return float(a)


# ----------------------------------------------------------------------------
# H. Direct evaluation of the kernel condition p_T >= 0 (Theorem 3.2)
# ----------------------------------------------------------------------------
class Order:
    """order alpha(t) with derivative; breaks = kinks of alpha' (panel boundaries)."""

    def __init__(self, f, df, amax, breaks=()):
        self.f, self.df, self.amax, self.breaks = f, df, amax, tuple(breaks)

    def coeffs(self, t):
        from scipy.special import rgamma
        a = np.asarray(self.f(t), float)
        return a, np.asarray(self.df(t), float) * rgamma(1.0 - a), digamma(1.0 - a)

    def kappa(self, t, tau):
        from scipy.special import rgamma
        a = np.asarray(self.f(t), float)
        return (np.asarray(t, float) - tau) ** (-a) * rgamma(1.0 - a)


def _gl01(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def _m_exponent(amax):
    # substitution s = s1 x^m on the singular panel; x-exponent m(1-a)-1 >= 5
    return float(max(12, int(np.ceil(6.0 / (1.0 - amax)))))


def pT_grid(order, T0, dt, n=10, n1=40):
    """p_T(tau) on the triangle tau_j = j dt < T_k = k dt <= T0 from one cumulative integral per tau.
    J_T(tau) = int_0^{T-tau} alpha'(tau+s) phi(tau+s,s)[psi(1-alpha(tau+s)) - ln s] ds; Gauss-Legendre
    (n points) on regular panels, substitution s = dt x^m and n1 points on the first panel."""
    K = int(round(T0 / dt))
    dt = T0 / K
    m = _m_exponent(order.amax)
    y, wq = _gl01(n)
    x1, w1 = _gl01(n1)
    W1 = dt * m * x1 ** (m - 1) * w1
    lnS1 = np.log(dt) + m * np.log(x1)
    idx = np.arange(K, dtype=float)[:, None]
    A, C, Ps = order.coeffs((idx + y[None, :]) * dt)
    A1, C1, Ps1 = order.coeffs((idx + (x1 ** m)[None, :]) * dt)
    P0 = np.sum(W1[None, :] * C1 * np.exp(-A1 * lnS1[None, :]) * (Ps1 - lnS1[None, :]), axis=1)
    tg = dt * np.arange(K + 1)
    ag = np.asarray(order.f(tg), float)
    rg = 1.0 / gamma(1.0 - ag)
    kap0 = np.full(K + 1, np.inf)
    kap0[1:] = np.exp(-ag[1:] * np.log(tg[1:])) * rg[1:]
    G = np.zeros(K + 1)
    minT = np.full(K + 1, np.inf)
    best, bj, bl = np.inf, -1, -1
    for l in range(1, K):
        if l == 1:
            G[:K] += P0[:K]
        else:
            lnS = np.log((l - 1 + y) * dt)
            rows = slice(l - 1, K)
            G[:K - l + 1] += dt * np.sum(wq[None, :] * C[rows] * np.exp(-A[rows] * lnS[None, :])
                                         * (Ps[rows] - lnS[None, :]), axis=1)
        k = np.arange(1 + l, K + 1)
        j = k - l
        p = np.exp(-ag[k] * np.log(l * dt)) * rg[k] + kap0[j] - G[j]
        im = int(np.argmin(p))
        if p[im] < best:
            best, bj, bl = float(p[im]), int(j[im]), l
        minT[k] = np.minimum(minT[k], p)
    return dict(dt=dt, p_min=best, tau=bj * dt, T=(bj + bl) * dt, tg=tg, minT=minT)


def pT_point(order, tau, T, ds=0.01, n=10, n1=40):
    """p_T(tau) at an arbitrary point, same quadrature as pT_grid (panels also split at kinks)."""
    L = T - tau
    m = _m_exponent(order.amax)
    kinks = sorted(b - tau for b in order.breaks if 0.0 < b - tau < L)
    s1 = min([ds, L] + kinks[:1])
    x1, w1 = _gl01(n1)
    lns1 = np.log(s1) + m * np.log(x1)
    a, c, ps = order.coeffs(tau + np.exp(lns1))
    J = np.sum(s1 * m * x1 ** (m - 1) * w1 * c * np.exp(-a * lns1) * (ps - lns1))
    if L > s1:
        br = np.unique(np.concatenate([np.arange(s1, L, ds), kinks, [L]]))
        br = br[br >= s1]
        lo, hi = br[:-1], br[1:]
        keep = hi - lo > 1e-15
        lo, hi = lo[keep], hi[keep]
        x, w = _gl01(n)
        s = lo[:, None] + (hi - lo)[:, None] * x[None, :]
        a, c, ps = order.coeffs(tau + s)
        ls = np.log(s)
        J += np.sum((hi - lo)[:, None] * w[None, :] * c * np.exp(-a * ls) * (ps - ls))
    return float(order.kappa(T, tau) + order.kappa(tau, 0.0) - J)


def pT_min_polished(order, T0, dt, ds=0.01):
    """grid minimum of p_T over the triangle, then a bounded Nelder-Mead polish in (T, tau/T)."""
    from scipy.optimize import minimize
    r = pT_grid(order, T0, dt)
    x0 = np.array([r["T"], r["tau"] / r["T"]])
    res = minimize(lambda z: pT_point(order, z[1] * z[0], z[0], ds=ds), x0, method="Nelder-Mead",
                   bounds=[(max(2 * r["dt"], r["T"] - 5 * r["dt"]), T0), (1e-9, 1 - 1e-9)],
                   options=dict(xatol=1e-10, fatol=1e-14, maxiter=800))
    if res.fun < r["p_min"]:
        return float(res.fun), float(res.x[1] * res.x[0]), float(res.x[0]), r
    return r["p_min"], r["tau"], r["T"], r


def pT_horizon(order, T0, dt):
    """largest T such that min_tau p_T(tau) >= 0 for all T' <= T (linear interpolation of the grid crossing)."""
    r = pT_grid(order, T0, dt)
    minT, tg = r["minT"], r["tg"]
    neg = np.nonzero(minT[2:] < 0)[0]
    if neg.size == 0:
        return T0
    k = neg[0] + 2
    m0, m1 = minT[k - 1], minT[k]
    return float(tg[k - 1] + (tg[k] - tg[k - 1]) * m0 / (m0 - m1)) if np.isfinite(m0) else float(tg[k - 1])


def dalpha_cex(t):
    a1, a2, eps, T = (float(CEX[k]) for k in ("a1", "a2", "eps", "T"))
    x = (np.asarray(t, float) - eps) / (T - 2 * eps)
    return np.where((x >= 0) & (x <= 1), (a2 - a1) * (6 * x - 6 * x * x) / (T - 2 * eps), 0.0)


def order_w(w):
    """sweep family alpha_w(t) = 0.70 + 0.20 sin(w t)."""
    return (lambda t: 0.70 + 0.20 * np.sin(w * np.asarray(t, float)),
            lambda t: 0.20 * w * np.cos(w * np.asarray(t, float)))


def range_w(w, T0=15.0):
    """exact range of alpha_w = 0.70 + 0.20 sin(w t) on [0,T0] and max|alpha_w'| = 0.20 w (at t = 0)."""
    x = w * T0
    smax = 1.0 if x >= np.pi / 2 else np.sin(x)
    smin = -1.0 if x >= 3 * np.pi / 2 else (np.sin(x) if x > np.pi else 0.0)
    # widen by 1e-14 to absorb floating-point rounding of the endpoints (conservative for the corollary)
    return 0.70 + 0.20 * smin - 1e-14, 0.70 + 0.20 * smax + 1e-14, 0.20 * w


# ----------------------------------------------------------------------------
# I. Mass matrix, exact continuous energies of piecewise-linear voltages, port form
# ----------------------------------------------------------------------------
def mass_matrix(h, N):
    """P1 mass matrix for hat functions phi_1..phi_N on [0, N h] with v_0 = 0."""
    M = np.diag(np.full(N, 2 * h / 3))
    M[-1, -1] = h / 3
    i = np.arange(N - 1)
    M[i, i + 1] = h / 6
    M[i + 1, i] = h / 6
    return M


def lam_min_pencil(S, M):
    return float(eigh(S, M, eigvals_only=True, subset_by_index=[0, 0])[0])


def is_pd(S):
    from scipy.linalg import cholesky, LinAlgError
    try:
        cholesky(S, lower=True, check_finite=False)
        return True
    except LinAlgError:
        return False


def _gl_sub(n, q):
    x, w = _gl01(n)
    return x ** q, w * q * x ** (q - 1)


def continuous_energy(alpha, hc, vnodes, cv, n=20, q=8, chunk=4000):
    """E(T) = int_0^T v_h(t) cv D^{alpha(t)} v_h(t) dt for the piecewise-linear v_h with nodal values
    vnodes on t_k = k hc (v_0 = 0). The current is evaluated in closed form,
    iota(t) = cv sum_k d_k (t - t_k)_+^{1-alpha(t)} / Gamma(2 - alpha(t)), d_k = slope jumps;
    on every interval t = t_k + hc x^q removes the (t - t_k)^{1-alpha} endpoint singularity."""
    N = len(vnodes)
    tk = hc * np.arange(0, N + 1)
    vv = np.concatenate([[0.0], vnodes])
    s = np.diff(vv) / hc
    d = np.concatenate([[s[0]], np.diff(s)])
    xs, ws = _gl_sub(n, q)
    tq = (tk[:-1, None] + hc * xs[None, :]).ravel()
    wq = np.tile(hc * ws, N)
    vq = np.interp(tq, tk, vv)
    iq = np.empty_like(tq)
    for a0 in range(0, len(tq), chunk):
        tt = tq[a0:a0 + chunk]
        a = alpha(tt)
        D = tt[:, None] - tk[None, :-1]
        B = np.where(D > 0, np.exp((1 - a)[:, None] * np.log(np.where(D > 0, D, 1.0))), 0.0)
        iq[a0:a0 + chunk] = cv * (B @ d) / gamma(2 - a)
    return float(np.sum(wq * vq * iq)), float(np.sum(wq * vq * vq))


def galerkin_energy_matrix(alpha, h, N, cv, n=16, q=8, block=200):
    """exact continuous energy matrix A_{mj} = int_0^{Nh} phi_m cv D^{alpha(t)} phi_j dt on P1 (v_0 = 0).
    phi_j has slopes 1/h, -1/h, so cv D^{alpha(t)} phi_j(t) = cv [B(t-t_{j-1}) - 2B(t-t_j) + B(t-t_{j+1})]/h
    with B(s) = s_+^{1-alpha(t)}/Gamma(2-alpha(t)) (last function: half hat).  Quadrature: on each
    interval t = t_{m-1} + h x^q with n Gauss-Legendre points (removes the endpoint singularity)."""
    tk = h * np.arange(0, N + 1)
    xs, ws = _gl_sub(n, q)
    A = np.zeros((N, N))
    for m0 in range(1, N + 1, block):
        ms = np.arange(m0, min(m0 + block, N + 1))                    # intervals (t_{m-1}, t_m]
        tt = (tk[ms - 1][:, None] + h * xs[None, :]).ravel()
        a = alpha(tt)
        D = tt[:, None] - tk[None, :]
        B = np.where(D > 0, np.exp((1 - a)[:, None] * np.log(np.where(D > 0, D, 1.0))), 0.0) \
            / gamma(2 - a)[:, None]
        I = np.empty((len(tt), N))
        I[:, :N - 1] = B[:, 0:N - 1] - 2 * B[:, 1:N] + B[:, 2:N + 1]
        I[:, N - 1] = B[:, N - 1] - B[:, N]
        W = (np.tile(h * ws, len(ms))[:, None] * I * (cv / h)).reshape(len(ms), n, N)
        A[ms - 1] += np.einsum("q,mqj->mj", xs, W)                   # test function phi_m = x
        right = ms >= 2
        A[ms[right] - 2] += np.einsum("q,mqj->mj", 1 - xs, W[right])  # test function phi_{m-1} = 1 - x
    return A


def galerkin_lambda(alpha, h, T, cv):
    N = int(round(T / h))
    A = galerkin_energy_matrix(alpha, h, N, cv)
    return lam_min_pencil(0.5 * (A + A.T), mass_matrix(h, N))


def port_lambda(H, h):
    """smallest eigenvalue / h of the discrete terminal energy form of the circuit,
    E_port = h sum u_n i_n = v^T h [R_s K^T K + (K + K^T)/2] v with K = H + I/R_p."""
    K = H + np.eye(H.shape[0]) / RP
    P = RS * (K.T @ K) + 0.5 * (K + K.T)
    return float(eigh(P, eigvals_only=True, subset_by_index=[0, 0])[0])


# ----------------------------------------------------------------------------
# F. Verification helpers
# ----------------------------------------------------------------------------
def verify_identities():
    rows = []
    al, dal = ORDERS["rapid"]["f"], ORDERS["rapid"]["df"]

    def kappa(t, tau):
        a = al(t)
        return (t - tau) ** (-a) / gamma(1 - a)

    # (1) kernel logarithmic derivative identity
    err = 0.0
    for (t, tau) in [(0.7, 0.2), (1.3, 1.25), (4.1, 0.5), (9.0, 2.0)]:
        d = 1e-6
        fd = (np.log(kappa(t + d, tau)) - np.log(kappa(t - d, tau))) / (2 * d)
        a = al(t)
        fo = dal(t) * digamma(1 - a) - dal(t) * np.log(t - tau) - a / (t - tau)
        err = max(err, abs(fd - fo) / abs(fo))
    rows.append(["d/dt ln kappa identity (finite differences, rapid order, 4 points with t <= 9)", f"{err:.3e}"])
    # (2) shift derivative (d/dt + d/dtau) kappa
    err = 0.0
    for (t, tau) in [(0.7, 0.2), (1.3, 1.25), (4.1, 0.5), (9.0, 2.0)]:
        d = 1e-6
        fd = (kappa(t + d, tau + d) - kappa(t - d, tau - d)) / (2 * d)
        a = al(t)
        fo = dal(t) * kappa(t, tau) * (digamma(1 - a) - np.log(t - tau))
        err = max(err, abs(fd - fo) / abs(fo))
    rows.append(["(d/dt + d/dtau) kappa identity (finite differences, rapid order, 4 points with t <= 9)", f"{err:.3e}"])

    # (3) derivative of Q_T and the energy identity of the theorem
    T = 1.5
    v = lambda s: np.sin(3 * s) + 0.5 * s ** 2
    dv = lambda s: 3 * np.cos(3 * s) + s
    opts = dict(limit=400, epsabs=1e-13, epsrel=1e-11)

    def J(tau):   # int_tau^T alpha' kappa [psi - ln(t-tau)] dt, substitution s = (T-tau) x^12
        L = T - tau
        if L <= 0:
            return 0.0
        m = 12.0

        def f(x):
            if x <= 0:
                return 0.0
            s = L * x ** m
            t = tau + s
            a = al(t)
            return dal(t) * s ** (-a) / gamma(1 - a) * (digamma(1 - a) - np.log(s)) * L * m * x ** (m - 1)

        return quad(f, 0, 1, **opts)[0]

    def Q(tau):   # int_tau^T kappa(t,tau) dt, same substitution
        L = T - tau
        m = 12.0

        def f(x):
            if x <= 0:
                return 0.0
            s = L * x ** m
            a = al(tau + s)
            return s ** (-a) / gamma(1 - a) * L * m * x ** (m - 1)

        return quad(f, 0, 1, **opts)[0]

    err = 0.0
    for tau in [0.1, 0.6, 1.2]:
        d = 1e-5
        fd = (Q(tau + d) - Q(tau - d)) / (2 * d)
        fo = -kappa(T, tau) + J(tau)
        err = max(err, abs(fd - fo) / abs(fo))
    rows.append(["Q_T'(tau) = -kappa(T,tau) + int (d_t+d_tau)kappa (T=1.5)", f"{err:.3e}"])
    # p_T evaluator of Section H (Gauss-Legendre panels) versus adaptive quadrature of J
    o = Order(al, dal, ORDERS["rapid"]["hi"])
    err = 0.0
    for tau in [0.01, 0.1, 0.6, 1.2, 1.49]:
        pq = kappa(T, tau) + kappa(tau, 0.0) - J(tau)
        err = max(err, abs(pT_point(o, tau, T) - pq) / abs(pq))
    rows.append(["p_T evaluator (Gauss panels) versus adaptive quadrature (T=1.5)", f"{err:.3e}"])
    # exact P1 energy matrix versus closed-form continuous energy of a piecewise-linear voltage
    hG, NG = 0.05, 30
    vG = np.sin(3 * hG * np.arange(1, NG + 1)) + 0.5 * (hG * np.arange(1, NG + 1)) ** 2
    AG = galerkin_energy_matrix(al, hG, NG, 1.0, n=24, q=10)
    EG, _ = continuous_energy(al, hG, vG, 1.0, n=40, q=12)
    rows.append(["P1 energy matrix v^T A v versus closed-form continuous energy (T=1.5)",
                 f"{abs(vG @ AG @ vG - EG) / abs(EG):.2e}"])

    def iota(t):
        a = al(t)
        return quad(dv, 0, t, weight="alg", wvar=(0.0, -a), **opts)[0] / gamma(1 - a)

    lhs = quad(lambda t: v(t) * iota(t), 0, T, **opts)[0]
    aT = al(T)
    t1 = 0.5 * quad(lambda s: v(s) ** 2, 0, T, weight="alg", wvar=(0.0, -aT), **opts)[0] / gamma(1 - aT)
    t2 = 0.5 * quad(lambda s: s ** (-al(s)) / gamma(1 - al(s)) * v(s) ** 2 if s > 0 else 0.0, 0, T, **opts)[0]
    t3 = -0.5 * quad(lambda s: J(s) * v(s) ** 2, 0, T, **opts)[0]

    def inner(t):
        a = al(t)

        def f(s):
            return ((v(t) - v(s)) / (t - s)) ** 2 if t - s > 1e-12 else dv(t) ** 2

        return a / gamma(1 - a) * quad(f, 0, t, weight="alg", wvar=(0.0, 1.0 - a), **opts)[0]

    t4 = 0.5 * quad(inner, 0, T, **opts)[0]
    rhs = t1 + t2 + t3 + t4
    rows.append(["energy identity: int v D v dt (direct quadrature)", f"{lhs:.12f}"])
    rows.append(["energy identity: (1/2)int p_T v^2 + (1/2)int int d_tau kappa (v(t)-v(tau))^2", f"{rhs:.12f}"])
    rows.append(["energy identity: relative difference", f"{abs(lhs - rhs) / abs(lhs):.3e}"])

    # (4) L1 exactness for a linear voltage and S_h quadratic form
    h, N = 0.01, 300
    H = build_H(al, h, N, 1.0)
    tt = h * np.arange(1, N + 1)
    ex = tt ** (1 - al(tt)) / gamma(2 - al(tt))
    rows.append(["L1 exactness for v(t)=t (max relative error at nodes)",
                 f"{np.max(np.abs(H @ tt - ex) / ex):.2e}"])
    rng = np.random.default_rng(1)
    x = rng.standard_normal(N)
    S = energy_matrix(H, h)
    rows.append(["v^T S v versus h*sum v_n (Hv)_n (relative difference)",
                 f"{abs(x @ S @ x - h * x @ (H @ x)) / abs(h * x @ (H @ x)):.2e}"])
    rows.append(["H lower triangular (max |upper part|)", f"{np.max(np.abs(np.triu(H, 1))):.1e}"])
    # (5) closed form of Nhat versus quadrature
    ghat, Nhat, Ghat = rate_bound_funcs(0.5, 0.9)
    err = 0.0
    for l in [0.3, 1.0, 4.0, 15.0]:
        m, a = 20.0, min(l, 1.0)   # s = a x^m removes the endpoint singularity at s = 0
        qv = quad(lambda x: float(Ghat(a * x ** m)) * a * m * x ** (m - 1) if x > 0 else 0.0,
                  0, 1, limit=400, epsabs=1e-13, epsrel=1e-12)[0]
        if l > 1:
            qv += quad(lambda s: float(Ghat(s)), 1, l, limit=400, epsabs=1e-13, epsrel=1e-12)[0]
        err = max(err, abs(qv - float(Nhat(l))) / qv)
    rows.append(["closed form of N_hat versus quadrature (max relative error)", f"{err:.3e}"])
    return rows


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()
    env = [["python", platform.python_version()], ["numpy", np.__version__],
           ["scipy", scipy.__version__], ["mpmath", mpmath.__version__],
           ["matplotlib", matplotlib.__version__], ["platform", platform.platform()],
           ["processor", platform.processor()]]
    write_csv("environment.csv", ["item", "value"], env)

    # ---------------- B. counterexample ----------------
    print("B: certified counterexample ...", flush=True)
    cert, Eh_acc, Etot_acc = certify_counterexample()
    rows = []
    for k, (lo_iv, hi_iv) in cert.items():
        lo_, hi_ = iv_endpoint(lo_iv), iv_endpoint(hi_iv)
        rows.append([k, round_dir(lo_, 12, up=False), round_dir(hi_, 12, up=True),
                     round_dir(lo_, 7, up=False), round_dir(hi_, 7, up=True), ""])
    rows.append(["E2_hold_quadrature (tanh-sinh, not certified)", "", "", "", "", f"{Eh_acc:.12f}"])
    rows.append(["E_total_quadrature (tanh-sinh, not certified)", "", "", "", "", f"{Etot_acc:.12f}"])
    write_csv("counterexample_certified.csv",
              ["quantity per unit C_v (interval arithmetic, 30 digits, P=%d hold subintervals; outward rounded)" % CEX["P"],
               "lower_12dp", "upper_12dp", "lower_7dp", "upper_7dp", "quadrature_value"], rows)
    cex_disc = []
    for h in [1e-3, 5e-4, 2.5e-4]:
        t, v, i, E, lm = discrete_counterexample(h)
        cex_disc.append([h, len(t), E[-1], lm, lm / h])
    write_csv("counterexample_discrete.csv", ["h", "N", "E_h(T)", "lambda_min(S_h)", "lambda_min/h"], cex_disc)
    write_csv("counterexample_series.csv", ["t", "alpha", "v", "iota_over_Cv", "E_h_over_Cv"],
              np.column_stack([t, alpha_cex(t), v, i, E]).tolist())
    print("   certified enclosure E(T)/C_v in [", rows[4][1], ",", rows[4][2], "]", flush=True)

    # ---------------- C. rate bound ----------------
    print("C: rate bound ...", flush=True)
    T0s = np.logspace(-6, np.log10(15.0), 121)
    beta_rows, curves = [], {}
    for case in ("slow", "rapid"):
        al, dal = ORDERS[case]["f"], ORDERS[case]["df"]
        for T0 in T0s:
            lo, hi, rate = order_range(al, dal, T0)
            bb = beta_star(T0, lo, max(hi, lo + 1e-15), J=40000)
            beta_rows.append([case, lo, hi, T0, bb, rate])
    write_csv("rate_bound_curves.csv",
              ["case", "alpha_lo_on_0_T0", "alpha_hi_on_0_T0", "T0", "beta_star_lower_bound", "max_abs_dalpha_on_0_T0"],
              beta_rows)
    horizon_rows = []
    for case in ("slow", "rapid"):
        al, dal = ORDERS[case]["f"], ORDERS[case]["df"]
        Tg = corollary_horizon(al, dal, local=False)
        Tl = corollary_horizon(al, dal, local=True)
        horizon_rows.append([f"{case}: range and rate on [0,15]", ORDERS[case]["lo"], ORDERS[case]["hi"],
                             ORDERS[case]["dmax"], "horizon_T0", Tg])
        horizon_rows.append([f"{case}: range and rate on [0,T0]", "", "", "", "horizon_T0", Tl])
    b15_float = beta_star(15.0, 0.70, 0.73)
    b15_iv = beta_star_iv(15.0, 0.70, 0.73)
    horizon_rows.append(["slow: beta_*(15), double precision, J=2e5", 0.70, 0.73, 0.006, "beta_star", b15_float])
    horizon_rows.append(["slow: beta_*(15), interval arithmetic lower bound, J=4e3", 0.70, 0.73, 0.006,
                         "beta_star", b15_iv])
    b_cex = beta_star(float(CEX["T"]), float(CEX["a1"]), float(CEX["a2"]))
    horizon_rows.append(["counterexample order: beta_*(0.1), double precision lower bound", CEX["a1"], CEX["a2"],
                         15.0, "beta_star", b_cex])
    write_csv("rate_bound_horizons.csv",
              ["quantity", "alpha_lo", "alpha_hi", "max_abs_dalpha", "kind", "value"], horizon_rows)
    print(f"   beta*(15; 0.70,0.73) = {b15_float:.7f} (float), >= {b15_iv:.7f} (interval)", flush=True)
    for r in horizon_rows[:4]:
        print("   corollary", r[0], "->", f"{r[5]:.4g}", flush=True)

    # ---------------- D/E. circuit and certificate ----------------
    print("D/E: circuit simulations and certificates ...", flush=True)
    en_rows, cert_rows, fine = [], [], {}
    store = {c: {} for c in CASES}
    for case in CASES:
        al = ORDERS[case]["f"]
        for h in GRIDS:
            N = int(round(T_END / h))
            t0 = time.perf_counter()
            H = build_H(al, h, N, CV)
            tA0 = time.perf_counter()
            A = circuit_L1(al, h, H)
            tA = time.perf_counter() - tA0
            tH = tA0 - t0
            t1 = time.perf_counter()
            B = circuit_GL(al, h)
            tB = time.perf_counter() - t1
            t2 = time.perf_counter()
            S = energy_matrix(H, h)
            lm = lam_min(S)
            tE = time.perf_counter() - t2
            t3 = time.perf_counter()
            lmM = lam_min_pencil(S, mass_matrix(h, N))     # min E_h / ||v_h||_{L2}^2 (P1 mass matrix)
            tEM = time.perf_counter() - t3
            # terminal (port) energy form of the whole circuit; element vs port passivity
            pl = port_lambda(H, h) if (case == "rapid" or h == GRIDS[0]) else np.nan
            norm1 = float(np.max(np.sum(np.abs(S), axis=0)))
            guard = N * U_ROUND * norm1
            # discrete passivity horizon: largest M with lambda_min(S[:M,:M]) >= 0
            # (lambda_min of the leading block is nonincreasing in M by Cauchy interlacing)
            if lm >= 0:
                Mstar = N
            else:
                lo_, hi_ = 0, N
                while hi_ - lo_ > 1:
                    mid = (lo_ + hi_) // 2
                    if lam_min(S[:mid, :mid]) >= 0:
                        lo_ = mid
                    else:
                        hi_ = mid
                Mstar = lo_
            EA, EB = A["E"], B["E"]
            store[case][h] = dict(EA=EA, EB=EB, lm=lm, lmM=lmM)
            after_charge = A["t"] >= 5.0
            en_rows.append([case, h, N, EA["sup"][-1], EA["Rs"][-1], EA["Rp"][-1], EA["frac"][-1],
                            EA["frac"][after_charge].min(), np.abs(A["u"]).max(),
                            EB["sup"], EB["Rs"], EB["Rp"], EB["frac"], tH, tA, tB])
            cert_rows.append([case, h, N, lm, lm / h, lmM, norm1, guard, Mstar * h, pl, tE, tEM])
            print(f"   {case:8s} h={h:<7} Esup={EA['sup'][-1]:.6f} (GL {EB['sup']:.6f}) "
                  f"Efrac={EA['frac'][-1]:.6f} lmin={lm:.4e} lmin(S,M)={lmM:.5f} t*={Mstar*h:.3f} port={pl:.4f} "
                  f"[H {tH:.2f}s, L1 {tA:.2f}s, GL {tB:.2f}s, eig {tE:.2f}s, pencil {tEM:.2f}s]", flush=True)
            if h == GRIDS[-1]:
                fine[case] = A
            if case == "rapid" and h in (GRIDS[0], GRIDS[-1]):
                w, V = eigh(S, subset_by_index=[0, 0])
                vec = V[:, 0] / np.sqrt(h)                 # normalized so that h*|v|^2 = 1
                vec = vec * np.sign(vec[np.argmax(np.abs(vec))])
                tt = h * np.arange(0, N + 1)
                vv = np.concatenate([[0.0], vec])
                if h == GRIDS[0]:
                    test_voltage = (tt, vv)
                else:
                    write_csv("eigvec_rapid.csv", ["t", "normalized_minimizing_eigenvector", "alpha_r"],
                              np.column_stack([tt, vv, ORDERS["rapid"]["f"](tt)]).tolist())
    write_csv("circuit_energies.csv",
              ["case", "h", "N", "A_E_sup", "A_E_Rs", "A_E_Rp", "A_E_frac", "A_min_E_frac_after_charge_t_ge_5",
               "A_max_abs_u", "B_E_sup", "B_E_Rs", "B_E_Rp", "B_E_frac_residual", "time_build_H_s",
               "time_L1_solve_s", "time_GL_s"], en_rows)
    write_csv("certificate.csv",
              ["case", "h", "N", "lambda_min_S", "lambda_min_over_h", "lambda_min_pencil_S_M", "norm1_S",
               "rounding_scale_N_u_norm1", "discrete_passivity_horizon", "port_lambda_min_over_h",
               "time_eigen_s", "time_pencil_s"], cert_rows)

    # Richardson extrapolation
    rich_rows = []
    for case in CASES:
        for q in ("sup", "frac"):
            eA = [store[case][h]["EA"][q][-1] for h in GRIDS]
            eB = [store[case][h]["EB"][q] for h in GRIDS]
            pA, xA = richardson(*eA)
            pB, xB = richardson(*eB)
            rich_rows.append([case, q, pA, xA, pB, xB, abs(xA - xB), abs(eA[-1] - eB[-1])])
    write_csv("richardson.csv", ["case", "energy", "order_A_L1", "extrapolated_A", "order_B_GL",
                                 "extrapolated_B", "abs_diff_extrapolated", "abs_diff_finest"], rich_rows)

    # fixed test voltage: minimizing direction of the coarsest grid (rapid order), used as a fixed
    # piecewise-linear voltage on all grids (exactly representable, so L1 currents are exact at nodes)
    tv_rows = []
    tc, vc = test_voltage
    for h in GRIDS:
        N = int(round(T_END / h))
        t = h * np.arange(1, N + 1)
        vv = np.interp(t, tc, vc)
        H = build_H(ORDERS["rapid"]["f"], h, N, CV)
        ii = H @ vv
        e_right = h * vv @ ii
        e_trap = e_right - 0.5 * h * vv[-1] * ii[-1]      # trapezoidal rule; v_0 = 0 removes the first node
        tv_rows.append([h, e_right, h * vv @ vv, e_trap])
    # exact continuous energy of the same fixed piecewise-linear voltage (closed-form current)
    alr = ORDERS["rapid"]["f"]
    Ec1, L2c = continuous_energy(alr, tc[1] - tc[0], vc[1:], CV, n=20, q=8)
    Ec2, _ = continuous_energy(alr, tc[1] - tc[0], vc[1:], CV, n=40, q=12)
    tv_rows.append(["continuous (n=20, q=8)", Ec1, L2c, ""])
    tv_rows.append(["continuous (n=40, q=12)", Ec2, L2c, ""])
    write_csv("test_voltage_rapid.csv",
              ["h or quadrature", "E(15) of fixed piecewise-linear voltage [J] (right-endpoint rule for h rows)",
               "h*|v|^2 or ||v_h||_L2^2", "E(15) with trapezoidal energy quadrature [J]"], tv_rows)
    print(f"   fixed voltage: discrete E_h = {', '.join(f'{r[1]:.4f}' for r in tv_rows[:3])}; "
          f"continuous E = {Ec2:.10f} (quadrature difference {abs(Ec1 - Ec2):.1e})", flush=True)

    # ---------------- J. exact P1 (Galerkin) energies: Rayleigh quotients and horizon ----------------
    print("J: exact P1 energy matrix (Galerkin) for alpha_r ...", flush=True)
    gal_rows = []
    for h in (0.01, 0.005):
        t0 = time.perf_counter()
        lg = galerkin_lambda(alr, h, T_END, CV)
        gal_rows.append(["min over P1 of E(15)/||v||_L2^2", h, lg, time.perf_counter() - t0])
        print(f"   h={h}: min_P1 E(15)/||v||^2 = {lg:.6f}", flush=True)
    for h in (0.01, 0.005):
        # scan (no monotonicity assumed; the P1 spaces on [0,T] are not nested because of the half hat)
        Ms = np.unique(np.concatenate([np.arange(int(round(0.2 / h)), int(round(1.5 / h)), int(round(0.05 / h))),
                                       np.arange(int(round(1.5 / h)), int(round(3.0 / h)) + 1)]))
        vals = np.array([galerkin_lambda(alr, h, M * h, CV) for M in Ms])
        kneg = int(np.nonzero(vals < 0)[0][0])
        gal_rows.append(["P1 exact-energy scan: last scanned T with min >= 0 before the first negative one", h,
                         Ms[kneg - 1] * h, ""])
        gal_rows.append(["P1 exact-energy scan: first scanned T with min < 0", h, Ms[kneg] * h, ""])
        gal_rows.append(["P1 exact-energy scan: min < 0 at every scanned T from there to 3.0", h,
                         int(np.all(vals[kneg:] < 0)), ""])
        print(f"   h={h}: P1 exact energy first negative at T={Ms[kneg] * h:.3f} (nonnegative at "
              f"{Ms[kneg - 1] * h:.3f}); negative at all later scanned T: {bool(np.all(vals[kneg:] < 0))}",
              flush=True)
    write_csv("galerkin_rapid.csv", ["quantity", "h", "value", "time_s"], gal_rows)

    # ---------------- H. kernel condition p_T >= 0 evaluated directly ----------------
    print("H: kernel condition p_T >= 0 on the triangle ...", flush=True)
    eps_c, T_c, a2_c = float(CEX["eps"]), float(CEX["T"]), float(CEX["a2"])
    ords = {c: (Order(ORDERS[c]["f"], ORDERS[c]["df"], ORDERS[c]["hi"]), 15.0, 0.005, 0.01) for c in CASES}
    ords["counterexample"] = (Order(alpha_cex, dalpha_cex, a2_c, breaks=(eps_c, T_c - eps_c)), T_c, 5e-5, 1e-3)
    kc_rows = []
    for name, (o, T0, dt, ds) in ords.items():
        t0 = time.perf_counter()
        pmin, taus, Ts, r = pT_min_polished(o, T0, dt, ds=ds)
        if name == "rapid":
            Tp = pT_horizon(o, 0.4, 2.5e-4)
        elif r["minT"][2:].min() >= 0:
            Tp = T0
        else:
            Tp = pT_horizon(o, T0, dt)
        kc_rows.append([name, T0, dt, r["p_min"], pmin, taus, Ts, Tp, 0.5 * CV * pmin, time.perf_counter() - t0])
        print(f"   {name:14s} p_min = {pmin:.6f} at (tau, T) = ({taus:.4f}, {Ts:.4f}); "
              f"kernel-condition horizon = {Tp:.5f}", flush=True)
    write_csv("kernel_condition.csv",
              ["order", "T0", "grid_dt", "p_min_grid", "p_min_polished", "tau_star", "T_star",
               "kernel_condition_horizon", "Cv_over_2_times_p_min with Cv=10 (lower bound of inf E/||v||^2 at T0)",
               "time_s"],
              kc_rows)

    # ---------------- I. criteria sweep alpha_w = 0.70 + 0.20 sin(w t) ----------------
    print("I: criteria sweep over w ...", flush=True)
    ws = np.unique(np.concatenate([np.logspace(-2, 1, 24), [8.0]]))
    sweep_rows = []

    def crit_beta(w):
        lo, hi, rate = range_w(w)
        return beta_star(15.0, lo, max(hi, lo + 1e-15), J=40000) >= rate

    def pmin_w(w):
        f, df = order_w(w)
        return pT_min_polished(Order(f, df, 0.9), 15.0, 0.01)[0]

    def crit_lambda(w, h):
        f, _ = order_w(w)
        N = int(round(T_END / h))
        return is_pd(energy_matrix(build_H(f, h, N, CV), h))

    def p1_w(w):
        f, _ = order_w(w)
        return galerkin_lambda(f, GRIDS[0], T_END, CV)

    for w in ws:
        f, df = order_w(w)
        lo, hi, rate = range_w(w)
        b = beta_star(15.0, lo, max(hi, lo + 1e-15), J=40000)
        pm = pmin_w(w)
        N = int(round(T_END / GRIDS[0]))
        lmM = lam_min_pencil(energy_matrix(build_H(f, GRIDS[0], N, CV), GRIDS[0]), mass_matrix(GRIDS[0], N))
        pd2, pd3 = int(crit_lambda(w, GRIDS[1])), int(crit_lambda(w, GRIDS[2]))
        gl = p1_w(w)
        sweep_rows.append([w, lo, hi, rate, b, int(b >= rate), pm, 0.5 * CV * pm, lmM, pd2, pd3, gl])
    write_csv("criteria_sweep.csv",
              ["w", "alpha_lo_on_0_15", "alpha_hi_on_0_15", "max_abs_dalpha", "beta_star_15", "corollary_ok",
               "min_p_T_over_triangle (attained at T=15)", "Cv_over_2_times_min_p", "mu_h_pencil_h0.01",
               "discrete_test_passed_h0.005", "discrete_test_passed_h0.0025", "exact_P1_min_Rayleigh_h0.01"],
              sweep_rows)

    def bisect_sign(fun, a, b, steps):
        """a: criterion satisfied, b: violated (both ends are checked); geometric bisection."""
        if not (fun(a) and not fun(b)):
            return np.nan, np.nan
        for _ in range(steps):
            m = np.sqrt(a * b)
            if fun(m):
                a = m
            else:
                b = m
        return a, b

    sw = np.array(sweep_rows, float)

    def first_fail(col_ok):
        """index of the first sampled w at which the criterion fails (all earlier samples pass)."""
        return int(np.nonzero(~col_ok)[0][0])

    crit_rows = []
    k = first_fail(sw[:, 5] > 0.5)
    a_, b_ = bisect_sign(crit_beta, sw[k - 1, 0], sw[k, 0], 40)
    crit_rows.append(["corollary (beta_* >= max|alpha'|, range on [0,15])", "", a_, b_,
                      int(np.all(sw[k:, 5] < 0.5))])
    k = first_fail(sw[:, 6] >= 0)
    a_, b_ = bisect_sign(lambda w: pmin_w(w) >= 0, sw[k - 1, 0], sw[k, 0], 18)
    crit_rows.append(["kernel condition (min p_T >= 0)", "", a_, b_, int(np.all(sw[k:, 6] < 0))])
    k = first_fail(sw[:, 11] >= 0)
    a_, b_ = bisect_sign(lambda w: p1_w(w) >= 0, sw[k - 1, 0], sw[k, 0], 16)
    crit_rows.append(["exact P1 energies (min E(15)/||v||^2 >= 0 on P1, h=0.01)", GRIDS[0], a_, b_,
                      int(np.all(sw[k:, 11] < 0))])
    cols = {GRIDS[0]: sw[:, 8] >= 0, GRIDS[1]: sw[:, 9] > 0.5, GRIDS[2]: sw[:, 10] > 0.5}
    for h in GRIDS:
        k = first_fail(cols[h])
        steps = {0.01: 18, 0.005: 16, 0.0025: 14}[h]
        a_, b_ = bisect_sign(lambda w: crit_lambda(w, h), sw[k - 1, 0], sw[k, 0], steps)
        crit_rows.append(["discrete test (lambda_min(S_h) >= 0, right-endpoint energy)", h, a_, b_,
                          int(not np.any(cols[h][k:]))])
    write_csv("criteria_critical_w.csv",
              ["criterion", "h", "w_lower (criterion satisfied)", "w_upper (criterion violated)",
               "violated at all larger sampled w"], crit_rows)
    for r in crit_rows:
        print(f"   critical w, {r[0]} {r[1]}: [{r[2]:.6f}, {r[3]:.6f}]", flush=True)

    # time series (finest grid)
    for case in CASES:
        A = fine[case]
        rows = np.column_stack([A["t"], A["i"], A["u"], A["v"], A["p"], A["pfrac"],
                                A["E"]["sup"], A["E"]["Rs"], A["E"]["Rp"], A["E"]["frac"]])
        write_csv(f"timeseries_{case}.csv",
                  ["t", "i_app", "u_terminal", "v_element", "p_supplied", "p_fractional",
                   "E_supplied", "E_Rs", "E_Rp", "E_fractional"], rows.tolist())

    # ---------------- F. verification ----------------
    print("F: verification of identities ...", flush=True)
    ver = verify_identities()
    write_csv("verification_checks.csv", ["check", "value"], ver)
    for r in ver:
        print("   ", r[0], ":", r[1], flush=True)

    # ---------------- G. figures ----------------
    make_figures()
    print(f"done in {time.perf_counter() - t_start:.1f} s", flush=True)


def read_csv(name):
    with open(os.path.join(RES, name), newline="") as fh:
        r = list(csv.reader(fh))
    return r[0], r[1:]


def make_figures():
    """Draw Figures 1-3 from the CSV files in ./results."""
    print("G: figures ...", flush=True)
    # Figure 1: counterexample (a-c) and rate bound of the corollary (d)
    _, rows = read_csv("counterexample_series.csv")
    d = np.array(rows, float)
    t, E = d[:, 0], d[:, 4]
    _, crow = read_csv("counterexample_certified.csv")
    ub = float([r for r in crow if r[0] == "E_total"][0][2])
    tt = np.linspace(0, float(CEX["T"]), 801)
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 1.7),
                           gridspec_kw=dict(width_ratios=[1, 1, 1.1, 1.35]))
    c0 = STYLE["constant"]["color"]
    ax[0].plot(tt, alpha_cex(tt), color=c0)
    ax[0].set_xlabel(r"$t$"); ax[0].set_ylabel(r"$\alpha(t)$"); ax[0].set_title("(a) order", loc="left")
    ax[1].plot(tt, v_cex(tt), color=c0)
    ax[1].set_xlabel(r"$t$"); ax[1].set_ylabel(r"$\upsilon(t)$"); ax[1].set_title("(b) voltage", loc="left")
    ax[2].plot(np.concatenate([[0], t]), np.concatenate([[0], E]), color=c0,
               label=r"$\mathcal{E}_h(t)/\mathcal{C}_{\mathrm{v}}$")
    ax[2].plot([float(CEX["T"])], [ub], marker="v", ms=5, color=INK, ls="none", label="certified bound")
    ax[2].axhline(0, color=MUTED, lw=0.8)
    ax[2].set_ylim(-0.15, 0.47)
    ax[2].set_xlabel(r"$t$"); ax[2].set_ylabel("energy"); ax[2].set_title("(c) supplied energy", loc="left")
    ax[2].legend(loc="upper left", fontsize=5.3, handlelength=1.5, borderpad=0.3, labelspacing=0.2)
    for a_ in ax[:3]:
        a_.set_xticks([0, 0.05, 0.1])
    _, rows = read_csv("rate_bound_curves.csv")
    for case, lab in (("slow", r"$\beta_*$, $\alpha_{\mathrm{s}}$"), ("rapid", r"$\beta_*$, $\alpha_{\mathrm{r}}$")):
        dd = np.array([[float(r[3]), float(r[4])] for r in rows if r[0] == case])
        ax[3].loglog(dd[:, 0], dd[:, 1], color=STYLE[case]["color"], ls=STYLE[case]["ls"], label=lab)
    for case, lab in (("slow", r"$\max|\dot{\alpha}_{\mathrm{s}}|$"), ("rapid", r"$\max|\dot{\alpha}_{\mathrm{r}}|$")):
        dd = np.array([[float(r[3]), float(r[5])] for r in rows if r[0] == case])
        ax[3].loglog(dd[:, 0], dd[:, 1], color=STYLE[case]["color"], lw=0.9, ls=":", label=lab)
    ax[3].set_ylim(1e-9, 1e8)
    ax[3].set_yticks([1e-8, 1e-4, 1e0, 1e4, 1e8])
    ax[3].set_xticks([1e-6, 1e-4, 1e-2, 1e0])
    ax[3].set_xlabel(r"horizon $T_0$"); ax[3].set_ylabel(r"$\beta_*(T_0)$")
    ax[3].set_title("(d) rate bound", loc="left")
    ax[3].legend(loc="lower left", fontsize=4.8, ncol=2, handlelength=1.6, columnspacing=0.6, borderpad=0.3, labelspacing=0.2)
    fig.tight_layout(w_pad=0.6)
    savefig(fig, "fig1_theory")

    # Figure 2: circuit responses (finest grid)
    ts = {}
    for case in CASES:
        hdr, rows = read_csv(f"timeseries_{case}.csv")
        dd = np.array(rows, float)
        ts[case] = {k: dd[:, j] for j, k in enumerate(hdr)}
    fig, ax = plt.subplots(2, 3, figsize=(6.8, 2.45), sharex=True)
    A0 = ts["constant"]
    ax[0, 0].plot(np.concatenate([[0], A0["t"]]), np.concatenate([[2.0], A0["i_app"]]), color=MUTED)
    ax[0, 0].set_ylabel(r"$\iota_{\mathrm{app}}$ [A]"); ax[0, 0].set_title("(a) applied current", loc="left")
    for case in CASES:
        A = ts[case]
        st = dict(color=STYLE[case]["color"], ls=STYLE[case]["ls"], lw=1.0)
        ax[0, 1].plot(A["t"], A["u_terminal"], label=STYLE[case]["label"], **st)
        ax[0, 2].plot(A["t"], A["p_supplied"], **st)
        ax[1, 0].plot(A["t"], A["E_supplied"], **st)
        ax[1, 1].plot(A["t"], A["p_fractional"], **st)
        ax[1, 2].plot(A["t"], A["E_fractional"], **st)
    ax[0, 1].set_ylabel(r"$u$ [V]"); ax[0, 1].set_title("(b) terminal voltage", loc="left")
    hl_, lb_ = ax[0, 1].get_legend_handles_labels()
    ax[0, 0].legend(hl_, lb_, loc="upper right", fontsize=5.5, handlelength=2.2, borderpad=0.3, labelspacing=0.25)
    ax[0, 2].set_ylabel(r"$u\,\iota_{\mathrm{app}}$ [W]"); ax[0, 2].set_title("(c) supplied power", loc="left")
    ax[1, 0].set_ylabel(r"$\mathcal{E}_{\mathrm{sup}}$ [J]"); ax[1, 0].set_title("(d) supplied energy", loc="left")
    ax[1, 1].set_ylabel(r"$\upsilon\,\mathcal{C}_{\mathrm{v}}D^{\alpha}\upsilon$ [W]")
    ax[1, 1].set_title("(e) element power", loc="left")
    ax[1, 2].set_ylabel(r"$\mathcal{E}_{\mathrm{frac}}$ [J]")
    ax[1, 2].set_title("(f) element energy", loc="left")
    for a_ in ax[1]:
        a_.set_xlabel(r"$t$ [s]")
        a_.set_xticks([0, 5, 10, 15])
    for a_ in (ax[0, 2], ax[1, 1]):
        a_.axhline(0, color=MUTED, lw=0.7)
    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    savefig(fig, "fig2_circuit")

    # Figure 3: certificate, criteria sweep, minimizing voltage
    _, rows = read_csv("certificate.csv")
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 1.7), gridspec_kw=dict(width_ratios=[1, 1.25, 1]))
    for case in CASES:
        dd = np.array([[float(r[1]), float(r[5])] for r in rows if r[0] == case])
        ax[0].semilogx(dd[:, 0], dd[:, 1], marker="o", ms=4, color=STYLE[case]["color"], ls=STYLE[case]["ls"],
                       label=STYLE[case]["label"])
    _, grows = read_csv("galerkin_rapid.csv")
    gd = np.array([[float(r[1]), float(r[2])] for r in grows if r[0].startswith("min over P1")])
    ax[0].semilogx(gd[:, 0], gd[:, 1], marker="x", ms=5, color=INK, ls="none", label=r"exact P1, $\alpha_{\mathrm{r}}$")
    ax[0].axhline(0, color=MUTED, lw=0.7)
    ax[0].set_xlabel(r"$h$"); ax[0].set_ylabel(r"$\lambda_{\min}(\mathbf{S}_{h,\alpha},\mathbf{M}_h)$")
    ax[0].set_title("(a) discrete test", loc="left")
    ax[0].set_ylim(-3.6, 2.1)
    ax[0].legend(fontsize=4.8, loc="center", ncol=2, handlelength=1.8, columnspacing=0.6, borderpad=0.3, labelspacing=0.2)
    ax[0].set_xticks(GRIDS); ax[0].set_xticklabels(["0.01", "0.005", "0.0025"])
    ax[0].minorticks_off()
    _, rows = read_csv("criteria_sweep.csv")
    sw = np.array(rows, float)
    _, crows = read_csv("criteria_critical_w.csv")
    ax[1].semilogx(sw[:, 0], sw[:, 7], color=STYLE["slow"]["color"], ls="--", marker=".", ms=3,
                   label=r"$\frac{\mathcal{C}_{\mathrm{v}}}{2}\min p_{15}$ (lower bound)")
    ax[1].semilogx(sw[:, 0], sw[:, 11], color=STYLE["constant"]["color"], ls="-", marker=".", ms=3,
                   label=r"exact P1, $h=0.01$ (upper bound)")
    ax[1].semilogx(sw[:, 0], sw[:, 8], color=STYLE["rapid"]["color"], ls="-.", marker=".", ms=3,
                   label=r"$\mu_h$, $h=0.01$")
    ax[1].axhline(0, color=MUTED, lw=0.7)
    wb = float(crows[0][2])
    ax[1].axvline(wb, color=INK, lw=0.8, ls=":")
    ax[1].text(wb * 1.15, 1.3, r"$\omega_\beta$", fontsize=7, color=INK)
    ax[1].axvline(8.0, color=MUTED, lw=0.6, ls="-")
    ax[1].text(8.0 * 0.5, 1.3, r"$\alpha_{\mathrm{r}}$", fontsize=7, color=MUTED)
    ax[1].set_ylim(-15.5, 3.0)
    ax[1].set_xlabel(r"$\omega$ in $0.7+0.2\sin(\omega t)$"); ax[1].set_ylabel("Rayleigh quotient")
    ax[1].set_title("(b) criteria sweep", loc="left"); ax[1].legend(fontsize=5.3, loc="lower left", bbox_to_anchor=(0.17, 0.0))
    _, rows = read_csv("eigvec_rapid.csv")
    dd = np.array(rows, float)
    tf, vec, alr = dd[:, 0], dd[:, 1], dd[:, 2]
    m = tf <= 3.0
    ax[2].plot(tf[m], vec[m] / vec.max(), color=STYLE["rapid"]["color"], lw=1.1, label="eigenvector (scaled)")
    ax[2].plot(tf[m], (alr[m] - 0.5) / 0.4, color=MUTED, lw=0.9, ls=":",
               label=r"$(\alpha_{\mathrm{r}}-0.5)/0.4$")
    ax[2].set_xlabel(r"$t$"); ax[2].set_ylabel("normalized value")
    ax[2].set_ylim(-0.05, 1.75); ax[2].set_yticks([0, 0.5, 1.0])
    ax[2].set_title(r"(c) minimizer on $[0,3]$", loc="left")
    ax[2].legend(fontsize=5.5, loc="upper left", ncol=1)
    fig.tight_layout(w_pad=0.8)
    savefig(fig, "fig3_certificate")


if __name__ == "__main__":
    import sys
    if "--figures-only" in sys.argv:
        make_figures()
    else:
        main()
