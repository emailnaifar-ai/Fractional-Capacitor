#!/usr/bin/env python3
"""
verify_certified.py

Re-runs only the two interval-arithmetic (certified) computations of the manuscript, in a few seconds:

  1. Proposition 3.5: two-sided enclosures of E_1, E_2, E_3^+, E_3^- and E(T)/C_v for the C^1 order
     alpha(t) = 0.1 + 0.8 S((t - 0.01)/0.08) and the trapezoidal voltage on [0, 0.1]
     (mpmath interval arithmetic, 30 significant digits, P = 2000 hold subintervals, outward rounding).
     The script checks that the upper endpoint of the enclosure of E(T)/C_v is strictly negative.

  2. Corollary 3.4 for alpha_s(t) = 0.70 + 0.03 sin(0.2 t) on [0, 15]: a certified lower bound of
     beta_*(15) (interval arithmetic, 4000-point partition), checked against max|alpha_s'| = 0.006.

Floating-point quantities (discrete eigenvalues, p_T, exact P1 energies, circuit simulations) are NOT
certified; they are produced by simulation.py.

Run:  python verify_certified.py
"""
import sys

import mpmath

import simulation as sim


def main():
    ok = True
    print("Proposition 3.5 (interval arithmetic, outward rounding):")
    cert, e2_quad, etot_quad = sim.certify_counterexample()
    for key, (lo_iv, hi_iv) in cert.items():
        lo, hi = sim.iv_endpoint(lo_iv), sim.iv_endpoint(hi_iv)
        print(f"   {key:18s} in [{sim.round_dir(lo, 10, up=False)}, {sim.round_dir(hi, 10, up=True)}]")
    lo, hi = sim.iv_endpoint(cert["E_total"][0]), sim.iv_endpoint(cert["E_total"][1])
    upper_negative = hi < 0
    print(f"   upper endpoint of E(T)/C_v is strictly negative: {upper_negative}")
    print(f"   uncertified tanh-sinh value E(T)/C_v = {etot_quad:.9f} (inside the enclosure: {lo <= etot_quad <= hi})")
    ok &= upper_negative

    print("Corollary 3.4 for alpha_s on [0, 15] (interval arithmetic):")
    b = sim.beta_star_iv(15.0, 0.70, 0.73)
    rate = 0.03 * 0.2
    print(f"   certified lower bound beta_*(15) >= {sim.round_dir(mpmath.mpf(b), 6, up=False)}; max|alpha_s'| = {rate}; "
          f"condition holds: {b > rate}")
    ok &= b > rate

    print("ALL CERTIFIED CHECKS PASSED" if ok else "A CERTIFIED CHECK FAILED")
    print(f"(mpmath {mpmath.__version__})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
