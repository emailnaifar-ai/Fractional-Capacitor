# Fractional-Capacitor

This repository contains the code and numerical results for the article:

> **When Is a Variable-Order Fractional Capacitor Passive? Energy Identity, Rate Condition, and a Certified Counterexample**
> 

It reproduces every figure, table and reported number of the article. That includes the interval-arithmetic (certified) computations.

## What the code computes

The element is `iota(t) = C_v * D^{alpha(t)} v(t)`, a Caputo derivative whose order is evaluated at the current time, starting from the relaxed state `v(0) = 0`. Time is measured in units of a reference time `t_r = 1 s`. The code computes:

- **Certified counterexample (Proposition 3.5).** Two-sided interval enclosures of the supplied energy for a C¹ order that rises from 0.1 to 0.9, together with a trapezoidal voltage (mpmath interval arithmetic, 30 digits, outward rounding).
- **Rate condition (Corollary 3.4).** The bound `beta_*(T0)`, including a certified lower bound for `alpha_s = 0.70 + 0.03 sin(0.2 t)` on [0, 15], and the horizons it guarantees.
- **Kernel condition (Theorem 3.2).** Direct evaluation of the weight `p_T(tau)` on the triangle `0 < tau < T <= T0`.
- **Discrete passivity test (Section 4).** The L1 matrix `H`, the energy matrix `S = (W H + H^T W)/2`, `lambda_min(S)`, the mass-matrix-normalized eigenvalue `mu_h`, the discrete passivity horizon and the port energy form.
- **Exact continuous energies of piecewise-linear voltages.** Closed-form current and the exact P1 (Galerkin) energy matrix.
- **Supercapacitor circuit (Section 5).** `u = R_s i_app + v`, `i_app = v/R_p + C_v D^{alpha(t)} v`, with `R_s = 0.1 Ohm`, `R_p = 100 Ohm`, `C_v = 10 A/V`, T = 15 and the orders `alpha_c = 0.70`, `alpha_s = 0.70 + 0.03 sin(0.2 t)` and `alpha_r = 0.70 + 0.20 sin(8 t)`. Solved on three grids (h = 0.01, 0.005, 0.0025) with two independent methods: L1 with right-endpoint energies, and Grünwald–Letnikov with trapezoidal energies.
- **Criteria sweep.** `alpha_w = 0.7 + 0.2 sin(w t)` on [0, 15], comparing the rate bound, the kernel condition, exact P1 energies and the discrete test.

## Repository structure

```
Fractional-Capacitor/
├── README.md             this file
├── LICENSE               MIT License
├── CITATION.cff          citation metadata
├── requirements.txt      pip dependencies (tested versions)
├── environment.yml       conda alternative
├── simulation.py         reproduces all results (sections A-J, see its docstring)
├── verify_certified.py   re-runs only the interval-arithmetic (certified) results, in a few seconds
├── compare_results.py    compares regenerated CSV files with the reference files
├── results/              reference outputs (CSV) used in the article
└── figures/              Figures 1-3 (PDF and PNG, 300 dpi)
```

## Installation

Python 3.12 was used for testing; newer versions should also work.

```bash
pip install -r requirements.txt
```

Alternatively, with conda:

```bash
conda env create -f environment.yml
conda activate fractional-capacitor
```

## Reproducing the results

1. **Certified results only (a few seconds).**

   ```bash
   python verify_certified.py
   ```

   It prints the enclosure `E(T)/C_v in [-0.1187708849, -0.1183686085]` and confirms that its upper endpoint is strictly negative. It also prints the certified bound `beta_*(15) >= 0.009550 > 0.006`.

2. **Everything (about 7–10 minutes on a laptop).**

   ```bash
   python simulation.py
   ```

   This rewrites `results/*.csv` and `figures/*`. To redraw only the figures from the CSV files:

   ```bash
   python simulation.py --figures-only
   ```

3. **Check the regenerated results against the reference files.**

   ```bash
   cp -r results results_reference
   python simulation.py
   python compare_results.py results_reference results
   ```

   Wall-clock timing columns and `environment.csv` are skipped by the comparison. All other quantities are deterministic. Different BLAS/LAPACK builds may change them only at the level of about 1e-12, relative.

The reference outputs were produced on Windows 11 with Python 3.12.10, NumPy 2.4.6, SciPy 1.17.1, mpmath 1.3.0 and Matplotlib 3.10.9 (see `results/environment.csv`).

## Where each result of the article comes from

| Article item | Output file(s) |
|---|---|
| Proposition 3.5: enclosures of E1, E2, E3+, E3− and E(T) | `results/counterexample_certified.csv` (also printed by `verify_certified.py`) |
| Fig. 1(a–c): counterexample order, voltage, discrete energy | `results/counterexample_series.csv` |
| Counterexample on three grids (`lambda_min/h`, `E_h(T)`) | `results/counterexample_discrete.csv` |
| Fig. 1(d): `beta_*(T0)` and maximal rates | `results/rate_bound_curves.csv` |
| Horizons from Corollary 3.4 and the certified `beta_*(15)` | `results/rate_bound_horizons.csv` |
| Minimum of `p_T` and kernel-condition horizons | `results/kernel_condition.csv` |
| Fig. 2: circuit time series | `results/timeseries_{constant,slow,rapid}.csv` |
| Table 1: energies, `lambda_min(S)`, `mu_h`, discrete horizons | `results/circuit_energies.csv`, `results/certificate.csv` |
| Convergence orders, Richardson extrapolation | `results/richardson.csv` |
| Fig. 3(a): `mu_h` and exact P1 minima | `results/certificate.csv`, `results/galerkin_rapid.csv` |
| Fig. 3(b), critical frequencies of each criterion | `results/criteria_sweep.csv`, `results/criteria_critical_w.csv` |
| Fig. 3(c): minimizing eigenvector | `results/eigvec_rapid.csv` |
| Right-endpoint, trapezoidal and exact energies of a fixed voltage | `results/test_voltage_rapid.csv` |
| Port (terminal) energy form | `results/certificate.csv` (column `port_lambda_min_over_h`) |
| Numerical checks of the identities in Lemma 3.1 and Theorem 3.2 | `results/verification_checks.csv` |
| Software versions | `results/environment.csv` |

## Numerical settings

- **Solvers.** All are direct (forward substitution; LAPACK `eigh` and Cholesky) and involve no iteration.
- **Adaptive quadrature.** Used only in the verification checks, with `epsrel = 1e-11` (`1e-12` in the closed-form check of `N_hat`) and `epsabs = 1e-13`.
- **Minimum of `p_T`.** Located by composite Gauss–Legendre panels with a graded first panel, followed by a Nelder–Mead polish (`xatol = 1e-10`).
- **Interval arithmetic.** mpmath `iv` with 30 significant digits and outward rounding. The counterexample uses P = 2000 hold subintervals, and the rate bound a 4000-point partition.
- **Bisections.** All have a fixed number of iterations, documented in `simulation.py`.

## Scope

Only the interval computations are certified: Proposition 3.5 and `beta_*(15)`. Everything else is a floating-point computation, reported as numerical evidence. This includes the discrete eigenvalues, `p_T`, the exact P1 energies and the circuit simulations. The circuit is a simulation-based illustration, not an experimental validation.

## License and citation

The code is released under the MIT License (see `LICENSE`). If you use it, please cite the article (reference to be added upon publication) and this repository (see `CITATION.cff`).
