#!/usr/bin/env python3
"""
compare_results.py

Compares a freshly generated results folder with the reference CSV files shipped in this repository.

Usage:
    python compare_results.py REFERENCE_DIR NEW_DIR [--rtol 1e-8] [--atol 1e-12]

Typical use: copy the shipped ./results folder to ./results_reference, run `python simulation.py`
(which rewrites ./results), then run
    python compare_results.py results_reference results

Wall-clock timing columns (header starting with "time_") and environment.csv are skipped, because they depend
on the machine. Every other numerical cell must agree within the tolerances; text cells must match exactly.
Small differences (about 1e-12 relative) can occur between machines because of the LAPACK/BLAS build.
"""
import argparse
import csv
import math
import os
import sys


def read(path):
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def as_float(x):
    try:
        return float(x)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("new")
    ap.add_argument("--rtol", type=float, default=1e-8)
    ap.add_argument("--atol", type=float, default=1e-12)
    a = ap.parse_args()

    names = sorted(f for f in os.listdir(a.reference) if f.endswith(".csv") and f != "environment.csv")
    bad = 0
    for name in names:
        pn = os.path.join(a.new, name)
        if not os.path.exists(pn):
            print(f"MISSING  {name}")
            bad += 1
            continue
        h0, r0 = read(os.path.join(a.reference, name))
        h1, r1 = read(pn)
        if h0 != h1 or len(r0) != len(r1):
            print(f"SHAPE    {name}: header or row count differs")
            bad += 1
            continue
        skip = {j for j, h in enumerate(h0) if h.lower().startswith("time_")}
        worst, nbad = 0.0, 0
        for x, y in zip(r0, r1):
            for j, (u, v) in enumerate(zip(x, y)):
                if j in skip:
                    continue
                fu, fv = as_float(u), as_float(v)
                if fu is None or fv is None:
                    nbad += u != v
                    continue
                if math.isnan(fu) or math.isnan(fv):
                    nbad += not (math.isnan(fu) and math.isnan(fv))
                    continue
                d = abs(fu - fv)
                worst = max(worst, d / max(abs(fu), 1e-300))
                nbad += d > a.atol + a.rtol * abs(fu)
        status = "OK      " if nbad == 0 else "DIFFERS "
        bad += nbad > 0
        print(f"{status} {name}: max relative difference {worst:.2e}, cells outside tolerance {nbad}")
    print("ALL FILES AGREE" if bad == 0 else f"{bad} FILE(S) DIFFER")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
