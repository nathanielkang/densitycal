"""Rebuild occupancy / residual-ratio diagnostics from a frozen results CSV.

Usage (from 2_Code/):
    python scripts/build_diagnostic_tables.py --csv results_v2/experiment_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("results_v2/experiment_results.csv"),
    )
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    print(f"rows={len(df)} cols={list(df.columns)[:12]}...")

    n0 = 30
    count_cols = [c for c in df.columns if c.startswith("n_bin") or c.startswith("bin_n")]
    if count_cols:
        sub = df.drop_duplicates(subset=[c for c in ("dataset", "regressor", "seed") if c in df.columns])
        sparse = 0
        for _, row in sub.iterrows():
            ns = [int(row[c]) for c in count_cols if pd.notna(row[c])]
            if ns and min(ns) < n0:
                sparse += 1
        print(f"sparse_runs {sparse}/{len(sub)}")
    else:
        print("no per-bin count columns; occupancy table stays a protocol count")

    if {"method", "covgap"}.issubset(df.columns):
        g = df.groupby("method")["covgap"].mean()
        print("mean CovGap by method:")
        print(g.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
