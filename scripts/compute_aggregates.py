"""Compute aggregate statistics from experiment_results.csv for updating main.tex tables."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
csv_path = ROOT / "results_v2" / "experiment_results.csv"
df = pd.read_csv(csv_path)

datasets = sorted(df["dataset"].unique())
methods = ["split", "cqr", "locally_adaptive", "mondrian", "credo", "calibrated_mixup", "densitycal"]
labels = {
    "split": "Split", "cqr": "CQR", "locally_adaptive": "Loc.-Ad.",
    "mondrian": "Mondrian", "credo": "CREDO",
    "calibrated_mixup": "Cal.Mix+CP", "densitycal": "DensityCal",
}

print(f"Total rows: {len(df)}")
print(f"Datasets ({len(datasets)}): {datasets}")
print(f"Methods ({len(df['method'].unique())}): {sorted(df['method'].unique())}")
print()

print("=" * 100)
print("AGGREGATE TABLE (for main-text Table 2: coverage)")
print("=" * 100)
header = f"{'Method':15s} {'Rare':>10s} {'Common':>10s} {'CovGap':>12s} {'MargCov':>8s}"
print(header)
for m in methods:
    sub = df[df["method"] == m]
    rare = sub["strat_cov_1"].mean() * 100
    rare_sd = sub["strat_cov_1"].std() * 100
    com = sub["strat_cov_5"].mean() * 100
    com_sd = sub["strat_cov_5"].std() * 100
    cg = sub["covgap"].mean()
    cg_sd = sub["covgap"].std()
    mc = sub["marg_cov"].mean() * 100
    print(f"{labels[m]:15s} {rare:5.1f}+-{rare_sd:4.1f} {com:5.1f}+-{com_sd:4.1f} {cg:.3f}+-{cg_sd:.3f} {mc:5.1f}")

print()
print("=" * 100)
print("AGGREGATE TABLE (for main-text Table 3: width)")
print("=" * 100)
header = f"{'Method':15s} {'RareW':>8s} {'CommonW':>8s} {'WRatio':>8s} {'MPIW':>8s}"
print(header)
for m in methods:
    sub = df[df["method"] == m]
    rw = sub["avg_width_rare"].mean()
    cw = sub["avg_width_common"].mean()
    wr = sub["width_ratio"].mean()
    mpiw = sub["mpiw"].mean()
    print(f"{labels[m]:15s} {rw:8.1f} {cw:8.1f} {wr:8.2f} {mpiw:8.1f}")

print()
print("=" * 100)
print("PER-DATASET RARE COVERAGE (%)")
print("=" * 100)
header = f"{'Dataset':15s}" + "".join(f" {labels[m]:>12s}" for m in methods)
print(header)
for ds in datasets:
    row = f"{ds:15s}"
    for m in methods:
        sub = df[(df["dataset"] == ds) & (df["method"] == m)]
        val = sub["strat_cov_1"].mean() * 100
        row += f" {val:12.1f}"
    print(row)
row = f"{'Average':15s}"
for m in methods:
    vals = [df[(df["dataset"] == ds) & (df["method"] == m)]["strat_cov_1"].mean() * 100 for ds in datasets]
    row += f" {np.mean(vals):12.1f}"
print(row)

print()
print("=" * 100)
print("PER-DATASET COVGAP")
print("=" * 100)
header = f"{'Dataset':15s}" + "".join(f" {labels[m]:>12s}" for m in methods)
print(header)
for ds in datasets:
    row = f"{ds:15s}"
    for m in methods:
        sub = df[(df["dataset"] == ds) & (df["method"] == m)]
        val = sub["covgap"].mean()
        row += f" {val:12.3f}"
    print(row)
row = f"{'Average':15s}"
for m in methods:
    vals = [df[(df["dataset"] == ds) & (df["method"] == m)]["covgap"].mean() for ds in datasets]
    row += f" {np.mean(vals):12.3f}"
print(row)

print()
print("=" * 100)
print("PER-DATASET MPIW")
print("=" * 100)
header = f"{'Dataset':15s}" + "".join(f" {labels[m]:>12s}" for m in methods)
print(header)
for ds in datasets:
    row = f"{ds:15s}"
    for m in methods:
        sub = df[(df["dataset"] == ds) & (df["method"] == m)]
        val = sub["mpiw"].mean()
        row += f" {val:12.2f}"
    print(row)
row = f"{'Average':15s}"
for m in methods:
    vals = [df[(df["dataset"] == ds) & (df["method"] == m)]["mpiw"].mean() for ds in datasets]
    row += f" {np.mean(vals):12.2f}"
print(row)
