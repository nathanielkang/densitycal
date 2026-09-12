"""Analyze experiment results and produce summary tables for manuscript."""
import pandas as pd
import numpy as np
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results_local')
CSV = os.path.join(RESULTS_DIR, 'experiment_results.csv')

def main():
    df = pd.read_csv(CSV)
    print(f"Loaded {len(df)} rows, {df['dataset'].nunique()} datasets, {df['method'].nunique()} methods")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print()

    methods_order = ['split', 'cqr', 'locally_adaptive', 'mondrian',
                     'calibrated_mixup', 'densitycal']

    # --- Table 1: Per-stratum coverage ---
    print("=" * 70)
    print("TABLE 1: Per-stratum coverage (percent), averaged over 8 datasets x 3 regressors x 3 seeds")
    print("=" * 70)
    rows = []
    for m in methods_order:
        sub = df[df['method'] == m]
        rare = sub['strat_cov_1'].mean() * 100
        rare_s = sub['strat_cov_1'].std() * 100
        common = sub['strat_cov_5'].mean() * 100
        common_s = sub['strat_cov_5'].std() * 100
        cg = sub['covgap'].mean()
        cg_s = sub['covgap'].std()
        mc = sub['marg_cov'].mean() * 100
        mc_s = sub['marg_cov'].std() * 100
        rows.append({
            'method': m, 'rare': rare, 'rare_s': rare_s,
            'common': common, 'common_s': common_s,
            'covgap': cg, 'covgap_s': cg_s,
            'marg': mc, 'marg_s': mc_s
        })
        print(f"  {m:20s}  Rare={rare:5.1f}+/-{rare_s:4.1f}  "
              f"Common={common:5.1f}+/-{common_s:4.1f}  "
              f"CovGap={cg:.4f}+/-{cg_s:.4f}  "
              f"MargCov={mc:5.1f}+/-{mc_s:4.1f}")
    print()

    # --- Table 2: Interval width ---
    print("=" * 70)
    print("TABLE 2: Interval width metrics")
    print("=" * 70)
    width_cols = [c for c in df.columns if 'width' in c.lower() or c == 'mpiw']
    print(f"  Available width columns: {width_cols}")
    if 'mpiw' in df.columns:
        for m in methods_order:
            sub = df[df['method'] == m]
            mpiw = sub['mpiw'].mean()
            mpiw_s = sub['mpiw'].std()
            wr = sub['width_ratio'].mean() if 'width_ratio' in sub.columns else float('nan')
            wr_s = sub['width_ratio'].std() if 'width_ratio' in sub.columns else float('nan')
            awr = sub['avg_width_rare'].mean() if 'avg_width_rare' in sub.columns else float('nan')
            awc = sub['avg_width_common'].mean() if 'avg_width_common' in sub.columns else float('nan')
            print(f"  {m:20s}  MPIW={mpiw:8.2f}+/-{mpiw_s:6.2f}  "
                  f"WidthRatio={wr:5.2f}+/-{wr_s:4.2f}  "
                  f"RareW={awr:8.2f}  CommonW={awc:8.2f}")
    print()

    # --- Per-dataset breakdown ---
    print("=" * 70)
    print("PER-DATASET: DensityCal vs Split (CovGap)")
    print("=" * 70)
    for ds in sorted(df['dataset'].unique()):
        sub = df[df['dataset'] == ds]
        avg_gap = sub.groupby('method')['covgap'].mean()
        best = avg_gap.idxmin()
        dc = avg_gap.get('densitycal', 999)
        sp = avg_gap.get('split', 999)
        imp = 100 * (sp - dc) / sp if sp > 0 else 0
        print(f"  {ds:15s}  best={best:20s}  DC={dc:.4f}  Split={sp:.4f}  impr={imp:+.1f}%")
    print()

    # --- Per-dataset per-stratum coverage for DensityCal vs Split ---
    print("=" * 70)
    print("PER-DATASET: Rare stratum coverage - Split vs DensityCal")
    print("=" * 70)
    for ds in sorted(df['dataset'].unique()):
        sub_sp = df[(df['dataset'] == ds) & (df['method'] == 'split')]
        sub_dc = df[(df['dataset'] == ds) & (df['method'] == 'densitycal')]
        sp_rare = sub_sp['strat_cov_1'].mean() * 100
        dc_rare = sub_dc['strat_cov_1'].mean() * 100
        sp_com = sub_sp['strat_cov_5'].mean() * 100
        dc_com = sub_dc['strat_cov_5'].mean() * 100
        print(f"  {ds:15s}  Split: rare={sp_rare:5.1f}% common={sp_com:5.1f}%  "
              f"DC: rare={dc_rare:5.1f}% common={dc_com:5.1f}%")

    # --- Per-regressor breakdown ---
    print()
    print("=" * 70)
    print("PER-REGRESSOR: CovGap")
    print("=" * 70)
    for reg in ['mlp', 'xgboost', 'rf']:
        sub = df[df['regressor'] == reg]
        print(f"  --- {reg} ---")
        for m in methods_order:
            msub = sub[sub['method'] == m]
            cg = msub['covgap'].mean()
            print(f"    {m:20s}  CovGap={cg:.4f}")


if __name__ == '__main__':
    main()
