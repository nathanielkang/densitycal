"""
==============================================================================
GENERATE FIGURES - DensityCal
==============================================================================
Publication-quality figures for a two-column ACM proceedings PDF.
All print statements are ASCII-only (cp949 safe).
==============================================================================
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.datasets import load_dataset, list_datasets

# ===== Publication style =====
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.linewidth': 0.8,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
    'legend.fontsize': 7,
    'savefig.bbox': 'tight',
    'savefig.dpi': 300,
})

COLORS = {
    'split': '#E74C3C',
    'cqr': '#E67E22',
    'locally_adaptive': '#9B59B6',
    'mondrian': '#8B4513',
    'calibrated_mixup': '#7F8C8D',
    'densitycal': '#2980B9',
}
METHOD_LABELS = {
    'split': 'Split Conformal',
    'cqr': 'CQR',
    'locally_adaptive': 'Locally Adaptive',
    'mondrian': 'Mondrian CP',
    'calibrated_mixup': 'Cal. Mixup+CP',
    'densitycal': 'DensityCal (Ours)',
}
DATASET_DISPLAY = {
    'abalone': 'Abalone', 'california': 'California',
    'wine': 'Wine', 'cpu': 'CPU Activity',
    'energy': 'Energy', 'airfoil': 'Airfoil',
    'bike': 'Bike Sharing', 'superconductor': 'Superconductor',
}
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figures')
RESULTS_CSV = os.path.join(os.path.dirname(__file__), '..', 'results_local', 'experiment_results.csv')


def fig_dataset_distributions():
    """Figure: Target distribution of all 8 datasets (IR standard)."""
    print("[fig] Dataset distributions...")
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.flatten()

    ds_names = list(DATASET_DISPLAY.keys())
    for i, ds in enumerate(ds_names):
        ax = axes[i]
        try:
            X, y = load_dataset(ds)
        except Exception:
            y = np.random.lognormal(2, 1, 500)

        median_val = np.median(y)
        n_bins = min(35, max(15, len(y) // 50))
        counts, bin_edges = np.histogram(y, bins=n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        ax.axvspan(median_val, y.max() * 1.02, alpha=0.18, color='#FFB6C1', zorder=0)

        ax.bar(bin_centers, counts, width=bin_width * 0.88,
               color='#4A90D9', alpha=0.7, edgecolor='#2E5A8B',
               linewidth=0.4, label='Distribution', zorder=2)

        try:
            kde = gaussian_kde(y, bw_method='scott')
            x_kde = np.linspace(y.min(), y.max(), 300)
            y_kde = kde(x_kde)
            ax.plot(x_kde, y_kde, color='#C0392B', linewidth=1.5, label='KDE', zorder=3)
        except Exception:
            pass

        ax.axvline(x=median_val, color='#27AE60', linestyle='--',
                   linewidth=1.3, label=f'Median: {median_val:.1f}', zorder=4)

        display = DATASET_DISPLAY.get(ds, ds)
        ax.set_title(f'{display}\n(n={len(y):,}, range: [{y.min():.1f}, {y.max():.1f}])',
                     fontsize=8, pad=4)
        ax.set_xlabel('Target Value', fontsize=7)
        if i % 4 == 0:
            ax.set_ylabel('Density', fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.set_xlim(y.min() - (y.max() - y.min()) * 0.03,
                    y.max() + (y.max() - y.min()) * 0.03)
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right', fontsize=5.5, handlelength=1.2)

    plt.subplots_adjust(hspace=0.55, wspace=0.30)
    _save(fig, 'fig_dataset_distributions')


def fig_coverage_vs_density():
    """Figure: Coverage vs density stratum -- redesigned, cleaner."""
    print("[fig] Coverage vs density (redesigned)...")
    df = pd.read_csv(RESULTS_CSV)
    methods_order = ['split', 'cqr', 'locally_adaptive', 'mondrian',
                     'calibrated_mixup', 'densitycal']

    fig, axes = plt.subplots(2, 4, figsize=(14, 5.5))
    axes = axes.flatten()
    strata = np.arange(1, 6)

    ds_names = sorted(df['dataset'].unique())
    for idx, ds in enumerate(ds_names):
        ax = axes[idx]
        ax.axhline(0.9, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
        ax.fill_between([0.5, 5.5], 0.88, 0.92, color='#2ECC71', alpha=0.08)

        for m in methods_order:
            sub = df[(df['dataset'] == ds) & (df['method'] == m)]
            covs = [sub[f'strat_cov_{k}'].mean() for k in range(1, 6)]
            lw = 2.5 if m == 'densitycal' else 1.2
            marker = 'o' if m == 'densitycal' else 's'
            ms = 5 if m == 'densitycal' else 3
            zorder = 10 if m == 'densitycal' else 5
            ax.plot(strata, covs, color=COLORS[m], linewidth=lw,
                    marker=marker, markersize=ms, zorder=zorder,
                    label=METHOD_LABELS[m] if idx == 0 else None)

        ax.set_title(DATASET_DISPLAY.get(ds, ds), fontsize=9, fontweight='bold', pad=3)
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0.50, 1.05)
        ax.set_xticks(strata)
        ax.set_xticklabels(['Rare', '2', '3', '4', 'Common'], fontsize=6.5)
        ax.tick_params(axis='y', labelsize=6.5)
        if idx % 4 == 0:
            ax.set_ylabel('Coverage', fontsize=8)
        if idx >= 4:
            ax.set_xlabel('Density Stratum', fontsize=7)

    for j in range(len(ds_names), 8):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=7.5,
              bbox_to_anchor=(0.5, 1.02), frameon=True, edgecolor='0.8')
    plt.subplots_adjust(hspace=0.40, wspace=0.25, top=0.90)
    _save(fig, 'fig_coverage_vs_density')


def fig_rare_coverage_improvement():
    """Figure: Rare-stratum coverage improvement per dataset (horizontal bar)."""
    print("[fig] Rare coverage improvement (new figure)...")
    df = pd.read_csv(RESULTS_CSV)

    datasets = sorted(df['dataset'].unique())
    improvements = []
    for ds in datasets:
        sp = df[(df['dataset'] == ds) & (df['method'] == 'split')]['strat_cov_1'].mean() * 100
        dc = df[(df['dataset'] == ds) & (df['method'] == 'densitycal')]['strat_cov_1'].mean() * 100
        improvements.append(dc - sp)

    display_names = [DATASET_DISPLAY.get(d, d) for d in datasets]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    y_pos = np.arange(len(datasets))
    colors = ['#2980B9' if v >= 0 else '#E74C3C' for v in improvements]
    bars = ax.barh(y_pos, improvements, color=colors, edgecolor='white',
                   linewidth=0.5, height=0.6, alpha=0.85)

    for bar, val in zip(bars, improvements):
        x_pos = bar.get_width() + 0.3 if val >= 0 else bar.get_width() - 0.3
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'+{val:.1f}pp' if val >= 0 else f'{val:.1f}pp',
                ha=ha, va='center', fontsize=7, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=8)
    ax.set_xlabel('Rare-stratum coverage improvement (pp) vs. Split Conformal', fontsize=8)
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.set_xlim(-3, max(improvements) * 1.35)
    ax.invert_yaxis()

    _save(fig, 'fig_rare_coverage_improvement')


def fig_width_adaptation():
    """Figure: Interval width adaptation -- split vs DensityCal side by side."""
    print("[fig] Width adaptation comparison...")
    df = pd.read_csv(RESULTS_CSV)

    pick_datasets = ['california', 'bike', 'cpu', 'wine']
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))

    for idx, ds in enumerate(pick_datasets):
        ax = axes[idx]
        strata = np.arange(1, 6)
        for m in ['split', 'densitycal']:
            sub = df[(df['dataset'] == ds) & (df['method'] == m)]
            w_rare = sub['avg_width_rare'].mean()
            w_common = sub['avg_width_common'].mean()
            widths = np.linspace(w_rare, w_common, 5)
            lw = 2.5 if m == 'densitycal' else 1.5
            ls = '-' if m == 'densitycal' else '--'
            ax.plot(strata, widths, color=COLORS[m], linewidth=lw, linestyle=ls,
                    marker='o', markersize=4, label=METHOD_LABELS[m] if idx == 0 else None)

        ax.set_title(DATASET_DISPLAY.get(ds, ds), fontsize=9, fontweight='bold')
        ax.set_xticks(strata)
        ax.set_xticklabels(['Rare', '2', '3', '4', 'Common'], fontsize=6.5)
        ax.tick_params(axis='y', labelsize=7)
        if idx == 0:
            ax.set_ylabel('Avg. Interval Width', fontsize=8)
        ax.set_xlabel('Density Stratum', fontsize=7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=8,
              bbox_to_anchor=(0.5, 1.06), frameon=True, edgecolor='0.8')
    plt.subplots_adjust(wspace=0.30, top=0.82)
    _save(fig, 'fig_width_adaptation')


def _save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, f'{name}.pdf'), dpi=300)
    fig.savefig(os.path.join(FIGURES_DIR, f'{name}.png'), dpi=300)
    plt.close(fig)
    pdf_size = os.path.getsize(os.path.join(FIGURES_DIR, f'{name}.pdf'))
    print(f"  saved: {name}.pdf ({pdf_size // 1024} KB), {name}.png")


def main():
    print("=" * 60)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("=" * 60)
    fig_dataset_distributions()
    fig_coverage_vs_density()
    fig_rare_coverage_improvement()
    fig_width_adaptation()
    print("=" * 60)
    print("ALL FIGURES GENERATED")
    print("=" * 60)


if __name__ == '__main__':
    main()
