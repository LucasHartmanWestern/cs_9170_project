"""
generate_all_v2.py
------------------
Generates LaTeX main results table (v2) and all paper figures for the v2 datasets.
Run from the project root:
    source ~/envs/rl/bin/activate
    python paper_figures/generate_all_v2.py

Outputs:
    paper_figures/tables_all_v2.tex              (combined table)
    paper_figures/table_{dataset}_v2.tex         (per-dataset stubs)
    paper_figures/fig_tradeoff_v2.png            (EO-AUC tradeoff, all 3 datasets)
    paper_figures/fig_pca_ablation_v2.png        (PCA1 vs PCA10)
    paper_figures/fig_twophase_ablation_v2.png   (phase1-only vs two-phase)
    paper_figures/fig_ffnn_ablation_v2.png       (FFNN epoch ablation)
"""

import json, csv, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings('ignore')

ROOT  = Path(__file__).parent.parent
TR    = ROOT / 'training_runs'
MAIN  = TR / 'paper_results' / 'main_results'
FFNN  = TR / 'paper_results' / 'ffnn_ablation'
NOPCA = TR / 'paper_results' / 'nopca_ablation'
OUT   = ROOT / 'paper_figures'
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})

# ── seeds ──────────────────────────────────────────────────────────────────────
SEEDS_5   = {'0', '1', '2', '3', '42'}
SEEDS_C24 = {'0', '1', '42'}   # degenerate splits on seeds 2 & 3 — see paper

# ── method style ───────────────────────────────────────────────────────────────
METHOD_COLORS = {
    'Group DRO':       '#4878CF',
    'OT Repair':       '#6ACC65',
    'CT-GAN':          '#D65F5F',
    'FairTabDDPM':     '#E58606',
    'FLB':             '#8B6BB1',
    'RL Framework':    '#B47CC7',
}
METHOD_MARKERS = {
    'Group DRO':    'o',
    'OT Repair':    's',
    'CT-GAN':       '^',
    'FairTabDDPM':  'P',
    'FLB':          'X',
    'RL Framework': 'D',
}
METHOD_ORDER = ['Group DRO', 'OT Repair', 'CT-GAN', 'FairTabDDPM', 'FLB', 'RL Framework']

# ── data loaders ──────────────────────────────────────────────────────────────

def _load_rl(base_dir, seeds=None):
    """Load RL seed results from final_test_metrics.csv files."""
    results = []
    base_dir = Path(base_dir)
    for sd in sorted(base_dir.glob('seed_*')):
        if seeds and sd.name.replace('seed_', '') not in seeds:
            continue
        f = sd / 'final_test_metrics.csv'
        if f.exists():
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    results.append({
                        k: float(v) if v not in ('', 'nan') else float('nan')
                        for k, v in row.items()
                        if k not in ('timestamp', 'run_id')
                    })
    return results


def _load_bl(base_dir, seeds=None):
    """Load baseline seed results from test_results.json files."""
    results = []
    base_dir = Path(base_dir)
    for sd in sorted(base_dir.glob('seed_*')):
        if seeds and sd.name.replace('seed_', '') not in seeds:
            continue
        f = sd / 'test_results.json'
        if f.exists():
            results.append(json.load(open(f)))
    return results


def _stats(results, key):
    vals = [float(d[key]) for d in results
            if not np.isnan(float(d.get(key, float('nan'))))]
    if not vals:
        return float('nan'), float('nan')
    return float(np.mean(vals)), float(np.std(vals))


def _fus(results):
    """FUS = (1 - EO) * AUC, computed per seed then averaged."""
    vals = []
    for d in results:
        eo  = float(d.get('beta_eo_tpr_diff', float('nan')))
        auc = float(d.get('beta_roc_auc',     float('nan')))
        if not (np.isnan(eo) or np.isnan(auc)):
            vals.append((1 - eo) * auc)
    if not vals:
        return float('nan'), float('nan')
    return float(np.mean(vals)), float(np.std(vals))


def _alpha_fus(results):
    """FUS from alpha (no-intervention) columns."""
    vals = []
    for d in results:
        eo  = float(d.get('alpha_eo_tpr_diff', float('nan')))
        auc = float(d.get('alpha_roc_auc',     float('nan')))
        if not (np.isnan(eo) or np.isnan(auc)):
            vals.append((1 - eo) * auc)
    if not vals:
        return float('nan'), float('nan')
    return float(np.mean(vals)), float(np.std(vals))


# ── dataset configs ────────────────────────────────────────────────────────────

DATASETS = [
    {
        'key':     'census_b010',
        'display': 'Census Income',
        'methods': [
            ('Group DRO',
             _load_bl(TR / 'BASELINE_group_dro_census_b010_gdro_5dd1a038__G202603261757')),
            ('OT Repair',
             _load_bl(TR / 'BASELINE_gaussian_ot_repair_census_b010_otrep_4bc4a87a__G202603261757')),
            ('CT-GAN',
             _load_bl(TR / 'BASELINE_ctgan_census_b010_ctgan_ec9135ff__G202603261758')),
            ('FairTabDDPM',
             _load_bl(TR / 'BASELINE_fairtabddpm_census_b010_fairtabddpm_d13b3144__G202603261758')),
            ('FLB',
             _load_bl(TR / 'BASELINE_fairness_loss_balancing_census_b010_flb_194dbfcd__G202603261757')),
            ('RL Framework',
             _load_rl(TR / 'SPECv18_ablation_global_only_bias010_census_3seeds_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.1_GG202603182150_4676389e')),
        ],
    },
    {
        'key':     'capture24_b002',
        'display': 'CAPTURE-24',
        'methods': [
            ('Group DRO',
             _load_bl(TR / 'BASELINE_group_dro_p1_capture24_bias002_gdro_5s_cf3e96ec__G202603261354', SEEDS_C24)),
            ('OT Repair',
             _load_bl(TR / 'BASELINE_gaussian_ot_repair_p1_capture24_bias002_otrep_5s_ff220b37__G202603261358', SEEDS_C24)),
            ('CT-GAN',
             _load_bl(TR / 'BASELINE_ctgan_p1_capture24_bias002_ctgan_5s_716ece6e__G202603261359', SEEDS_C24)),
            ('FairTabDDPM',
             _load_bl(TR / 'BASELINE_fairtabddpm_p1_capture24_bias002_fairtabddpm_5s_b0c3e054__G202603261411', SEEDS_C24)),
            ('FLB',
             _load_bl(TR / 'BASELINE_fairness_loss_balancing_p1_capture24_bias002_flb_5s_f925a6cd__G202603261410', SEEDS_C24)),
            ('RL Framework',
             _load_rl(TR / 'SPECp1_capture24_v18_k3_3s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.02_GG202603260112_d834f9d4')),
        ],
    },
    {
        'key':     'credit_b010',
        'display': 'Credit Card',
        'methods': [
            ('Group DRO',
             _load_bl(MAIN / 'BASELINE_group_dro_p1_credit_bias010_gdro_5s_dcca27f2__G202603211544')),
            ('OT Repair',
             _load_bl(MAIN / 'BASELINE_gaussian_ot_repair_p1_credit_bias010_otrep_5s_ep20_d580f330__G202603201755')),
            ('CT-GAN',
             _load_bl(MAIN / 'BASELINE_ctgan_p1_credit_bias010_ctgan_5s_ep20_49701d91__G202603211545')),
            ('FairTabDDPM',
             _load_bl(TR / 'BASELINE_fairtabddpm_v18_bias010_fairtabddpm_credit_5s_13535203__G202603261413')),
            ('FLB',
             _load_bl(TR / 'BASELINE_fairness_loss_balancing_v18_bias010_flb_credit_5s_2a8c1118__G202603261356')),
            ('RL Framework',
             _load_rl(MAIN / 'SPECp1_main_credit_bias010_global_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.1_GG202603200009_abfd28d9')),
        ],
    },
]

# ── table formatting ────────────────────────────────────────────────────────────

def _fmt(mean, std, bold=False):
    if np.isnan(mean):
        return r'--'
    s = rf'{mean:.3f}$_{{\pm {std:.3f}}}$'
    return rf'\textbf{{{s}}}' if bold else s


def _best(methods_results, fn, lower=True):
    vals = [fn(r) for _, r in methods_results if r]
    means = [m for m, _ in vals if not np.isnan(m)]
    if not means:
        return float('nan')
    return min(means) if lower else max(means)


# ── 1. Combined LaTeX Table ────────────────────────────────────────────────────

def build_combined_table():
    n_methods = 1 + len(DATASETS[0]['methods'])  # No Intervention + baselines
    caption = (r'Fairness and Utility Performance under Positive-Class Scarcity. '
               r'Best result per column and dataset is \textbf{bolded}. '
               r'FUS $= (1-\text{EO})\times\text{AUC}$.')
    label = 'tab:main_results'

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\setlength{\tabcolsep}{5pt}')
    lines.append(rf'\caption{{{caption}}}')
    lines.append(rf'\label{{{label}}}')
    lines.append(r'\begin{tabular}{llccccc}')
    lines.append(r'\toprule')
    lines.append(r'Dataset & Method & EO $\downarrow$ & DP $\downarrow$ '
                 r'& AUC $\uparrow$ & Acc $\uparrow$ & FUS $\uparrow$ \\')
    lines.append(r'\midrule')

    for di, ds in enumerate(DATASETS):
        methods = ds['methods']
        display = ds['display']
        n_rows  = 1 + len(methods)

        alpha_src = next((r for _, r in methods if r), None)

        best_eo  = _best(methods, lambda r: _stats(r, 'beta_eo_tpr_diff'), lower=True)
        best_dp  = _best(methods, lambda r: _stats(r, 'beta_dp_diff'),     lower=True)
        best_auc = _best(methods, lambda r: _stats(r, 'beta_roc_auc'),     lower=False)
        best_acc = _best(methods, lambda r: _stats(r, 'beta_acc'),         lower=False)
        best_fus = _best(methods, lambda r: _fus(r),                       lower=False)

        def is_best(val, best, lower):
            return (not np.isnan(val)) and abs(val - best) < 1e-9

        mr = rf'\multirow{{{n_rows}}}{{*}}{{{display}}}'
        if alpha_src:
            aeo_m, aeo_s = _stats(alpha_src, 'alpha_eo_tpr_diff')
            adp_m, adp_s = _stats(alpha_src, 'alpha_dp_diff')
            aauc_m, _    = _stats(alpha_src, 'alpha_roc_auc')
            aacc_m, _    = _stats(alpha_src, 'alpha_acc')
            afus_m, _    = _alpha_fus(alpha_src)
            lines.append(
                rf'{mr} & \textit{{No Intervention}} '
                rf'& {_fmt(aeo_m, aeo_s)} & {_fmt(adp_m, adp_s)} '
                rf'& {aauc_m:.3f} & {aacc_m:.3f} & {afus_m:.3f} \\'
            )
        else:
            lines.append(rf'{mr} & \textit{{No Intervention}} & -- & -- & -- & -- & -- \\')

        for name, results in methods:
            if not results:
                lines.append(rf' & \textbf{{{name}}} & -- & -- & -- & -- & -- \\'
                              if name == 'RL Framework' else
                              rf' & {name} & -- & -- & -- & -- & -- \\')
                continue
            eo_m,  eo_s  = _stats(results, 'beta_eo_tpr_diff')
            dp_m,  dp_s  = _stats(results, 'beta_dp_diff')
            auc_m, auc_s = _stats(results, 'beta_roc_auc')
            acc_m, acc_s = _stats(results, 'beta_acc')
            fus_m, fus_s = _fus(results)

            tex_name = rf'\textbf{{{name}}}' if name == 'RL Framework' else name
            lines.append(
                rf' & {tex_name} '
                rf'& {_fmt(eo_m,  eo_s,  bold=is_best(eo_m,  best_eo,  lower=True))} '
                rf'& {_fmt(dp_m,  dp_s,  bold=is_best(dp_m,  best_dp,  lower=True))} '
                rf'& {_fmt(auc_m, auc_s, bold=is_best(auc_m, best_auc, lower=False))} '
                rf'& {_fmt(acc_m, acc_s, bold=is_best(acc_m, best_acc, lower=False))} '
                rf'& {_fmt(fus_m, fus_s, bold=is_best(fus_m, best_fus, lower=False))} \\'
            )

        if di < len(DATASETS) - 1:
            lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)


# ── 2. Tradeoff Plot ────────────────────────────────────────────────────────────

def make_tradeoff_plot():
    print('Generating tradeoff plot (v2)...')

    legend_handles = [
        mpatches.Patch(color=METHOD_COLORS[m], label=m) for m in METHOD_ORDER
    ]

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5 * len(DATASETS), 4.5),
                             sharey=False)
    fig.suptitle('Fairness–Utility Tradeoff (EO Gap vs ROC-AUC)',
                 fontsize=13, fontweight='bold')

    for ax, ds in zip(axes, DATASETS):
        alpha_src = next((r for _, r in ds['methods'] if r), None)

        # No Intervention marker (grey cross)
        if alpha_src:
            aeo_m, aeo_s = _stats(alpha_src, 'alpha_eo_tpr_diff')
            aauc_m, aauc_s = _stats(alpha_src, 'alpha_roc_auc')
            ax.errorbar(aeo_m, aauc_m, xerr=aeo_s, yerr=aauc_s,
                        fmt='x', color='#888888', capsize=3, markersize=9,
                        linestyle='none', label='No Intervention', zorder=2)

        for name, results in ds['methods']:
            if not results:
                continue
            eo_m,  eo_s  = _stats(results, 'beta_eo_tpr_diff')
            auc_m, auc_s = _stats(results, 'beta_roc_auc')
            ax.errorbar(eo_m, auc_m, xerr=eo_s, yerr=auc_s,
                        fmt=METHOD_MARKERS[name], color=METHOD_COLORS[name],
                        capsize=3, markersize=8, linestyle='none',
                        label=name, zorder=3)

        ax.set_title(ds['display'], fontsize=11, fontweight='bold')
        ax.set_xlabel('EO Gap ↓', fontsize=9)
        ax.set_ylabel('ROC-AUC ↑', fontsize=9)
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.annotate('', xy=(0.04, 0.96), xytext=(0.14, 0.86),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='#888888',
                                   lw=1.2, connectionstyle='arc3,rad=0.0'))
        ax.text(0.15, 0.85, 'Ideal', transform=ax.transAxes,
                fontsize=7, color='#888888', va='top', ha='left', style='italic')

    ni_handle = mpatches.Patch(color='#888888', label='No Intervention')
    fig.legend(handles=[ni_handle] + legend_handles,
               loc='lower center', ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    fname = 'fig_tradeoff_v2.png'
    plt.savefig(OUT / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


# ── 3. PCA Ablation ─────────────────────────────────────────────────────────────
# Compares PCA1 (no dimensionality reduction) vs PCA10 (our method).
# Uses global-only reward. Census and Credit, bias=010 and bias=005.

def _nopca_path(dataset, bias, reward):
    ds = dataset.lower()
    pattern = f'SPECp1_nopca_{ds}_bias{bias}_{reward}_5s_EP800_PCA1_'
    for d in NOPCA.iterdir():
        if d.name.startswith(pattern):
            p = d / 'final_test_metrics.csv'
            if p.exists():
                return str(p)
    return None


def _pca10_path(dataset, bias):
    """PCA10 reference: the main v18 result for census/credit bias010."""
    ds = dataset.lower()
    if ds == 'census' and bias == '010':
        p = TR / 'SPECv18_ablation_global_only_bias010_census_3seeds_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.1_GG202603182150_4676389e'
    elif ds == 'credit' and bias == '010':
        p = MAIN / 'SPECp1_main_credit_bias010_global_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.1_GG202603200009_abfd28d9'
    else:
        return None
    p_csv = p / 'final_test_metrics.csv'
    return str(p_csv) if p_csv.exists() else None


def make_pca_ablation():
    print('Generating PCA ablation plot (v2)...')
    metrics = [
        ('beta_eo_tpr_diff', 'EO Gap ↓', True),
        ('beta_dp_diff',     'DP Gap ↓', True),
        ('beta_roc_auc',     'AUC ↑',    False),
        ('beta_acc',         'Acc ↑',    False),
        ('beta_f1_weighted', 'F1$_w$ ↑', False),
    ]
    datasets_biases = [('Census', '010'), ('Credit', '010')]
    colors = {'PCA1\n(no reduction)': '#cccccc', 'PCA10\n(ours)': '#B47CC7'}

    fig, axes = plt.subplots(len(datasets_biases), len(metrics),
                             figsize=(len(metrics) * 2.8, len(datasets_biases) * 3.2),
                             squeeze=False)
    fig.suptitle('PCA Dimensionality Reduction Ablation (bias=10%)',
                 fontsize=12, fontweight='bold')

    for row_i, (dataset, bias) in enumerate(datasets_biases):
        axes[row_i][0].set_ylabel(dataset, fontsize=10)

        import pandas as pd
        pca1_path  = _nopca_path(dataset, bias, 'global')
        pca10_path = _pca10_path(dataset, bias)
        df_pca1  = pd.read_csv(pca1_path)  if pca1_path  and Path(pca1_path).exists()  else None
        df_pca10 = pd.read_csv(pca10_path) if pca10_path and Path(pca10_path).exists() else None

        for col_i, (col, col_label, lower) in enumerate(metrics):
            ax = axes[row_i][col_i]
            if row_i == 0:
                ax.set_title(col_label, fontsize=10)

            variants = [
                ('PCA1\n(no reduction)', df_pca1),
                ('PCA10\n(ours)',         df_pca10),
            ]
            vals, errs, labels, bar_colors = [], [], [], []
            for lbl, df in variants:
                if df is not None and col in df.columns:
                    vals.append(df[col].mean())
                    errs.append(df[col].std())
                    labels.append(lbl)
                    bar_colors.append(colors[lbl])

            if not vals:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='gray')
                continue

            x = np.arange(len(vals))
            bars = ax.bar(x, vals, yerr=errs, capsize=4, color=bar_colors,
                          alpha=0.85, width=0.5, edgecolor='black', linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (max(errs) if errs else 0) * 0.05,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=8)
            if len(vals) == 2:
                better_idx = (0 if vals[0] < vals[1] else 1) if lower else \
                             (0 if vals[0] > vals[1] else 1)
                bars[better_idx].set_edgecolor('#2ca02c')
                bars[better_idx].set_linewidth(2.0)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)

    plt.tight_layout()
    fname = 'fig_pca_ablation_v2.png'
    plt.savefig(OUT / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


# ── 4. Curriculum Learning Ablation ───────────────────────────────────────────
# Compares v18 config with curriculum enabled (start_dim=2, stage_count=5)
# vs v18 with curriculum disabled (start_dim=10, stage_count=1 — our method).
# Census bias=010, 5 seeds each, all other hyperparameters identical.

def make_curriculum_ablation():
    print('Generating curriculum learning ablation (v2)...')
    import pandas as pd

    curr_dir = sorted(
        TR.glob('SPECv18_curr_ablation_census_bias010_5s_*'),
        key=lambda d: d.stat().st_mtime, reverse=True
    )
    nocurr_dir = TR / 'SPECv18_ablation_global_only_bias010_census_3seeds_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.1_GG202603182150_4676389e'

    curr_csv   = curr_dir[0] / 'final_test_metrics.csv' if curr_dir else None
    nocurr_csv = nocurr_dir / 'final_test_metrics.csv'

    df_curr   = pd.read_csv(curr_csv)   if curr_csv   and Path(curr_csv).exists()   else None
    df_nocurr = pd.read_csv(nocurr_csv) if nocurr_csv.exists() else None

    if df_curr is None and df_nocurr is None:
        print('  No data found for curriculum ablation, skipping')
        return

    metrics = [
        ('beta_eo_tpr_diff', 'EO Gap ↓', True),
        ('beta_dp_diff',     'DP Gap ↓', True),
        ('beta_roc_auc',     'AUC ↑',    False),
        ('beta_acc',         'Acc ↑',    False),
        ('beta_f1_weighted', 'F1$_w$ ↑', False),
    ]
    variants = [
        ('Curriculum\n(start 2D→10D)', df_curr,   '#cccccc'),
        ('No Curriculum\n(ours)',       df_nocurr, '#B47CC7'),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics) * 2.8, 3.8))
    fig.suptitle('Curriculum Learning Ablation — Census Income (bias=10%)',
                 fontsize=11, fontweight='bold')

    for ax, (col, col_label, lower) in zip(axes, metrics):
        ax.set_title(col_label, fontsize=10)
        vals, errs, labels, colors = [], [], [], []
        for lbl, df, color in variants:
            if df is not None and col in df.columns:
                vals.append(df[col].mean())
                errs.append(df[col].std())
                labels.append(lbl)
                colors.append(color)

        if not vals:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='gray')
            continue

        x = np.arange(len(vals))
        bars = ax.bar(x, vals, yerr=errs, capsize=4, color=colors,
                      alpha=0.85, width=0.5, edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(errs) if errs else 0) * 0.05,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        if len(vals) == 2:
            better_idx = (0 if vals[0] < vals[1] else 1) if lower else \
                         (0 if vals[0] > vals[1] else 1)
            bars[better_idx].set_edgecolor('#2ca02c')
            bars[better_idx].set_linewidth(2.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)

    n_curr   = len(df_curr)   if df_curr   is not None else 0
    n_nocurr = len(df_nocurr) if df_nocurr is not None else 0
    fig.text(0.01, 0.01, f'n seeds — Curriculum: {n_curr}, No Curriculum: {n_nocurr}',
             fontsize=7, color='gray')

    plt.tight_layout()
    fname = 'fig_curriculum_ablation_v2.png'
    plt.savefig(OUT / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


# ── 5. Reward Ablation: DVRL (v17a) vs Global-Only (v18) ──────────────────────
# v17a uses DVRL local reward; v18 is pure global reward.
# Both: census bias=010, 5 seeds, otherwise identical config.

def make_reward_ablation():
    print('Generating reward ablation plot (v17a vs v18)...')
    import pandas as pd

    v17a_csv = TR / 'SPECv17a_bias010_rl_census_5seeds_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.1_GG202603182150_e7a6f2fe' / 'final_test_metrics.csv'
    v18_csv  = TR / 'SPECv18_ablation_global_only_bias010_census_3seeds_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.1_GG202603182150_4676389e' / 'final_test_metrics.csv'

    df_v17a = pd.read_csv(v17a_csv) if v17a_csv.exists() else None
    df_v18  = pd.read_csv(v18_csv)  if v18_csv.exists()  else None

    if df_v17a is None and df_v18 is None:
        print('  No data found for reward ablation, skipping')
        return

    metrics = [
        ('beta_eo_tpr_diff', 'EO Gap ↓', True),
        ('beta_dp_diff',     'DP Gap ↓', True),
        ('beta_roc_auc',     'AUC ↑',    False),
        ('beta_acc',         'Acc ↑',    False),
        ('beta_f1_weighted', 'F1$_w$ ↑', False),
    ]
    variants = [
        ('Local+Global\n(DVRL)', df_v17a, '#E58606'),
        ('Global-Only\n(ours)',  df_v18,  '#B47CC7'),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics) * 2.8, 3.8))
    fig.suptitle('Reward Ablation: DVRL vs Global-Only — Census Income (bias=10%)',
                 fontsize=11, fontweight='bold')

    for ax, (col, col_label, lower) in zip(axes, metrics):
        ax.set_title(col_label, fontsize=10)
        vals, errs, labels, colors = [], [], [], []
        for lbl, df, color in variants:
            if df is not None and col in df.columns:
                vals.append(df[col].mean())
                errs.append(df[col].std())
                labels.append(lbl)
                colors.append(color)

        if not vals:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='gray')
            continue

        x = np.arange(len(vals))
        bars = ax.bar(x, vals, yerr=errs, capsize=4, color=colors,
                      alpha=0.85, width=0.5, edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(errs) if errs else 0) * 0.05,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        if len(vals) == 2:
            better_idx = (0 if vals[0] < vals[1] else 1) if lower else \
                         (0 if vals[0] > vals[1] else 1)
            bars[better_idx].set_edgecolor('#2ca02c')
            bars[better_idx].set_linewidth(2.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)

    n_v17a = len(df_v17a) if df_v17a is not None else 0
    n_v18  = len(df_v18)  if df_v18  is not None else 0
    fig.text(0.01, 0.01, f'n seeds — DVRL: {n_v17a}, Global-Only: {n_v18}',
             fontsize=7, color='gray')

    plt.tight_layout()
    fname = 'fig_reward_ablation_v2.png'
    plt.savefig(OUT / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


# ── 5. Two-Phase vs One-Phase Ablation ─────────────────────────────────────────
# Compares: phase 1 only (minority augmentation only) vs
#           two-phase (minority + majority recovery) — census bias=010, 5 seeds.

def make_twophase_ablation():
    print('Generating two-phase ablation plot (v2)...')
    import pandas as pd

    phase1_dir = TR / 'SPECp1_phase1only_census_bias010_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.1_GG202603221848_c7c7b7f3'
    phase2_dir = TR / 'SPECp1_phase2_census_bias010_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.1_GG202603231036_16efa573'

    p1_csv = phase1_dir / 'final_test_metrics.csv'
    p2_csv = phase2_dir / 'final_test_metrics.csv'
    df_p1 = pd.read_csv(p1_csv) if p1_csv.exists() else None
    df_p2 = pd.read_csv(p2_csv) if p2_csv.exists() else None

    if df_p1 is None and df_p2 is None:
        print('  No data found for two-phase ablation, skipping')
        return

    metrics = [
        ('beta_eo_tpr_diff', 'EO Gap ↓', True),
        ('beta_dp_diff',     'DP Gap ↓', True),
        ('beta_roc_auc',     'AUC ↑',    False),
        ('beta_acc',         'Acc ↑',    False),
        ('beta_f1_weighted', 'F1$_w$ ↑', False),
    ]
    variants = [
        ('Phase 1 Only\n(minority)', df_p1, '#cccccc'),
        ('Two-Phase\n(ours)',        df_p2, '#B47CC7'),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics) * 2.8, 3.8))
    fig.suptitle('Two-Phase Training Ablation — Census Income (bias=10%)',
                 fontsize=11, fontweight='bold')

    for ax, (col, col_label, lower) in zip(axes, metrics):
        ax.set_title(col_label, fontsize=10)
        vals, errs, labels, colors = [], [], [], []
        for lbl, df, color in variants:
            if df is not None and col in df.columns:
                vals.append(df[col].mean())
                errs.append(df[col].std())
                labels.append(lbl)
                colors.append(color)

        if not vals:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9, color='gray')
            continue

        x = np.arange(len(vals))
        bars = ax.bar(x, vals, yerr=errs, capsize=4, color=colors,
                      alpha=0.85, width=0.5, edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(errs) if errs else 0) * 0.05,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        if len(vals) == 2:
            better_idx = (0 if vals[0] < vals[1] else 1) if lower else \
                         (0 if vals[0] > vals[1] else 1)
            bars[better_idx].set_edgecolor('#2ca02c')
            bars[better_idx].set_linewidth(2.0)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)

    n_p1 = len(df_p1) if df_p1 is not None else 0
    n_p2 = len(df_p2) if df_p2 is not None else 0
    fig.text(0.01, 0.01, f'n seeds — Phase 1 Only: {n_p1}, Two-Phase: {n_p2}',
             fontsize=7, color='gray')

    plt.tight_layout()
    fname = 'fig_twophase_ablation_v2.png'
    plt.savefig(OUT / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


# ── 5. FFNN Epoch Ablation ─────────────────────────────────────────────────────
# Tests β classifier training epochs: 5, 20 (selected), 50.
# Census and Credit, bias=010.

def _ffnn_path(dataset, ep):
    ds = dataset.lower()
    pattern = f'SPECp1_ffnn_{ds}_ep{ep:02d}_bias010_global_5s'
    for d in FFNN.iterdir():
        if d.name.startswith(pattern):
            p = d / 'final_test_metrics.csv'
            if p.exists():
                return str(p)
    return None


def make_ffnn_ablation():
    print('Generating FFNN epoch ablation (v2)...')
    import pandas as pd

    epochs = [5, 20, 50]
    metrics = [
        ('beta_eo_tpr_diff', 'EO Gap ↓', True),
        ('beta_dp_diff',     'DP Gap ↓', True),
        ('beta_roc_auc',     'AUC ↑',    False),
        ('beta_acc',         'Acc ↑',    False),
        ('beta_f1_weighted', 'F1$_w$ ↑', False),
    ]

    for dataset in ['Census', 'Credit']:
        fig, axes = plt.subplots(1, len(metrics), figsize=(len(metrics) * 2.8, 3.8))
        fig.suptitle(f'FFNN Epoch Ablation: {dataset} (bias=10%)',
                     fontsize=12, fontweight='bold')

        for ax, (col, col_label, lower) in zip(axes, metrics):
            ax.set_title(col_label, fontsize=10)
            vals, errs, labels = [], [], []
            for ep in epochs:
                p = _ffnn_path(dataset, ep)
                df = pd.read_csv(p) if p else None
                if df is not None and col in df.columns:
                    vals.append(df[col].mean())
                    errs.append(df[col].std())
                    labels.append(f'ep={ep}')

            if not vals:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='gray')
                continue

            x = np.arange(len(vals))
            bar_colors = ['#B47CC7' if l == 'ep=20' else '#cccccc' for l in labels]
            bars = ax.bar(x, vals, yerr=errs, capsize=4, color=bar_colors,
                          alpha=0.85, width=0.6, edgecolor='black', linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (max(errs) if errs else 0) * 0.05,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_xlabel('β classifier epochs', fontsize=9)

        axes[-1].annotate('★ selected', xy=(0, 0), xycoords='axes fraction',
                          fontsize=8, color='#B47CC7', xytext=(0, -0.2))
        plt.tight_layout()
        fname = f'fig_ffnn_ablation_{dataset.lower()}_v2.png'
        plt.savefig(OUT / fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved {fname}')


# ── run ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    header = ('% Auto-generated by paper_figures/generate_all_v2.py\n'
              '% Run: python paper_figures/generate_all_v2.py\n'
              '% Requires: \\usepackage{booktabs} in preamble\n')

    # Tables
    tex = build_combined_table()
    combined = OUT / 'tables_all_v2.tex'
    combined.write_text(header + '\n' + tex + '\n')
    print(f'Wrote {combined}')
    for ds in DATASETS:
        out_path = OUT / f'table_{ds["key"]}_v2.tex'
        out_path.write_text(header + '\n' + tex + '\n')
        print(f'Wrote {out_path}')

    # Figures
    make_tradeoff_plot()
    make_curriculum_ablation()
    make_reward_ablation()
    make_pca_ablation()
    make_twophase_ablation()
    make_ffnn_ablation()

    print('\nDone. All v2 outputs in paper_figures/')
