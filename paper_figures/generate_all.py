"""
generate_all.py
---------------
Single script to reproduce every paper figure and table.
Run from the project root:
    source ~/envs/rl/bin/activate
    python paper_figures/generate_all.py

Outputs are written to paper_figures/.
Missing data is handled gracefully (shown as '--' in tables, skipped in plots).
"""

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
PAPER_TR  = ROOT / 'training_runs' / 'training_runs_paper'
SPEC_TR   = PAPER_TR / 'training_runs'
LOCAL_TR  = ROOT / 'training_runs'
OUT       = ROOT / 'paper_figures'
OUT.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
METHOD_COLORS = {
    'Group DRO':    '#4878CF',
    'OT Repair':    '#6ACC65',
    'CT-GAN':       '#D65F5F',
    'RL Framework': '#B47CC7',
}
METHOD_ORDER = ['Group DRO', 'OT Repair', 'CT-GAN', 'RL Framework']
BIAS_LABELS  = {'nobias': r'No Bias (0\%)', '005': r'Moderate Bias (5\%)', '010': r'High Bias (10\%)'}

plt.rcParams.update({
    'font.family': 'serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})

# ── Data registry ──────────────────────────────────────────────────────────────
# Each entry: (path_to_final_test_metrics.csv, is_3seed_provisional)
def _s(p): return str(p)

RESULTS = {
    # ── Census ──
    ('Census', 'nobias', 'Group DRO'):
        _s(LOCAL_TR/'BASELINE_group_dro_p1_nobias_census_gdro_5s_5951e26e__G202603201755/final_test_metrics.csv'),
    ('Census', 'nobias', 'OT Repair'):
        _s(LOCAL_TR/'BASELINE_gaussian_ot_repair_p1_nobias_census_otrep_5s_216fbc09__G202603201755/final_test_metrics.csv'),
    ('Census', 'nobias', 'CT-GAN'):
        _s(LOCAL_TR/'BASELINE_ctgan_p1_nobias_census_ctgan_5s_900a27b1__G202603201756/final_test_metrics.csv'),
    ('Census', 'nobias', 'RL Framework'):
        _s(PAPER_TR/'SPECp1_nobias_census_ours_ep20_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_GG202603202304_352917e1/final_test_metrics.csv'),

    ('Census', '005', 'Group DRO'):
        _s(LOCAL_TR/'BASELINE_group_dro_v16_bias05_gdro_census_3seeds_9ff6e175__G202603171753/final_test_metrics.csv'),
    ('Census', '005', 'OT Repair'):
        _s(LOCAL_TR/'BASELINE_gaussian_ot_repair_p1_census_bias05_otrep_5s_ep20_539e2739__G202603201755/final_test_metrics.csv'),
    ('Census', '005', 'CT-GAN'):
        _s(LOCAL_TR/'BASELINE_ctgan_p1_census_bias05_ctgan_5s_ep20_6db6db20__G202603201756/final_test_metrics.csv'),
    ('Census', '005', 'RL Framework'):
        _s(SPEC_TR/'SPECp1_main_census_bias05_global_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.05_GG202603200009_28c96eee/final_test_metrics.csv'),

    ('Census', '010', 'Group DRO'):
        _s(LOCAL_TR/'BASELINE_group_dro_v16_bias010_gdro_census_3seeds_cd68eb16__G202603171800/final_test_metrics.csv'),
    ('Census', '010', 'OT Repair'):
        _s(LOCAL_TR/'BASELINE_gaussian_ot_repair_p1_census_bias010_otrep_5s_ep20_2084dedc__G202603201755/final_test_metrics.csv'),
    ('Census', '010', 'CT-GAN'):
        _s(LOCAL_TR/'BASELINE_ctgan_p1_census_bias010_ctgan_5s_ep20_579aa0ca__G202603201756/final_test_metrics.csv'),
    ('Census', '010', 'RL Framework'):
        _s(SPEC_TR/'SPECp1_ffnn_census_ep20_bias010_global_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.1_GG202603192345_591446a3/final_test_metrics.csv'),

    # ── Credit ──
    # Credit no-bias: pending (no run yet)
    ('Credit', '005', 'Group DRO'):
        _s(LOCAL_TR/'BASELINE_group_dro_v16_bias05_gdro_credit_3seeds_c161369f__G202603171827/final_test_metrics.csv'),
    ('Credit', '005', 'OT Repair'):
        _s(LOCAL_TR/'BASELINE_gaussian_ot_repair_p1_credit_bias05_otrep_5s_ep20_33727f8e__G202603201755/final_test_metrics.csv'),
    # Credit bias=0.05 CT-GAN: no ep20 rerun available
    ('Credit', '005', 'RL Framework'):
        _s(SPEC_TR/'SPECp1_main_credit_bias05_global_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.05_GG202603200009_501656c4/final_test_metrics.csv'),

    ('Credit', '010', 'Group DRO'):
        _s(LOCAL_TR/'BASELINE_group_dro_v16_bias010_gdro_credit_3seeds_70d5567c__G202603171827/final_test_metrics.csv'),
    ('Credit', '010', 'OT Repair'):
        _s(LOCAL_TR/'BASELINE_gaussian_ot_repair_p1_credit_bias010_otrep_5s_ep20_d580f330__G202603201755/final_test_metrics.csv'),
    # Credit bias=0.10 CT-GAN: missing
    ('Credit', '010', 'RL Framework'):
        _s(SPEC_TR/'SPECp1_main_credit_bias010_global_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.1_GG202603200009_abfd28d9/final_test_metrics.csv'),
}

# Learning curve run directories (seed-level, contains metrics.csv)
LEARNING_RUNS = {
    ('Census', 'nobias'): str(PAPER_TR / 'SPECp1_nobias_census_ours_ep20_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_GG202603202304_352917e1'),
    ('Census', '010'):    str(SPEC_TR  / 'SPECp1_ffnn_census_ep20_bias010_global_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.1_GG202603192345_591446a3'),
    ('Credit', '010'):    str(SPEC_TR  / 'SPECp1_main_credit_bias010_global_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.1_GG202603200009_abfd28d9'),
    ('Credit', '005'):    str(SPEC_TR  / 'SPECp1_main_credit_bias05_global_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.05_GG202603200009_501656c4'),
}

# Reward ablation: dvrl vs global, across budget configs (no curriculum)
# Format: (dataset, bias, budget_variant) -> {dvrl: path, global: path}
BUDGET_CONFIGS = ['hireal', 'hisynth', 'scale1', 'scale2']

def _budget_path(dataset, bias, variant, reward):
    ds  = dataset.lower()
    b   = f'_BIAS0.{bias[-1]}' if bias != 'nobias' else ''
    # find matching dir
    pattern = f'SPECp1_budget_{ds}_{variant}_{bias}_{reward}_5s_EP800'
    for d in SPEC_TR.iterdir():
        if d.name.startswith(pattern):
            p = d / 'final_test_metrics.csv'
            return str(p) if p.exists() else None
    return None

def _ffnn_path(dataset, ep):
    ds = dataset.lower()
    pattern = f'SPECp1_ffnn_{ds}_ep{ep:02d}_bias010_global_5s'
    for d in SPEC_TR.iterdir():
        if d.name.startswith(pattern):
            p = d / 'final_test_metrics.csv'
            return str(p) if p.exists() else None
    return None

def _delta_path(dataset, ds_str, reward):
    ds = dataset.lower()
    pattern = f'SPECp1_delta_{ds}_{ds_str}_bias010_{reward}_5s'
    for d in SPEC_TR.iterdir():
        if d.name.startswith(pattern):
            p = d / 'final_test_metrics.csv'
            return str(p) if p.exists() else None
    return None

# ── Helpers ────────────────────────────────────────────────────────────────────

def load(path):
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return None

def n_seeds(df):
    return len(df) if df is not None else 0

def provisional(df):
    return df is not None and len(df) < 5

def mean_std(df, col):
    if df is None or col not in df.columns:
        return None, None
    return df[col].mean(), df[col].std()

def fus(df):
    """Fairness-Utility Score = (1 - EO) * AUC. Range (0,1), higher is better."""
    if df is None:
        return None, None
    score = (1 - df['beta_eo_tpr_diff']) * df['beta_roc_auc']
    return score.mean(), score.std()

def smooth(x, w=20):
    if len(x) < w:
        return x
    return pd.Series(x).rolling(w, min_periods=1, center=True).mean().values

def load_learning_curves(run_dir):
    """Load and average metrics.csv across all completed seeds."""
    run_path = Path(run_dir)
    seed_dirs = sorted([d for d in run_path.iterdir() if d.name.startswith('seed_')])
    dfs = []
    for sd in seed_dirs:
        p = sd / 'metrics.csv'
        if p.exists():
            df = pd.read_csv(p)
            dfs.append(df)
    if not dfs:
        return None
    # Align on episode index
    min_ep = min(len(d) for d in dfs)
    truncated = [d.iloc[:min_ep].reset_index(drop=True) for d in dfs]
    mean_df = pd.concat(truncated).groupby(level=0).mean(numeric_only=True)
    std_df  = pd.concat(truncated).groupby(level=0).std(numeric_only=True)
    return mean_df, std_df, len(dfs)

# ── 1. Main Results Tables ─────────────────────────────────────────────────────

FULL_METRICS = [
    # (col, display_name, lower_is_better)
    ('beta_eo_tpr_diff',  'EO',      True),
    ('beta_dp_diff',      'DP',      True),
    ('beta_eod_max_diff', 'EOd',     True),
    ('beta_roc_auc',      'AUC',     False),
    ('beta_acc',          'Acc',     False),
    ('beta_f1_weighted',  'F1$_w$',  False),
    ('beta_f1_minority',  'F1$_m$',  False),
    ('beta_brier',        'Brier',   True),
]

COMPRESSED_METRICS = [
    ('beta_eo_tpr_diff', 'EO',    True),
    ('beta_dp_diff',     'DP',    True),
    ('beta_roc_auc',     'AUC',   False),
    ('beta_acc',         'Acc',   False),
    # FUS handled separately
]

DATASETS = ['Census', 'Credit', 'HAR']
BIASES   = ['nobias', '005', '010']
METHODS  = ['Group DRO', 'OT Repair', 'CT-GAN', 'RL Framework']


def best_in_group(dataset, bias, col, lower):
    vals = {}
    for m in METHODS:
        df = load(RESULTS.get((dataset, bias, m)))
        if df is not None and col in df.columns:
            vals[m] = df[col].mean()
    if not vals:
        return set()
    best = min(vals.values()) if lower else max(vals.values())
    return {m for m, v in vals.items() if abs(v - best) < 1e-9}


def fmt_cell(df, col, bold=False, prov=False):
    if df is None or col not in df.columns:
        return '--'
    m, s = df[col].mean(), df[col].std()
    dag = r'\dagger' if prov else ''
    val = f'{m:.3f}$_{{\\pm {s:.3f}{dag}}}$'
    return r'\textbf{' + val + '}' if bold else val


def build_table(metrics_list, bias, include_fus=False, label='tab:results', caption=''):
    """Build a single table for one bias level."""
    n_metrics = len(metrics_list) + (1 if include_fus else 0)
    col_spec = 'll' + 'c' * n_metrics

    down, up = r'$\downarrow$', r'$\uparrow$'
    metric_names = [f'{n} {down if lo else up}' for _, n, lo in metrics_list]
    if include_fus:
        metric_names.append(r'FUS $\uparrow$')
    header = 'Dataset & Method & ' + ' & '.join(metric_names)

    footnote = (r'$\dagger$ provisional ($<$5 seeds). '
                r'Best per column is \textbf{bolded}.')
    if include_fus:
        footnote += r' FUS $= (1-\text{EO})\times\text{AUC}$.'

    lines = [
        r'\begin{table}[t]',
        r'\centering',
        r'\setlength{\tabcolsep}{5pt}',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        f'\\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
        header + r' \\',
        r'\midrule',
    ]

    for di, dataset in enumerate(DATASETS):
        for mi, method in enumerate(METHODS):
            prefix = r'\multirow{4}{*}{' + dataset + '}' if mi == 0 else ''
            row = f'{prefix} & {method}'
            if dataset == 'HAR':
                row += ' & --' * n_metrics
            else:
                df = load(RESULTS.get((dataset, bias, method)))
                prov = provisional(df)
                for col, _, lower in metrics_list:
                    bests = best_in_group(dataset, bias, col, lower)
                    bold = method in bests
                    row += ' & ' + fmt_cell(df, col, bold=bold, prov=prov)
                if include_fus:
                    fus_m, fus_s = fus(df)
                    if fus_m is not None:
                        fus_vals = {}
                        for mm in METHODS:
                            dfm = load(RESULTS.get((dataset, bias, mm)))
                            fm, _ = fus(dfm)
                            if fm is not None:
                                fus_vals[mm] = fm
                        bold_fus = method == max(fus_vals, key=fus_vals.get) if fus_vals else False
                        dag = r'\dagger' if prov else ''
                        cell = f'{fus_m:.3f}$_{{\\pm {fus_s:.3f}{dag}}}$'
                        row += ' & ' + (r'\textbf{' + cell + '}' if bold_fus else cell)
                    else:
                        row += ' & --'
            lines.append(row + r' \\')
        if di < len(DATASETS) - 1:
            lines.append(r'\midrule')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\smallskip',
        f'\\par\\raggedright\\small {footnote}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def make_tables():
    print('Generating tables...')

    bias_titles = {
        'nobias': 'No Bias',
        '005':    'Moderate Bias (5\\%)',
        '010':    'High Bias (10\\%)',
    }

    for suffix, metrics_list, include_fus in [
        ('compressed', COMPRESSED_METRICS, True),
        ('full',       FULL_METRICS,       False),
    ]:
        blocks = []
        for bias, bias_title in bias_titles.items():
            cap_c = (
                f'Results under {bias_title}. '
                r'EO and DP ($\downarrow$) are fairness metrics; '
                r'AUC and Acc ($\uparrow$) are utility metrics. '
                r'FUS $= (1-\text{EO})\times\text{AUC}$.'
            )
            cap_f = f'Full results under {bias_title} including all tracked metrics.'
            caption = cap_c if include_fus else cap_f
            label   = f'tab:{bias}_{suffix}'
            blocks.append(build_table(metrics_list, bias,
                                      include_fus=include_fus,
                                      label=label, caption=caption))
        fname = f'table_main_{suffix}.tex'
        (OUT / fname).write_text('\n\n'.join(blocks) + '\n')
        print(f'  Saved {fname}')


# ── 2. Fairness-Utility Tradeoff ───────────────────────────────────────────────

def make_tradeoff_plots():
    print('Generating fairness-utility tradeoff plots...')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle('Fairness–Utility Tradeoff (EO vs ROC-AUC)', fontsize=13, fontweight='bold')

    markers = {'Group DRO': 'o', 'OT Repair': 's', 'CT-GAN': '^', 'RL Framework': 'D'}
    bias_styles = {'nobias': '-', '005': '--', '010': ':'}
    bias_alpha  = {'nobias': 1.0, '005': 0.75, '010': 0.55}

    for ax, dataset in zip(axes, ['Census', 'Credit', 'HAR']):
        ax.set_title(dataset, fontsize=11, fontweight='bold')
        ax.set_xlabel('EO Gap ↓ (lower is fairer)', fontsize=9)
        ax.set_ylabel('ROC-AUC ↑', fontsize=9)

        for method in METHOD_ORDER:
            for bias in BIASES:
                if dataset == 'HAR':
                    continue
                df = load(RESULTS.get((dataset, bias, method)))
                if df is None:
                    continue
                eo_m, eo_s = mean_std(df, 'beta_eo_tpr_diff')
                auc_m, auc_s = mean_std(df, 'beta_roc_auc')
                color = METHOD_COLORS[method]
                ax.errorbar(eo_m, auc_m, xerr=eo_s, yerr=auc_s,
                            fmt=markers[method], color=color,
                            alpha=bias_alpha[bias], capsize=3,
                            markersize=7 if bias == 'nobias' else 5,
                            linestyle='none',
                            label=f'{method} ({BIAS_LABELS[bias]})' if bias == 'nobias' else None)

        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

    # Shared legend
    legend_handles = [
        mpatches.Patch(color=METHOD_COLORS[m], label=m) for m in METHOD_ORDER
    ]
    bias_handles = [
        plt.Line2D([0], [0], color='k', alpha=a, linewidth=2,
                   label=BIAS_LABELS[b], linestyle=s)
        for b, s, a in zip(BIASES, ['-','--',':'], [1.0,0.75,0.55])
    ]
    fig.legend(handles=legend_handles + bias_handles,
               loc='lower center', ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.08))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(OUT / 'fig_tradeoff_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved fig_tradeoff_all.png')

    # Also one plot per dataset for detail
    for dataset in ['Census', 'Credit']:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title(f'{dataset} — Fairness–Utility Tradeoff', fontsize=12, fontweight='bold')
        ax.set_xlabel('EO Gap ↓', fontsize=10)
        ax.set_ylabel('ROC-AUC ↑', fontsize=10)

        for method in METHOD_ORDER:
            xs, ys, xerrs, yerrs = [], [], [], []
            for bias in BIASES:
                df = load(RESULTS.get((dataset, bias, method)))
                if df is None:
                    continue
                eo_m, eo_s = mean_std(df, 'beta_eo_tpr_diff')
                auc_m, auc_s = mean_std(df, 'beta_roc_auc')
                ax.annotate(BIAS_LABELS[bias].split(' ')[0],
                            (eo_m, auc_m), textcoords='offset points',
                            xytext=(4, 4), fontsize=6, color=METHOD_COLORS[method], alpha=0.7)
                ax.errorbar(eo_m, auc_m, xerr=eo_s, yerr=auc_s,
                            fmt=markers[method], color=METHOD_COLORS[method],
                            alpha=bias_alpha[bias], capsize=3, markersize=8,
                            label=method if bias == 'nobias' else None)
            # Connect dots across bias levels for this method
            pts = []
            for bias in BIASES:
                df = load(RESULTS.get((dataset, bias, method)))
                if df is not None:
                    pts.append((df['beta_eo_tpr_diff'].mean(), df['beta_roc_auc'].mean()))
            if len(pts) > 1:
                xs2, ys2 = zip(*pts)
                ax.plot(xs2, ys2, color=METHOD_COLORS[method], alpha=0.3, linewidth=1)

        handles = [mpatches.Patch(color=METHOD_COLORS[m], label=m) for m in METHOD_ORDER]
        ax.legend(handles=handles, fontsize=9)
        plt.tight_layout()
        plt.savefig(OUT / f'fig_tradeoff_{dataset.lower()}.png', dpi=150, bbox_inches='tight')
        plt.close()
    print('  Saved per-dataset tradeoff plots')


# ── 3. Learning Verification ───────────────────────────────────────────────────

def make_learning_plots():
    print('Generating learning verification plots...')

    plot_cols = [
        ('fairness.eo_tpr_diff',        'EO Gap (val)', True),
        ('fairness.dp_diff',             'DP Gap (val)', True),
        ('meta.avg_reward',              'Avg Reward',   False),
        ('fairness.worst_group_beta',    'Worst-Group Loss β', True),
    ]

    available = {k: v for k, v in LEARNING_RUNS.items() if Path(v).exists()}
    if not available:
        print('  No learning curve data found, skipping')
        return

    n_runs = len(available)
    n_cols_plot = len(plot_cols)
    fig, axes = plt.subplots(n_runs, n_cols_plot,
                             figsize=(n_cols_plot * 4, n_runs * 3.5),
                             squeeze=False)
    fig.suptitle('RL Framework — Learning Verification', fontsize=13, fontweight='bold')

    for row_i, ((dataset, bias), run_dir) in enumerate(sorted(available.items())):
        result = load_learning_curves(run_dir)
        if result is None:
            continue
        mean_df, std_df, n = result
        ep = mean_df['episode'] if 'episode' in mean_df.columns else np.arange(len(mean_df))

        row_label = f'{dataset}\n{BIAS_LABELS[bias]}\n(n={n})'
        axes[row_i][0].set_ylabel(row_label, fontsize=9, labelpad=5)

        for col_i, (col, col_label, lower) in enumerate(plot_cols):
            ax = axes[row_i][col_i]
            if col not in mean_df.columns:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(col_label if row_i == 0 else '', fontsize=9)
                continue

            y  = smooth(mean_df[col].values)
            ys = smooth(std_df[col].values) if col in std_df.columns else np.zeros_like(y)

            ax.plot(ep, y, color='#B47CC7', linewidth=1.5)
            ax.fill_between(ep, y - ys, y + ys, alpha=0.2, color='#B47CC7')

            # Alpha baseline for fairness metrics
            if col == 'fairness.eo_tpr_diff' and 'fairness.eo_alpha_baseline' in mean_df.columns:
                alpha_line = smooth(mean_df['fairness.eo_alpha_baseline'].values)
                ax.plot(ep, alpha_line, color='gray', linewidth=1, linestyle='--',
                        label='α baseline')
                ax.legend(fontsize=7)
            if col == 'fairness.worst_group_beta' and 'fairness.worst_loss_alpha_baseline' in mean_df.columns:
                alpha_line = smooth(mean_df['fairness.worst_loss_alpha_baseline'].values)
                ax.plot(ep, alpha_line, color='gray', linewidth=1, linestyle='--',
                        label='α baseline')
                ax.legend(fontsize=7)

            ax.set_title(col_label if row_i == 0 else '', fontsize=9)
            ax.set_xlabel('Episode' if row_i == n_runs - 1 else '', fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / 'fig_learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved fig_learning_curves.png')


# ── 4. Reward Ablation (DVRL vs Global) ───────────────────────────────────────

def make_reward_ablation():
    print('Generating reward ablation plots...')
    # For each dataset × bias (excluding curriculum), aggregate across
    # budget configs (hireal, hisynth, scale1, scale2) — same budget, only
    # reward type differs. This gives a robust dvrl vs global comparison.

    reward_labels = {'dvrl': 'DVRL-Weighted', 'global': 'Global-Only'}
    reward_colors = {'dvrl': '#E58606', 'global': '#5D69B1'}
    metrics_ra = [
        ('beta_eo_tpr_diff', 'EO Gap ↓', True),
        ('beta_dp_diff',     'DP Gap ↓', True),
        ('beta_roc_auc',     'AUC ↑',    False),
        ('beta_acc',         'Acc ↑',    False),
    ]

    for dataset in ['Census', 'Credit']:
        for bias in BIASES:
            results_by_reward = {'dvrl': [], 'global': []}
            for variant in BUDGET_CONFIGS:
                for reward in ['dvrl', 'global']:
                    p = _budget_path(dataset, bias, variant, reward)
                    df = load(p)
                    if df is not None:
                        results_by_reward[reward].append(df)

            if not any(results_by_reward.values()):
                continue

            fig, axes = plt.subplots(1, len(metrics_ra), figsize=(14, 4))
            fig.suptitle(f'Reward Ablation: {dataset} — {BIAS_LABELS[bias]}',
                         fontsize=12, fontweight='bold')

            for ax, (col, col_label, lower) in zip(axes, metrics_ra):
                vals, errs, labels, colors = [], [], [], []
                for reward in ['dvrl', 'global']:
                    dfs = results_by_reward[reward]
                    if not dfs:
                        continue
                    all_vals = pd.concat([df[col] for df in dfs])
                    vals.append(all_vals.mean())
                    errs.append(all_vals.std())
                    labels.append(reward_labels[reward])
                    colors.append(reward_colors[reward])

                if not vals:
                    ax.set_visible(False)
                    continue

                x = np.arange(len(vals))
                bars = ax.bar(x, vals, yerr=errs, capsize=5,
                              color=colors, alpha=0.85, width=0.5,
                              edgecolor='black', linewidth=0.5)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + max(errs) * 0.05,
                            f'{v:.3f}', ha='center', va='bottom', fontsize=9)
                ax.set_title(col_label, fontsize=10)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=9)
                if lower:
                    ax.invert_yaxis()

            plt.tight_layout()
            fname = f'fig_reward_ablation_{dataset.lower()}_{bias}.png'
            plt.savefig(OUT / fname, dpi=150, bbox_inches='tight')
            plt.close()
            print(f'  Saved {fname}')


# ── 5. Synthetic Budget Ablation ───────────────────────────────────────────────

def make_budget_ablation():
    print('Generating budget ablation plots...')
    # Show EO and AUC vs budget variant for each dataset × bias
    # Use global reward only (cleaner signal, our best config)
    budget_labels = {
        'hireal':  'Hi-Real\n(2×data)',
        'hisynth': 'Hi-Synth\n(3×traj)',
        'scale1':  'Scale-1\n(1.5×both)',
        'scale2':  'Scale-2\n(3×both)',
    }
    metrics_ba = [
        ('beta_eo_tpr_diff', 'EO Gap ↓', True),
        ('beta_roc_auc',     'AUC ↑',    False),
        ('beta_acc',         'Acc ↑',    False),
    ]

    for dataset in ['Census', 'Credit']:
        fig, axes = plt.subplots(len(BIASES), len(metrics_ba),
                                 figsize=(len(metrics_ba)*4, len(BIASES)*3.5),
                                 squeeze=False)
        fig.suptitle(f'Budget Ablation: {dataset}', fontsize=13, fontweight='bold')

        for row_i, bias in enumerate(BIASES):
            for col_i, (col, col_label, lower) in enumerate(metrics_ba):
                ax = axes[row_i][col_i]
                ax.set_title(col_label if row_i == 0 else '', fontsize=10)
                ax.set_ylabel(BIAS_LABELS[bias] if col_i == 0 else '', fontsize=9)

                vals, errs, labels = [], [], []
                for variant in BUDGET_CONFIGS:
                    p = _budget_path(dataset, bias, variant, 'global')
                    df = load(p)
                    if df is not None and col in df.columns:
                        vals.append(df[col].mean())
                        errs.append(df[col].std())
                        labels.append(budget_labels[variant])

                if not vals:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            transform=ax.transAxes, fontsize=9, color='gray')
                    continue

                x = np.arange(len(vals))
                bars = ax.bar(x, vals, yerr=errs, capsize=4,
                              color='#B47CC7', alpha=0.8, width=0.6,
                              edgecolor='black', linewidth=0.5)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + (max(errs) if errs else 0)*0.05,
                            f'{v:.3f}', ha='center', va='bottom', fontsize=8)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=8)

        plt.tight_layout()
        fname = f'fig_budget_ablation_{dataset.lower()}.png'
        plt.savefig(OUT / fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved {fname}')


# ── 6. Delta Action Ablation ───────────────────────────────────────────────────

def make_delta_ablation():
    print('Generating delta action ablation...')
    # Census and credit, bias=0.10 only (where delta ablation was run)
    # δ in {0.05, 0.10 (default), 0.20}
    # Compare global reward across delta scales; also show dvrl for reference
    delta_variants = {
        'ds005': 'δ=0.05',
        'ds010': 'δ=0.10\n(default)',
        'ds020': 'δ=0.20',
    }
    metrics_da = [
        ('beta_eo_tpr_diff', 'EO Gap ↓', True),
        ('beta_roc_auc',     'AUC ↑',    False),
        ('beta_acc',         'Acc ↑',    False),
    ]

    for dataset in ['Census', 'Credit']:
        fig, axes = plt.subplots(1, len(metrics_da), figsize=(12, 4))
        fig.suptitle(f'Delta Action Scale Ablation: {dataset} (bias=10%)',
                     fontsize=12, fontweight='bold')

        for ax, (col, col_label, lower) in zip(axes, metrics_da):
            ax.set_title(col_label, fontsize=10)
            for reward, style, color in [('global', '-', '#5D69B1'), ('dvrl', '--', '#E58606')]:
                vals, errs, xs = [], [], []
                for vi, (ds_str, ds_label) in enumerate(delta_variants.items()):
                    p = _delta_path(dataset, ds_str, reward)
                    df = load(p)
                    if df is not None and col in df.columns:
                        vals.append(df[col].mean())
                        errs.append(df[col].std())
                        xs.append(vi)
                if vals:
                    lbl = 'Global-Only' if reward == 'global' else 'DVRL-Weighted'
                    ax.errorbar(xs, vals, yerr=errs, fmt='o'+style,
                                color=color, capsize=4, label=lbl, linewidth=1.5)

            ax.set_xticks(range(len(delta_variants)))
            ax.set_xticklabels(list(delta_variants.values()), fontsize=9)
            ax.set_xlabel('Delta Scale', fontsize=9)
            ax.legend(fontsize=8)

        plt.tight_layout()
        fname = f'fig_delta_ablation_{dataset.lower()}.png'
        plt.savefig(OUT / fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved {fname}')


# ── 7. FFNN Epoch Ablation ─────────────────────────────────────────────────────

def make_ffnn_ablation():
    print('Generating FFNN epoch ablation...')
    epochs = [5, 20, 50]
    metrics_fa = [
        ('beta_eo_tpr_diff', 'EO Gap ↓', True),
        ('beta_dp_diff',     'DP Gap ↓', True),
        ('beta_roc_auc',     'AUC ↑',    False),
        ('beta_acc',         'Acc ↑',    False),
        ('beta_f1_weighted', 'F1w ↑',    False),
    ]

    for dataset in ['Census', 'Credit']:
        fig, axes = plt.subplots(1, len(metrics_fa), figsize=(16, 4))
        fig.suptitle(f'FFNN Epoch Ablation: {dataset} (bias=10%)',
                     fontsize=12, fontweight='bold')

        for ax, (col, col_label, lower) in zip(axes, metrics_fa):
            ax.set_title(col_label, fontsize=10)
            vals, errs, labels = [], [], []
            for ep in epochs:
                p = _ffnn_path(dataset, ep)
                df = load(p)
                if df is not None and col in df.columns:
                    vals.append(df[col].mean())
                    errs.append(df[col].std())
                    labels.append(f'ep={ep}')

            if not vals:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='gray')
                continue

            x = np.arange(len(vals))
            bar_colors = ['#cccccc' if l != 'ep=20' else '#B47CC7' for l in labels]
            bars = ax.bar(x, vals, yerr=errs, capsize=4, color=bar_colors,
                          alpha=0.85, width=0.6, edgecolor='black', linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (max(errs) if errs else 0)*0.05,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_xlabel('β classifier epochs', fontsize=9)

        # Legend: highlight ep=20 as selected
        axes[-1].annotate('★ selected', xy=(0, 0), xycoords='axes fraction',
                          fontsize=8, color='#B47CC7', xytext=(0, -0.2))
        plt.tight_layout()
        fname = f'fig_ffnn_ablation_{dataset.lower()}.png'
        plt.savefig(OUT / fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved {fname}')


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f'Output directory: {OUT}\n')
    make_tables()
    print()
    make_tradeoff_plots()
    print()
    make_learning_plots()
    print()
    make_reward_ablation()
    print()
    make_budget_ablation()
    print()
    make_delta_ablation()
    print()
    make_ffnn_ablation()
    print('\nDone. All outputs in paper_figures/')
