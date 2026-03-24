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
MAIN_TR   = ROOT / 'training_runs' / 'paper_results' / 'main_results'
FFNN_TR   = ROOT / 'training_runs' / 'paper_results' / 'ffnn_ablation'
BUDGET_TR = ROOT / 'training_runs' / 'paper_results' / 'budget_ablation'
DELTA_TR  = ROOT / 'training_runs' / 'paper_results' / 'delta_ablation'
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
        _s(MAIN_TR/'BASELINE_group_dro_p1_nobias_census_gdro_5s_5951e26e__G202603201755/final_test_metrics.csv'),
    ('Census', 'nobias', 'OT Repair'):
        _s(MAIN_TR/'BASELINE_gaussian_ot_repair_p1_nobias_census_otrep_5s_216fbc09__G202603201755/final_test_metrics.csv'),
    ('Census', 'nobias', 'CT-GAN'):
        _s(MAIN_TR/'BASELINE_ctgan_p1_nobias_census_ctgan_5s_900a27b1__G202603201756/final_test_metrics.csv'),
    ('Census', 'nobias', 'RL Framework'):
        _s(MAIN_TR/'SPECp1_nobias_census_ours_ep20_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_GG202603202304_352917e1/final_test_metrics.csv'),

    ('Census', '005', 'Group DRO'):
        _s(MAIN_TR/'BASELINE_group_dro_p1_census_bias05_gdro_5s_8bdc58e8__G202603211520/final_test_metrics.csv'),
    ('Census', '005', 'OT Repair'):
        _s(MAIN_TR/'BASELINE_gaussian_ot_repair_p1_census_bias05_otrep_5s_ep20_539e2739__G202603201755/final_test_metrics.csv'),
    ('Census', '005', 'CT-GAN'):
        _s(MAIN_TR/'BASELINE_ctgan_p1_census_bias05_ctgan_5s_ep20_6db6db20__G202603201756/final_test_metrics.csv'),
    ('Census', '005', 'RL Framework'):
        _s(MAIN_TR/'SPECp1_main_census_bias05_global_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.05_GG202603200009_28c96eee/final_test_metrics.csv'),

    ('Census', '010', 'Group DRO'):
        _s(MAIN_TR/'BASELINE_group_dro_p1_census_bias010_gdro_5s_c5421696__G202603211538/final_test_metrics.csv'),
    ('Census', '010', 'OT Repair'):
        _s(MAIN_TR/'BASELINE_gaussian_ot_repair_p1_census_bias010_otrep_5s_ep20_2084dedc__G202603201755/final_test_metrics.csv'),
    ('Census', '010', 'CT-GAN'):
        _s(MAIN_TR/'BASELINE_ctgan_p1_census_bias010_ctgan_5s_ep20_579aa0ca__G202603201756/final_test_metrics.csv'),
    ('Census', '010', 'RL Framework'):
        _s(FFNN_TR/'SPECp1_ffnn_census_ep20_bias010_global_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.1_GG202603192345_591446a3/final_test_metrics.csv'),

    # ── Credit ──
    ('Credit', 'nobias', 'Group DRO'):
        _s(MAIN_TR/'BASELINE_group_dro_p1_nobias_credit_gdro_5s_fc6d4e7b__G202603211540/final_test_metrics.csv'),
    ('Credit', 'nobias', 'OT Repair'):
        _s(MAIN_TR/'BASELINE_gaussian_ot_repair_p1_nobias_credit_otrep_5s_8edcf461__G202603211520/final_test_metrics.csv'),
    ('Credit', 'nobias', 'CT-GAN'):
        _s(MAIN_TR/'BASELINE_ctgan_p1_nobias_credit_ctgan_5s_eb4604c7__G202603211520/final_test_metrics.csv'),
    ('Credit', 'nobias', 'RL Framework'):
        _s(MAIN_TR/'SPECp1_nobias_credit_ours_ep20_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_GG202603211520_ed14ca96/final_test_metrics.csv'),

    ('Credit', '005', 'Group DRO'):
        _s(MAIN_TR/'BASELINE_group_dro_p1_credit_bias05_gdro_5s_3e922b9b__G202603211542/final_test_metrics.csv'),
    ('Credit', '005', 'OT Repair'):
        _s(MAIN_TR/'BASELINE_gaussian_ot_repair_p1_credit_bias05_otrep_5s_ep20_33727f8e__G202603201755/final_test_metrics.csv'),
    ('Credit', '005', 'CT-GAN'):
        _s(MAIN_TR/'BASELINE_ctgan_p1_credit_bias05_ctgan_5s_ep20_0bd0101c__G202603211545/final_test_metrics.csv'),
    ('Credit', '005', 'RL Framework'):
        _s(MAIN_TR/'SPECp1_main_credit_bias05_global_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.05_GG202603200009_501656c4/final_test_metrics.csv'),

    ('Credit', '010', 'Group DRO'):
        _s(MAIN_TR/'BASELINE_group_dro_p1_credit_bias010_gdro_5s_dcca27f2__G202603211552/final_test_metrics.csv'),
    ('Credit', '010', 'OT Repair'):
        _s(MAIN_TR/'BASELINE_gaussian_ot_repair_p1_credit_bias010_otrep_5s_ep20_d580f330__G202603201755/final_test_metrics.csv'),
    ('Credit', '010', 'CT-GAN'):
        _s(MAIN_TR/'BASELINE_ctgan_p1_credit_bias010_ctgan_5s_ep20_49701d91__G202603211545/final_test_metrics.csv'),
    ('Credit', '010', 'RL Framework'):
        _s(MAIN_TR/'SPECp1_main_credit_bias010_global_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.1_GG202603200009_abfd28d9/final_test_metrics.csv'),
}

# Learning curve run directories (seed-level, contains metrics.csv)
LEARNING_RUNS = {
    ('Census', 'nobias'): str(MAIN_TR / 'SPECp1_nobias_census_ours_ep20_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_GG202603202304_352917e1'),
    ('Census', '010'):    str(FFNN_TR / 'SPECp1_ffnn_census_ep20_bias010_global_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.1_GG202603192345_591446a3'),
    ('Credit', '010'):    str(MAIN_TR / 'SPECp1_main_credit_bias010_global_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.1_GG202603200009_abfd28d9'),
    ('Credit', '005'):    str(MAIN_TR / 'SPECp1_main_credit_bias05_global_5s_EP800_PCA10_REWfairness_minID1_majID0_TRJ2000_REAL3000_BIAS0.05_GG202603200009_501656c4'),
}

# Reward ablation: dvrl vs global, across budget configs (no curriculum)
# Format: (dataset, bias, budget_variant) -> {dvrl: path, global: path}
BUDGET_CONFIGS = ['hireal', 'hisynth', 'scale1', 'scale2']

def _budget_path(dataset, bias, variant, reward):
    ds  = dataset.lower()
    pattern = f'SPECp1_budget_{ds}_{variant}_{bias}_{reward}_5s_EP800'
    for d in BUDGET_TR.iterdir():
        if d.name.startswith(pattern):
            p = d / 'final_test_metrics.csv'
            return str(p) if p.exists() else None
    return None

def _ffnn_path(dataset, ep):
    ds = dataset.lower()
    pattern = f'SPECp1_ffnn_{ds}_ep{ep:02d}_bias010_global_5s'
    for d in FFNN_TR.iterdir():
        if d.name.startswith(pattern):
            p = d / 'final_test_metrics.csv'
            return str(p) if p.exists() else None
    return None

def _delta_path(dataset, ds_str, reward):
    ds = dataset.lower()
    pattern = f'SPECp1_delta_{ds}_{ds_str}_bias010_{reward}_5s'
    for d in DELTA_TR.iterdir():
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

def alpha_fus(df):
    """FUS computed from alpha (no-intervention) columns."""
    if df is None or 'alpha_eo_tpr_diff' not in df.columns:
        return None, None
    score = (1 - df['alpha_eo_tpr_diff']) * df['alpha_roc_auc']
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
            if 'meta.phase' in df.columns:
                df = df[df['meta.phase'] == 'phase1_class1'].reset_index(drop=True)
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

DATASETS     = ['Census', 'Credit', 'HAR']
BIASES       = ['nobias', '005', '010']   # full set, used for degradation plot
MAIN_BIASES  = ['nobias', '010']          # shown in main tables and tradeoff plots
METHODS      = ['Group DRO', 'OT Repair', 'CT-GAN', 'RL Framework']

# Maps beta metric columns to their alpha (no-intervention) equivalents
ALPHA_COL_MAP = {
    'beta_eo_tpr_diff':  'alpha_eo_tpr_diff',
    'beta_dp_diff':      'alpha_dp_diff',
    'beta_eod_max_diff': 'alpha_eod_max_diff',
    'beta_roc_auc':      'alpha_roc_auc',
    'beta_acc':          'alpha_acc',
    'beta_f1_weighted':  'alpha_f1_weighted',
    'beta_f1_minority':  'alpha_f1_minority',
    'beta_brier':        'alpha_brier',
}


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
        n_rows = len(METHODS) + 1  # +1 for No Intervention
        # ── No Intervention row (alpha model metrics) ──
        ni_prefix = r'\multirow{' + str(n_rows) + r'}{*}{' + dataset + '}'
        ni_row = f'{ni_prefix} & \\textit{{No Intervention}}'
        if dataset == 'HAR':
            ni_row += ' & --' * n_metrics
        else:
            ni_df = load(RESULTS.get((dataset, bias, 'RL Framework')))
            ni_prov = provisional(ni_df)
            for col, _, _ in metrics_list:
                alpha_col = ALPHA_COL_MAP.get(col, col)
                ni_row += ' & ' + fmt_cell(ni_df, alpha_col, bold=False, prov=ni_prov)
            if include_fus:
                ni_fus_m, ni_fus_s = alpha_fus(ni_df)
                if ni_fus_m is not None:
                    dag = r'\dagger' if ni_prov else ''
                    ni_row += ' & ' + f'{ni_fus_m:.3f}$_{{\\pm {ni_fus_s:.3f}{dag}}}$'
                else:
                    ni_row += ' & --'
        lines.append(ni_row + r' \\')

        for mi, method in enumerate(METHODS):
            row = f' & {method}'
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
        r'\end{table}',
    ]
    return '\n'.join(lines)


def make_tables():
    print('Generating tables...')

    bias_titles = {
        'nobias': 'No Bias',
        '010':    'High Bias (10\\%)',
    }

    for suffix, metrics_list, include_fus in [
        ('compressed', COMPRESSED_METRICS, True),
        ('full',       FULL_METRICS,       False),
    ]:
        blocks = []
        for bias, bias_title in bias_titles.items():
            if bias == '010' and include_fus:
                cap_c = r'Fairness and Utility Performance under High Bias (10\%).'
            else:
                cap_c = f'Results under {bias_title}.'
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

MARKERS = {'Group DRO': 'o', 'OT Repair': 's', 'CT-GAN': '^', 'RL Framework': 'D'}
BIAS_DISPLAY = {
    'nobias': 'No Bias',
    '005':    'Moderate Bias (5%)',
    '010':    'High Bias (10%)',
}
PLOT_DATASETS = ['Census', 'Credit']


def _draw_tradeoff_ax(ax, dataset, bias):
    """Plot all four methods for one (dataset, bias) panel."""
    for method in METHOD_ORDER:
        df = load(RESULTS.get((dataset, bias, method)))
        if df is None:
            continue
        eo_m, eo_s   = mean_std(df, 'beta_eo_tpr_diff')
        auc_m, auc_s = mean_std(df, 'beta_roc_auc')
        ax.errorbar(eo_m, auc_m, xerr=eo_s, yerr=auc_s,
                    fmt=MARKERS[method], color=METHOD_COLORS[method],
                    capsize=3, markersize=8, linestyle='none',
                    label=method)

    ax.set_xlabel('EO Gap ↓', fontsize=9)
    ax.set_ylabel('ROC-AUC ↑', fontsize=9)
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

    # Ideal region indicator — top-left corner (low EO, high AUC)
    ax.annotate('', xy=(0.04, 0.96), xytext=(0.14, 0.86),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='#888888',
                                lw=1.2, connectionstyle='arc3,rad=0.0'))
    ax.text(0.15, 0.85, 'Ideal', transform=ax.transAxes,
            fontsize=7, color='#888888', va='top', ha='left', style='italic')


def make_tradeoff_plots():
    print('Generating fairness-utility tradeoff plots...')

    legend_handles = [
        mpatches.Patch(color=METHOD_COLORS[m], label=m) for m in METHOD_ORDER
    ]

    # Individual plots — one per (dataset, bias)
    individual_figs = {}
    for dataset in PLOT_DATASETS:
        for bias in MAIN_BIASES:
            fig, ax = plt.subplots(figsize=(4.5, 4))
            _draw_tradeoff_ax(ax, dataset, bias)
            ax.set_title(f'{dataset} — {BIAS_DISPLAY[bias]}', fontsize=10, fontweight='bold')
            ax.legend(handles=legend_handles, fontsize=8, loc='best')
            plt.tight_layout()
            fname = f'fig_tradeoff_{dataset.lower()}_{bias}.png'
            plt.savefig(OUT / fname, dpi=150, bbox_inches='tight')
            individual_figs[(dataset, bias)] = fname
            plt.close()
            print(f'  Saved {fname}')

    # Grid plot — 2 rows (datasets) × 2 cols (bias levels)
    fig, axes = plt.subplots(
        len(PLOT_DATASETS), len(MAIN_BIASES),
        figsize=(9, 8),
        sharey='row',
    )
    fig.suptitle('Fairness–Utility Tradeoff (EO vs ROC-AUC)', fontsize=13, fontweight='bold')

    for row, dataset in enumerate(PLOT_DATASETS):
        for col, bias in enumerate(MAIN_BIASES):
            ax = axes[row][col]
            _draw_tradeoff_ax(ax, dataset, bias)
            if row == 0:
                ax.set_title(BIAS_DISPLAY[bias], fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{dataset}\nROC-AUC ↑', fontsize=9)
            else:
                ax.set_ylabel('')
            if row < len(PLOT_DATASETS) - 1:
                ax.set_xlabel('')

    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(OUT / 'fig_tradeoff_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved fig_tradeoff_all.png')


# ── 3. EO Degradation vs Bias Level ───────────────────────────────────────────

BIAS_X = {'nobias': 0, '010': 10}   # numeric x positions (%)

def make_bias_degradation_plot():
    print('Generating EO vs bias level plot...')

    fig, axes = plt.subplots(1, len(PLOT_DATASETS), figsize=(11, 4.5), sharey=False)
    fig.suptitle('EO Gap vs Positive-Class Scarcity Level', fontsize=13, fontweight='bold')

    for ax, dataset in zip(axes, PLOT_DATASETS):
        for method in METHOD_ORDER:
            xs, ys, errs = [], [], []
            for bias in ['nobias', '010']:
                df = load(RESULTS.get((dataset, bias, method)))
                if df is None:
                    continue
                m, s = mean_std(df, 'beta_eo_tpr_diff')
                xs.append(BIAS_X[bias])
                ys.append(m)
                errs.append(s)

            if not xs:
                continue

            color = METHOD_COLORS[method]
            ax.plot(xs, ys, marker=MARKERS[method], color=color,
                    linewidth=2, markersize=8, label=method)
            ax.fill_between(xs,
                            [y - e for y, e in zip(ys, errs)],
                            [y + e for y, e in zip(ys, errs)],
                            color=color, alpha=0.12)

        ax.set_title(dataset, fontsize=11, fontweight='bold')
        ax.set_xlabel('Positive-Class Scarcity (bias %)', fontsize=9)
        ax.set_ylabel('EO Gap ↓  (lower is fairer)', fontsize=9)
        ax.set_xticks([0, 10])
        ax.set_xticklabels(['0%\n(No Bias)', '10%\n(High)'], fontsize=8)
        ax.set_xlim(-1, 11)
        ax.set_ylim(bottom=0)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

        # Annotate the ideal direction
        ax.annotate('Lower = fairer', xy=(0.02, 0.08), xycoords='axes fraction',
                    fontsize=7, color='#888888', style='italic')

    handles = [mpatches.Patch(color=METHOD_COLORS[m], label=m) for m in METHOD_ORDER]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(OUT / 'fig_eo_vs_bias.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  Saved fig_eo_vs_bias.png')


# ── 4. Learning Verification ───────────────────────────────────────────────────

def make_learning_plots():
    print('Generating learning verification plots...')

    plot_cols = [
        ('fairness.eo_tpr_diff',        'EO Gap (val)', True),
        ('fairness.dp_diff',             'DP Gap (val)', True),
        ('meta.episode_return',          'Episode Return', False),
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


# ── 4. Local vs Global Reward Ablation ────────────────────────────────────────
# Each of the four budget variants (hireal, hisynth, scale1, scale2) was run
# with both DVRL-weighted and Global-only reward — identical in every other way.
# Aggregating across variants gives a robust paired estimate of reward-type
# effect while marginalising over budget choices.

def make_local_global_ablation():
    print('Generating local vs global reward ablation...')

    reward_labels = {'dvrl': 'Local+Global\n(DVRL)', 'global': 'Global-Only'}
    reward_colors = {'dvrl': '#E58606', 'global': '#5D69B1'}
    metrics_lg = [
        ('beta_eo_tpr_diff', 'EO Gap ↓',  True),
        ('beta_dp_diff',     'DP Gap ↓',  True),
        ('beta_roc_auc',     'AUC ↑',     False),
        ('beta_acc',         'Acc ↑',     False),
        ('beta_f1_weighted', 'F1$_w$ ↑',  False),
    ]

    # One figure per dataset; rows = bias levels that have data
    for dataset in ['Census', 'Credit']:
        bias_with_data = []
        rows_data = {}
        for bias in BIASES:
            by_reward = {'dvrl': [], 'global': []}
            for variant in BUDGET_CONFIGS:
                for reward in ['dvrl', 'global']:
                    p = _budget_path(dataset, bias, variant, reward)
                    df = load(p)
                    if df is not None:
                        by_reward[reward].append(df)
            if any(by_reward.values()):
                bias_with_data.append(bias)
                rows_data[bias] = by_reward

        if not bias_with_data:
            continue

        n_rows = len(bias_with_data)
        n_cols = len(metrics_lg)
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 2.8, n_rows * 3.2),
                                 squeeze=False)
        fig.suptitle(f'Local vs Global Reward — {dataset}',
                     fontsize=12, fontweight='bold')

        for row_i, bias in enumerate(bias_with_data):
            by_reward = rows_data[bias]
            axes[row_i][0].set_ylabel(BIAS_LABELS[bias], fontsize=9)

            for col_i, (col, col_label, lower) in enumerate(metrics_lg):
                ax = axes[row_i][col_i]
                if row_i == 0:
                    ax.set_title(col_label, fontsize=10)

                vals, errs, labels, colors = [], [], [], []
                for reward in ['dvrl', 'global']:
                    dfs = by_reward[reward]
                    if not dfs:
                        continue
                    all_vals = pd.concat([df[col] for df in dfs
                                          if col in df.columns])
                    if all_vals.empty:
                        continue
                    vals.append(all_vals.mean())
                    errs.append(all_vals.std())
                    labels.append(reward_labels[reward])
                    colors.append(reward_colors[reward])

                if not vals:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            transform=ax.transAxes, fontsize=9, color='gray')
                    continue

                x = np.arange(len(vals))
                bars = ax.bar(x, vals, yerr=errs, capsize=4,
                              color=colors, alpha=0.85, width=0.5,
                              edgecolor='black', linewidth=0.5)
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (max(errs) if errs else 0) * 0.05,
                            f'{v:.3f}', ha='center', va='bottom', fontsize=8)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, fontsize=8)

        plt.tight_layout()
        fname = f'fig_local_global_{dataset.lower()}.png'
        plt.savefig(OUT / fname, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved {fname}')


# ── 4b. Delta Actions vs Exact-Point Generation ────────────────────────────────
# Lookup helper for no-delta ablation runs.
NODELTA_TR = ROOT / 'training_runs' / 'paper_results' / 'nodelta_ablation'

def _nodelta_path(dataset, bias):
    """Return path to final_test_metrics.csv for a no-delta run, or None."""
    ds = dataset.lower()
    pattern = f'SPECp1_nodelta_{ds}_{bias}_'
    for root_dir in [NODELTA_TR, ROOT / 'training_runs']:
        if not root_dir.exists():
            continue
        for d in root_dir.iterdir():
            if d.name.startswith(pattern):
                p = d / 'final_test_metrics.csv'
                if p.exists():
                    return str(p)
    return None


def make_delta_vs_nodelta():
    print('Generating delta vs no-delta (exact point) ablation...')

    metrics_dn = [
        ('beta_eo_tpr_diff', 'EO Gap ↓',  True),
        ('beta_dp_diff',     'DP Gap ↓',  True),
        ('beta_roc_auc',     'AUC ↑',     False),
        ('beta_acc',         'Acc ↑',     False),
        ('beta_f1_weighted', 'F1$_w$ ↑',  False),
    ]
    variant_labels  = ['Delta\n(default)', 'Exact\nPoint']
    variant_colors  = ['#B47CC7', '#888888']

    # Census bias=010 only (where the ablation was run)
    dataset, bias = 'Census', '010'

    # Delta = main ep20 result; no-delta = nodelta ablation
    delta_path   = RESULTS.get((dataset, bias, 'RL Framework'))
    nodelta_path = _nodelta_path(dataset, bias)

    if delta_path is None and nodelta_path is None:
        print('  No data found for delta vs no-delta ablation, skipping')
        return

    df_delta   = load(delta_path)
    df_nodelta = load(nodelta_path)

    fig, axes = plt.subplots(1, len(metrics_dn), figsize=(len(metrics_dn) * 2.8, 3.8))
    fig.suptitle(f'Delta Actions vs Exact-Point Generation — {dataset} (bias={bias[:-1] if bias != "nobias" else "0"}%)',
                 fontsize=11, fontweight='bold')

    for ax, (col, col_label, lower) in zip(axes, metrics_dn):
        ax.set_title(col_label, fontsize=10)
        vals, errs, labels, colors = [], [], [], []

        for df, lbl, color in [(df_delta, variant_labels[0], variant_colors[0]),
                                (df_nodelta, variant_labels[1], variant_colors[1])]:
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
        bars = ax.bar(x, vals, yerr=errs, capsize=4,
                      color=colors, alpha=0.85, width=0.5,
                      edgecolor='black', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(errs) if errs else 0) * 0.05,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)

        # Mark which is better
        if len(vals) == 2:
            better_idx = (0 if vals[0] < vals[1] else 1) if lower else \
                         (0 if vals[0] > vals[1] else 1)
            ax.get_children()[better_idx].set_edgecolor('#2ca02c')
            ax.get_children()[better_idx].set_linewidth(2.0)

    # Footnote: seed count per variant
    n_delta   = len(df_delta)   if df_delta   is not None else 0
    n_nodelta = len(df_nodelta) if df_nodelta is not None else 0
    fig.text(0.01, 0.01, f'n seeds — Delta: {n_delta}, Exact Point: {n_nodelta}',
             fontsize=7, color='gray')

    plt.tight_layout()
    fname = 'fig_delta_vs_nodelta.png'
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
    make_bias_degradation_plot()
    print()
    make_learning_plots()
    print()
    make_local_global_ablation()
    print()
    make_delta_vs_nodelta()
    print()
    make_budget_ablation()
    print()
    make_delta_ablation()
    print()
    make_ffnn_ablation()
    print('\nDone. All outputs in paper_figures/')
