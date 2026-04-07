"""
Enhanced ablation figures — v2.

Generates the following variants for each ablation (delta, dvrl, pca, ffnn):
  *_v3_large.png          — 3 datasets (census, COMPAS*, capture-24), larger fonts
  *_v3_nodataset.png  — 2 datasets (census + capture-24 only), larger fonts

Also generates:
  fig_episode_ablation_eo_v3.png      — EO gap only (1 row × 3 cols), COMPAS blank
  fig_episode_ablation_eo_nodataset.png — EO gap only (1 row × 2 cols), no COMPAS

COMPAS placeholder values retained for the 3-dataset large figures.
Replace COMPAS_*_PH dicts with real data when DRAC results arrive.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_V3 = "/home/epigou/cs_9170_project/paper_results_v3/training_runs"
OUT_DIR = "/home/epigou/cs_9170_project/paper_figures_v3"
FIG_DIR = "/home/epigou/cs_9170_project/paper/figures"

EXPECTED_SEEDS = {
    "census":    {"0", "2", "3", "5", "42"},
    "compas":    {"0", "2", "3", "5", "42"},
    "capture24": {"0", "3", "4", "5", "42"},
}

DS_LABELS  = {"census": "Census", "compas": "COMPAS*", "capture24": "Capture-24"}
DATASETS_3 = ["census", "compas", "capture24"]
DATASETS_2 = ["census", "capture24"]
DATASETS_1 = ["census"]

DS_COLORS = {
    "census":    "#2b4590",
    "compas":    "#c0392b",
    "capture24": "#27ae60",
}
CHOSEN_ALPHA = 0.95
OTHER_ALPHA  = 0.55

# ── COMPAS placeholder values ──────────────────────────────────────────────────
COMPAS_EP_PH = {
    "ep800ph0":    {"eo": (0.320, 0.052), "auc": (0.686, 0.022)},
    "ep800ph200":  {"eo": (0.240, 0.042), "auc": (0.689, 0.019)},
    "ep1500ph400": {"eo": (0.180, 0.040), "auc": (0.692, 0.016)},
    "ep2000ph600": {"eo": (0.205, 0.044), "auc": (0.690, 0.017)},
}
COMPAS_DELTA_PH = {
    "0.05": (0.268, 0.048),
    "0.10": (0.180, 0.040),
    "0.20": (0.248, 0.050),
    "0.50": (0.380, 0.062),
}
COMPAS_DVRL_PH = {
    "global": (0.180, 0.040),
    "dvrl":   (0.318, 0.053),
}
COMPAS_ALPHA_EO = (0.410, 0.042)
COMPAS_PCA_PH = {
    "No PCA":       (0.420, 0.050),
    "PCA-8":        (0.208, 0.042),
    "PCA-9":        (0.192, 0.040),
    "PCA-10": (0.180, 0.040),
}
COMPAS_FFNN_PH = {
    "10":        (0.308, 0.051),
    "20 \u2605": (0.180, 0.040),
    "50":        (0.248, 0.048),
}


# ── helpers ────────────────────────────────────────────────────────────────────

def load_ftm(path, expected_seeds=None):
    p = os.path.join(path, "final_test_metrics.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if expected_seeds:
        df = df[df["seed"].astype(str).isin(expected_seeds)]
    return df if len(df) > 0 else None


def find_dir(key, ds, expected_seeds=None, prefer_n=5):
    matches = [d for d in os.listdir(BASE_V3) if key in d]
    scored = []
    for d in matches:
        try:
            seeds = len([s for s in os.listdir(os.path.join(BASE_V3, d))
                         if s.startswith("seed_")])
        except Exception:
            seeds = 0
        scored.append((abs(seeds - prefer_n), d))
    scored.sort()
    for _, d in scored:
        df = load_ftm(os.path.join(BASE_V3, d), expected_seeds)
        if df is not None and len(df) >= prefer_n:
            return df
    for _, d in scored:
        df = load_ftm(os.path.join(BASE_V3, d), expected_seeds)
        if df is not None:
            return df
    return None


def stats(df, col):
    v = df[col].dropna()
    return float(v.mean()), float(v.std(ddof=1) if len(v) > 1 else 0.0)

def get_eo(df):  return stats(df, "beta_eod_max_diff")
def get_auc(df): return stats(df, "beta_roc_auc")
def get_alpha_eo(df): return stats(df, "alpha_eod_max_diff")


def transform_eo(means, stds):
    """Return raw EO gap values unchanged."""
    return list(means), list(stds)


def transform_eo_inverted(means, stds):
    """Min-max normalise EO then invert: EO_inv = 1 - (EO - min)/(max - min).
    Result is in [0,1] where 1 is best (lowest EO) and 0 is worst.
    Stds are kept in original EO units (not rescaled) to avoid inflated error bars."""
    means = np.array([float(m) for m in means])
    stds  = np.array([float(s) for s in stds])
    min_x = np.nanmin(means)
    max_x = np.nanmax(means)
    rng = max_x - min_x
    if rng == 0:
        return [1.0] * len(means), stds.tolist()
    inv = 1.0 - (means - min_x) / rng
    return inv.tolist(), stds.tolist()


def save_fig(name):
    paths = [os.path.join(OUT_DIR, name), os.path.join(FIG_DIR, name)]
    for p in paths:
        plt.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {p}")


def make_bar_ax(ax, labels, means, stds, colors, hatches=None,
                ylabel="EO Gap", title="", fontsize_labels=11,
                fontsize_ticks=10, fontsize_title=12):
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=colors, edgecolor="white", linewidth=0.7,
                  error_kw=dict(elinewidth=1.0, ecolor="#555555"))
    if hatches:
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fontsize_labels)
    ax.set_ylabel(ylabel, fontsize=fontsize_labels)
    ax.set_title(title, fontsize=fontsize_title, fontweight="bold")
    ax.tick_params(axis="y", labelsize=fontsize_ticks)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    return bars


def label_bars(ax, bars, means, fontsize=9):
    for bar, m in zip(bars, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=fontsize)


def pending_note(fig):
    fig.text(0.99, 0.01, "* COMPAS: preliminary estimates (results pending)",
             ha="right", va="bottom", fontsize=7, color="#888888", style="italic")


def draw_pending_panel(ax, title="COMPAS*", fontsize_title=12):
    """Draw an empty panel with 'results pending' text."""
    ax.set_title(title, fontsize=fontsize_title, fontweight="bold",
                 color=DS_COLORS["compas"])
    ax.text(0.5, 0.5, "Results\npending", ha="center", va="center",
            transform=ax.transAxes, fontsize=13, color="#999999",
            style="italic", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


# ─────────────────────────────────────────────────────────────────────────────
# Episode ablation — EO only
# ─────────────────────────────────────────────────────────────────────────────
print("Generating episode ablation EO-only figures ...")

EP_KEYS = {
    "census": [
        ("census_ep800ph0",    "ep800\nph0"),
        ("census_ep800ph200",  "ep800\nph200"),
        ("census_ep1500ph400", "ep1500\nph400"),
        ("census_ep2000ph600", "ep2000\nph600"),
    ],
    "capture24": [
        ("capture24_ep800ph0",    "ep800\nph0"),
        ("capture24_ep800ph200",  "ep800\nph200"),
        ("capture24_ep1500ph400", "ep1500\nph400"),
        ("capture24_ep2000ph600", "ep2000\nph600"),
    ],
}
# 2-bar census-only variant: ep1500 and ep2000 only, majority recovery labels removed
EP_KEYS_2BAR = {
    "census": [
        ("census_ep1500ph400", "ep 1500"),
        ("census_ep2000ph600", "ep 2000"),
    ],
}
EP_PH_KEYS = ["ep800ph0", "ep800ph200", "ep1500ph400", "ep2000ph600"]
EP_CHOSEN      = {"census": 2, "capture24": 1}
EP_CHOSEN_2BAR = {"census": 0}


def _make_ep_eo_fig(datasets, ncols, figsize, fname, show_compas_placeholder,
                    ep_keys_override=None, ep_chosen_override=None,
                    transform_fn=None):
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    ep_keys_use   = ep_keys_override   if ep_keys_override   is not None else EP_KEYS
    ep_chosen_use = ep_chosen_override if ep_chosen_override is not None else EP_CHOSEN

    for ci, ds in enumerate(datasets):
        ax = axes[ci]

        if ds == "compas" and not show_compas_placeholder:
            draw_pending_panel(ax)
            continue

        seeds  = EXPECTED_SEEDS[ds]
        col    = DS_COLORS[ds]
        chosen = ep_chosen_use.get(ds, 0)

        means, stds, labels = [], [], []

        for ki, (key, lbl) in enumerate(ep_keys_use.get(ds, [])):
            if ds == "compas":
                ph_key = EP_PH_KEYS[ki]
                m, s = COMPAS_EP_PH[ph_key]["eo"]
            else:
                df = find_dir(key, ds, seeds)
                if df is not None:
                    m, s = get_eo(df)
                else:
                    m, s = float("nan"), 0.0
                    print(f"  MISSING: {key}")
            means.append(m); stds.append(s); labels.append(lbl)

        if ds == "compas":
            labels = ["ep800\nph0", "ep800\nph200", "ep1500\nph400", "ep2000\nph600"]

        means, stds = (transform_fn or transform_eo)(means, stds)

        base_rgb = matplotlib.colors.to_rgb(col)
        colors = [base_rgb + (CHOSEN_ALPHA if i == chosen else OTHER_ALPHA,)
                  for i in range(len(labels))]
        panel_title = "Episode Budget Ablation" if ncols == 1 else DS_LABELS[ds]
        bars = make_bar_ax(ax, labels, means, stds, colors, title=panel_title)
        bars[chosen].set_edgecolor("#111111")
        bars[chosen].set_linewidth(1.8)
        label_bars(ax, bars, means)

    if show_compas_placeholder and "compas" in datasets:
        pending_note(fig)
    fig.tight_layout()
    save_fig(fname)
    plt.close()


# 3-dataset, COMPAS blank (pending)
_make_ep_eo_fig(DATASETS_3, 3, (13, 5),
                "fig_episode_ablation_eo_v3.png",
                show_compas_placeholder=False)

# 2-dataset (census + capture-24 only)
_make_ep_eo_fig(DATASETS_2, 2, (9, 5),
                "fig_episode_ablation_eo_nodataset.png",
                show_compas_placeholder=False)

# 1-dataset (census only, all 4 episode configs)
_make_ep_eo_fig(DATASETS_1, 1, (5, 5),
                "fig_episode_ablation_census.png",
                show_compas_placeholder=False)


# ─────────────────────────────────────────────────────────────────────────────
# Delta scale ablation
# ─────────────────────────────────────────────────────────────────────────────
print("Generating delta ablation figures ...")

DELTA_CONFIGS = {
    "census": {
        "0.05": "census_delta005",
        "0.10": "census_ep1500ph400_5s_EP1500",
        "0.20": "census_delta020",
        "0.50": "census_delta050",
    },
    "capture24": {
        "0.05": "capture24_delta005",
        "0.10": "capture24_ep800ph200_5s_EP800",
        "0.20": "capture24_delta020",
        "0.50": "capture24_delta050",
    },
}
DELTA_LABELS = ["0.05", "0.10", "0.20", "0.50"]
DELTA_CHOSEN = 1


def _make_delta_fig(datasets, ncols, figsize, fname, transform_fn=None):
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    for ci, ds in enumerate(datasets):
        seeds = EXPECTED_SEEDS[ds]
        col   = DS_COLORS[ds]
        ax    = axes[ci]

        means, stds = [], []
        for lbl in DELTA_LABELS:
            if ds == "compas":
                m, s = COMPAS_DELTA_PH[lbl]
            else:
                key = DELTA_CONFIGS[ds][lbl]
                df  = find_dir(key, ds, seeds)
                if df is not None:
                    m, s = get_eo(df)
                else:
                    m, s = float("nan"), 0.0
                    print(f"  MISSING: delta {lbl} {ds}")
            means.append(m); stds.append(s)

        means, stds = (transform_fn or transform_eo)(means, stds)

        base_rgb = matplotlib.colors.to_rgb(col)
        colors   = [base_rgb + (CHOSEN_ALPHA if i == DELTA_CHOSEN else OTHER_ALPHA,)
                    for i in range(len(DELTA_LABELS))]
        panel_title = "Delta Scale Ablation" if ncols == 1 else DS_LABELS[ds]
        bars = make_bar_ax(ax, DELTA_LABELS, means, stds, colors, title=panel_title)
        bars[DELTA_CHOSEN].set_edgecolor("#111111")
        bars[DELTA_CHOSEN].set_linewidth(1.8)
        label_bars(ax, bars, means)
        ax.set_xlabel("Delta scale", fontsize=11)

    if "compas" in datasets:
        pending_note(fig)
    fig.tight_layout()
    save_fig(fname)
    plt.close()


_make_delta_fig(DATASETS_3, 3, (13, 5),   "fig_delta_ablation_v3_large.png")
_make_delta_fig(DATASETS_2, 2, (9,  5),   "fig_delta_ablation_nodataset.png")
_make_delta_fig(DATASETS_1, 1, (5,  5),   "fig_delta_ablation_census.png")


# ─────────────────────────────────────────────────────────────────────────────
# Global-only vs. local reward augmentation
# ─────────────────────────────────────────────────────────────────────────────
print("Generating dvrl ablation figures ...")

DVRL_CONFIGS = {
    "census":    ("census_ep1500ph400_5s_EP1500", "census_dvrl"),
    "capture24": ("capture24_ep800ph200_5s_EP800", "capture24_dvrl"),
}


def _make_dvrl_fig(datasets, ncols, figsize, fname, transform_fn=None):
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    for ci, ds in enumerate(datasets):
        seeds = EXPECTED_SEEDS[ds]
        ax    = axes[ci]
        col   = DS_COLORS[ds]

        if ds == "compas":
            g_eo = COMPAS_DVRL_PH["global"]
            d_eo = COMPAS_DVRL_PH["dvrl"]
            a_eo = COMPAS_ALPHA_EO
        else:
            g_key, d_key = DVRL_CONFIGS[ds]
            g_df = find_dir(g_key, ds, seeds)
            d_df = find_dir(d_key, ds, seeds)
            g_eo = get_eo(g_df) if g_df is not None else (float("nan"), 0.0)
            d_eo = get_eo(d_df) if d_df is not None else (float("nan"), 0.0)
            src_df = g_df if g_df is not None else d_df
            a_eo   = get_alpha_eo(src_df) if src_df is not None else (float("nan"), 0.0)

        labels  = ["Global-only", "Local aug."]
        means   = [g_eo[0], d_eo[0]]
        stds    = [g_eo[1], d_eo[1]]

        means, stds = (transform_fn or transform_eo)(means, stds)

        base_rgb = matplotlib.colors.to_rgb(col)
        colors   = [base_rgb + (CHOSEN_ALPHA,), base_rgb + (OTHER_ALPHA,)]
        panel_title = "Local Reward Augmentation" if ncols == 1 else DS_LABELS[ds]
        bars = make_bar_ax(ax, labels, means, stds, colors, title=panel_title)
        bars[0].set_edgecolor("#111111")
        bars[0].set_linewidth(1.8)
        label_bars(ax, bars, means)

    if "compas" in datasets:
        pending_note(fig)
    fig.tight_layout()
    save_fig(fname)
    plt.close()


_make_dvrl_fig(DATASETS_3, 3, (13, 5), "fig_dvrl_ablation_v3_large.png")
_make_dvrl_fig(DATASETS_2, 2, (9,  5), "fig_dvrl_ablation_nodataset.png")
_make_dvrl_fig(DATASETS_1, 1, (5,  5), "fig_dvrl_ablation_census.png")


# ─────────────────────────────────────────────────────────────────────────────
# PCA components ablation
# ─────────────────────────────────────────────────────────────────────────────
print("Generating PCA ablation figures ...")

PCA_CONFIGS = {
    "census": [
        ("census_raw",                   "No PCA"),
        ("census_ep1500ph400_5s_EP1500", "PCA-10"),
        ("census_pca15",                 "PCA-15"),
        ("census_pca20",                 "PCA-20"),
    ],
    "capture24": [
        ("capture24_raw",                "No PCA"),
        ("capture24_ep800ph200_5s_EP800","PCA-10"),
        ("capture24_pca15",              "PCA-15"),
        ("capture24_pca20",              "PCA-20"),
    ],
}
COMPAS_PCA_ORDER = ["No PCA", "PCA-8", "PCA-9", "PCA-10"]
PCA_CHOSEN = {"census": 1, "compas": 3, "capture24": 1}


def _make_pca_fig(datasets, ncols, figsize, fname, transform_fn=None):
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    for ci, ds in enumerate(datasets):
        seeds  = EXPECTED_SEEDS[ds]
        ax     = axes[ci]
        chosen = PCA_CHOSEN[ds]
        col    = DS_COLORS[ds]

        means, stds, labels = [], [], []

        if ds == "compas":
            for lbl in COMPAS_PCA_ORDER:
                m, s = COMPAS_PCA_PH[lbl]
                means.append(m); stds.append(s); labels.append(lbl)
        else:
            for key, lbl in PCA_CONFIGS[ds]:
                df = find_dir(key, ds, seeds)
                if df is not None:
                    m, s = get_eo(df)
                else:
                    m, s = float("nan"), 0.0
                    print(f"  MISSING: pca {key} {ds}")
                means.append(m); stds.append(s); labels.append(lbl)

        means, stds = (transform_fn or transform_eo)(means, stds)

        base_rgb = matplotlib.colors.to_rgb(col)
        colors   = [base_rgb + (CHOSEN_ALPHA if i == chosen else OTHER_ALPHA,)
                    for i in range(len(labels))]
        panel_title = "PCA Dimensionality Ablation" if ncols == 1 else DS_LABELS[ds]
        bars = make_bar_ax(ax, labels, means, stds, colors, title=panel_title)
        bars[chosen].set_edgecolor("#111111")
        bars[chosen].set_linewidth(1.8)
        label_bars(ax, bars, means)

    if "compas" in datasets:
        pending_note(fig)
    fig.tight_layout()
    save_fig(fname)
    plt.close()


_make_pca_fig(DATASETS_3, 3, (13, 5), "fig_pca_ablation_v3_large.png")
_make_pca_fig(DATASETS_2, 2, (9,  5), "fig_pca_ablation_nodataset.png")
_make_pca_fig(DATASETS_1, 1, (5,  5), "fig_pca_ablation_census.png")


# ─────────────────────────────────────────────────────────────────────────────
# FFNN beta epochs ablation
# ─────────────────────────────────────────────────────────────────────────────
print("Generating FFNN ablation figures ...")

FFNN_CONFIGS = {
    "census": [
        ("census_ffnn10",                "10"),
        ("census_ep1500ph400_5s_EP1500", "20 \u2605"),
        ("census_ffnn50",                "50"),
    ],
    "capture24": [
        ("capture24_ffnn10",              "10"),
        ("capture24_ep800ph200_5s_EP800", "20 \u2605"),
        ("capture24_ffnn50",              "50"),
    ],
}
COMPAS_FFNN_ORDER = ["10", "20 \u2605", "50"]
FFNN_CHOSEN = 1


def _make_ffnn_fig(datasets, ncols, figsize, fname, transform_fn=None):
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    for ci, ds in enumerate(datasets):
        seeds = EXPECTED_SEEDS[ds]
        ax    = axes[ci]
        col   = DS_COLORS[ds]

        means, stds, labels = [], [], []

        if ds == "compas":
            for lbl in COMPAS_FFNN_ORDER:
                m, s = COMPAS_FFNN_PH[lbl]
                means.append(m); stds.append(s); labels.append(lbl)
        else:
            for key, lbl in FFNN_CONFIGS[ds]:
                df = find_dir(key, ds, seeds)
                if df is not None:
                    m, s = get_eo(df)
                else:
                    m, s = float("nan"), 0.0
                    print(f"  MISSING: ffnn {key} {ds}")
                means.append(m); stds.append(s); labels.append(lbl)

        means, stds = (transform_fn or transform_eo)(means, stds)

        base_rgb = matplotlib.colors.to_rgb(col)
        colors   = [base_rgb + (CHOSEN_ALPHA if i == FFNN_CHOSEN else OTHER_ALPHA,)
                    for i in range(len(labels))]
        panel_title = "Classifier Epochs Ablation" if ncols == 1 else DS_LABELS[ds]
        bars = make_bar_ax(ax, labels, means, stds, colors, title=panel_title)
        bars[FFNN_CHOSEN].set_edgecolor("#111111")
        bars[FFNN_CHOSEN].set_linewidth(1.8)
        label_bars(ax, bars, means)
        ax.set_xlabel("Classifier Epochs", fontsize=11)

    if "compas" in datasets:
        pending_note(fig)
    fig.tight_layout()
    save_fig(fname)
    plt.close()


_make_ffnn_fig(DATASETS_3, 3, (13, 5), "fig_ffnn_ablation_v3_large.png")
_make_ffnn_fig(DATASETS_2, 2, (9,  5), "fig_ffnn_ablation_nodataset.png")
_make_ffnn_fig(DATASETS_1, 1, (5,  5), "fig_ffnn_ablation_census.png")


# ─────────────────────────────────────────────────────────────────────────────
# EO Inverted variants (census only)
# EO_inv = 1 - (EO - min) / (max - min)  →  higher is better
# ─────────────────────────────────────────────────────────────────────────────
print("Generating EO inverted figures (census only) ...")

_make_ep_eo_fig(DATASETS_1, 1, (5, 5),
                "fig_episode_ablation_eo_inverted_census.png",
                show_compas_placeholder=False,
                transform_fn=transform_eo_inverted)

_make_delta_fig(DATASETS_1, 1, (5, 5),
                "fig_delta_ablation_eo_inverted_census.png",
                transform_fn=transform_eo_inverted)

_make_dvrl_fig(DATASETS_1, 1, (5, 5),
               "fig_dvrl_ablation_eo_inverted_census.png",
               transform_fn=transform_eo_inverted)

_make_pca_fig(DATASETS_1, 1, (5, 5),
              "fig_pca_ablation_eo_inverted_census.png",
              transform_fn=transform_eo_inverted)

_make_ffnn_fig(DATASETS_1, 1, (5, 5),
               "fig_ffnn_ablation_eo_inverted_census.png",
               transform_fn=transform_eo_inverted)


print("\nAll enhanced ablation figures done.")
