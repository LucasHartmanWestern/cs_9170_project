"""
Generate ablation figures for paper_figures_v3.

Five figures:
  fig_episode_ablation.png   — ep800ph0, ep800ph200, ep1500ph400, ep2000ph600 × 3 datasets
  fig_delta_ablation.png     — delta 0.05, 0.10, 0.20, 0.50 × 3 datasets
  fig_dvrl_ablation.png      — DVRL vs global-only × 3 datasets (with α-EO annotation)
  fig_pca_ablation.png       — PCA components × 3 datasets (raw/8/9/10/15/20)
  fig_ffnn_ablation.png      — FFNN beta epochs 10, 20, 50 × 3 datasets
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_V3 = "/home/epigou/cs_9170_project/paper_results_v3/training_runs"
OUT_DIR = "/home/epigou/cs_9170_project/paper_figures_v3"

EXPECTED_SEEDS = {
    "census":    {"0", "2", "3", "5", "42"},
    "compas":    {"1", "3", "6", "7", "42"},
    "capture24": {"0", "3", "4", "5", "42"},
}

DS_LABELS = {"census": "Census", "compas": "COMPAS", "capture24": "Capture-24"}
DATASETS   = ["census", "compas", "capture24"]

# ── helpers ──────────────────────────────────────────────────────────────────

def load_ftm(path, expected_seeds=None):
    p = os.path.join(path, "final_test_metrics.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if expected_seeds:
        df = df[df["seed"].astype(str).isin(expected_seeds)]
    return df if len(df) > 0 else None


def find_dir(key, ds, expected_seeds=None, prefer_n=5):
    """Find best matching directory in BASE_V3 containing `key`."""
    matches = [d for d in os.listdir(BASE_V3) if key in d]
    # prefer dirs with more seeds
    scored = []
    for d in matches:
        try:
            seeds = len([s for s in os.listdir(os.path.join(BASE_V3, d)) if s.startswith("seed_")])
        except:
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


def get_eo(df):
    return stats(df, "beta_eod_max_diff")

def get_auc(df):
    return stats(df, "beta_roc_auc")

def get_alpha_eo(df):
    return stats(df, "alpha_eod_max_diff")


# ── colour helpers ────────────────────────────────────────────────────────────
DS_COLORS = {
    "census":    "#2b4590",
    "compas":    "#c0392b",
    "capture24": "#27ae60",
}

CHOSEN_ALPHA = 0.95   # full opacity for chosen / primary
OTHER_ALPHA  = 0.55   # dimmed for non-chosen variants


def make_bar_ax(ax, labels, means, stds, colors, hatches=None,
                ylabel="EO gap ↓", title="", star_idx=None):
    """Draw a simple grouped bar chart on ax."""
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=3,
                  color=colors, edgecolor="white", linewidth=0.6,
                  error_kw=dict(elinewidth=0.8, ecolor="#555555"))
    if hatches:
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
    if star_idx is not None:
        ax.get_children()[star_idx].set_edgecolor("#333333")
        ax.get_children()[star_idx].set_linewidth(1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    return bars


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Episode ablation
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fig_episode_ablation.png ...")

EP_KEYS = {
    "census":    [
        ("census_ep800ph0",    "ep800\nph0"),
        ("census_ep800ph200",  "ep800\nph200"),
        ("census_ep1500ph400", "ep1500\nph400"),  # ← chosen
        ("census_ep2000ph600", "ep2000\nph600"),
    ],
    "compas":    [
        ("compas_ep800ph0",    "ep800\nph0"),
        ("compas_ep800ph200",  "ep800\nph200"),
        ("compas_ep1500ph400", "ep1500\nph400"),
        ("compas_ep2000ph600", "ep2000\nph600"),  # ← chosen (via dvrl run)
    ],
    "capture24": [
        ("capture24_ep800ph0",    "ep800\nph0"),
        ("capture24_ep800ph200",  "ep800\nph200"),  # ← chosen
        ("capture24_ep1500ph400", "ep1500\nph400"),
        ("capture24_ep2000ph600", "ep2000\nph600"),
    ],
}
# Chosen config index per dataset (0-based)
EP_CHOSEN = {"census": 2, "compas": 3, "capture24": 1}

fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharey=False)
fig.suptitle("Episode budget ablation", fontsize=11, fontweight="bold", y=1.01)

for ci, ds in enumerate(DATASETS):
    seeds = EXPECTED_SEEDS[ds]
    col   = DS_COLORS[ds]
    chosen = EP_CHOSEN[ds]

    # Top row: EO, Bottom row: AUC
    for ri, (metric_fn, metric_label, better) in enumerate([
        (get_eo,  "EO gap ↓", "min"),
        (get_auc, "AUC ↑",    "max"),
    ]):
        ax = axes[ri][ci]
        means, stds, labels = [], [], []
        for key, lbl in EP_KEYS[ds]:
            df = find_dir(key, ds, seeds)
            if df is not None:
                m, s = metric_fn(df)
            else:
                m, s = float("nan"), 0.0
                print(f"  MISSING: {key}")
            means.append(m)
            stds.append(s)
            labels.append(lbl)

        colors = [col if i == chosen else col + "88" for i in range(len(labels))]
        # fix hex alpha notation — use tuple approach
        base_rgb = matplotlib.colors.to_rgb(col)
        colors = [base_rgb + (CHOSEN_ALPHA if i == chosen else OTHER_ALPHA,)
                  for i in range(len(labels))]

        bars = make_bar_ax(ax, labels, means, stds, colors,
                           ylabel=metric_label,
                           title=f"{DS_LABELS[ds]}" if ri == 0 else "")
        # highlight chosen bar with bold edge
        bars[chosen].set_edgecolor("#111111")
        bars[chosen].set_linewidth(1.6)

        # Add value labels on bars
        for bar, m in zip(bars, means):
            if not np.isnan(m):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{m:.3f}", ha="center", va="bottom", fontsize=6.5)

fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_episode_ablation.png"),
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: fig_episode_ablation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Delta scale ablation
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fig_delta_ablation.png ...")

# delta=0.10 is always the main run (chosen config)
DELTA_CONFIGS = {
    "census": {
        "0.05": "census_delta005",
        "0.10": "census_ep1500ph400_5s_EP1500",   # main run
        "0.20": "census_delta020",
        "0.50": "census_delta050",
    },
    "compas": {
        "0.05": "compas_delta005",
        "0.10": "compas_dvrl_5s_EP2000",           # main run (DVRL, but same delta)
        "0.20": "compas_delta020",
        "0.50": "compas_delta050",
    },
    "capture24": {
        "0.05": "capture24_delta005",
        "0.10": "capture24_ep800ph200_5s_EP800",   # main run
        "0.20": "capture24_delta020",
        "0.50": "capture24_delta050",
    },
}
DELTA_LABELS  = ["0.05", "0.10", "0.20", "0.50"]
DELTA_CHOSEN  = 1  # index of 0.10

fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
fig.suptitle("Delta scale ablation (EO gap)", fontsize=11, fontweight="bold")

for ci, ds in enumerate(DATASETS):
    seeds = EXPECTED_SEEDS[ds]
    col   = DS_COLORS[ds]
    ax    = axes[ci]

    means, stds = [], []
    for lbl in DELTA_LABELS:
        key = DELTA_CONFIGS[ds][lbl]
        df  = find_dir(key, ds, seeds)
        if df is not None:
            m, s = get_eo(df)
        else:
            m, s = float("nan"), 0.0
            print(f"  MISSING: delta {lbl} {ds}")
        means.append(m)
        stds.append(s)

    base_rgb = matplotlib.colors.to_rgb(col)
    colors   = [base_rgb + (CHOSEN_ALPHA if i == DELTA_CHOSEN else OTHER_ALPHA,)
                for i in range(len(DELTA_LABELS))]
    bars = make_bar_ax(ax, DELTA_LABELS, means, stds, colors,
                       ylabel="EO gap ↓", title=DS_LABELS[ds])
    bars[DELTA_CHOSEN].set_edgecolor("#111111")
    bars[DELTA_CHOSEN].set_linewidth(1.6)

    for bar, m in zip(bars, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xlabel("δ scale", fontsize=8)

fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_delta_ablation.png"),
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: fig_delta_ablation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: DVRL vs global-only
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fig_dvrl_ablation.png ...")

# For each dataset: which is global-only (ep-ablation), which is dvrl
DVRL_CONFIGS = {
    #  ds       : (global_key,                              dvrl_key,                    chosen)
    "census":    ("census_ep1500ph400_5s_EP1500",           "census_dvrl",               "global"),
    "compas":    ("compas_ep2000ph600",                     "compas_dvrl_5s_EP2000",     "dvrl"),
    "capture24": ("capture24_ep800ph200_5s_EP800",          "capture24_dvrl",            "global"),
}

fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
fig.suptitle("Global-only vs. DVRL local reward (EO gap)", fontsize=11, fontweight="bold")

for ci, ds in enumerate(DATASETS):
    seeds = EXPECTED_SEEDS[ds]
    ax    = axes[ci]
    g_key, d_key, chosen = DVRL_CONFIGS[ds]

    g_df = find_dir(g_key, ds, seeds)
    d_df = find_dir(d_key, ds, seeds)

    g_eo = get_eo(g_df)   if g_df is not None else (float("nan"), 0.0)
    d_eo = get_eo(d_df)   if d_df is not None else (float("nan"), 0.0)
    a_eo = get_alpha_eo(g_df if g_df is not None else d_df)  # alpha EO from either

    labels  = ["Global-only", "DVRL"]
    means   = [g_eo[0], d_eo[0]]
    stds    = [g_eo[1], d_eo[1]]

    col = DS_COLORS[ds]
    base_rgb = matplotlib.colors.to_rgb(col)
    chosen_i = 0 if chosen == "global" else 1
    colors   = [base_rgb + (CHOSEN_ALPHA if i == chosen_i else OTHER_ALPHA,)
                for i in range(2)]
    bars = make_bar_ax(ax, labels, means, stds, colors,
                       ylabel="EO gap ↓", title=DS_LABELS[ds])
    bars[chosen_i].set_edgecolor("#111111")
    bars[chosen_i].set_linewidth(1.6)

    for bar, m in zip(bars, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=7)

    # Annotate alpha EO
    ax.axhline(a_eo[0], color="#777777", linestyle="--", linewidth=0.9, label=f"α-EO={a_eo[0]:.3f}")
    ax.legend(fontsize=6.5, loc="upper right")

fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_dvrl_ablation.png"),
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: fig_dvrl_ablation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: PCA components ablation
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fig_pca_ablation.png ...")

PCA_CONFIGS = {
    "census": [
        ("census_raw",          "No PCA"),
        ("census_ep1500ph400_5s_EP1500", "PCA-10 ★"),  # main
        ("census_pca15",        "PCA-15"),
        ("census_pca20",        "PCA-20"),
    ],
    "compas": [
        ("compas_raw",          "No PCA"),
        ("compas_pca8",         "PCA-8"),
        ("compas_pca9",         "PCA-9"),
        ("compas_dvrl_5s_EP2000", "PCA-10 ★"),         # main
    ],
    "capture24": [
        ("capture24_raw",       "No PCA"),
        ("capture24_ep800ph200_5s_EP800", "PCA-10 ★"),  # main
        ("capture24_pca15",     "PCA-15"),
        ("capture24_pca20",     "PCA-20"),
    ],
}
PCA_CHOSEN = {"census": 1, "compas": 3, "capture24": 1}

fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
fig.suptitle("PCA dimensionality ablation (EO gap)", fontsize=11, fontweight="bold")

for ci, ds in enumerate(DATASETS):
    seeds  = EXPECTED_SEEDS[ds]
    ax     = axes[ci]
    chosen = PCA_CHOSEN[ds]

    means, stds, labels = [], [], []
    for key, lbl in PCA_CONFIGS[ds]:
        df = find_dir(key, ds, seeds)
        if df is not None:
            m, s = get_eo(df)
        else:
            m, s = float("nan"), 0.0
            print(f"  MISSING: pca {key} {ds}")
        means.append(m)
        stds.append(s)
        labels.append(lbl)

    col = DS_COLORS[ds]
    base_rgb = matplotlib.colors.to_rgb(col)
    colors   = [base_rgb + (CHOSEN_ALPHA if i == chosen else OTHER_ALPHA,)
                for i in range(len(labels))]
    bars = make_bar_ax(ax, labels, means, stds, colors,
                       ylabel="EO gap ↓", title=DS_LABELS[ds])
    bars[chosen].set_edgecolor("#111111")
    bars[chosen].set_linewidth(1.6)

    for bar, m in zip(bars, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=6.5)

fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_pca_ablation.png"),
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: fig_pca_ablation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: FFNN beta epochs ablation
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fig_ffnn_ablation.png ...")

FFNN_CONFIGS = {
    "census": [
        ("census_ffnn10",                "10"),
        ("census_ep1500ph400_5s_EP1500", "20 ★"),  # main
        ("census_ffnn50",                "50"),
    ],
    "compas": [
        ("compas_ffnn10",               "10"),
        ("compas_dvrl_5s_EP2000",       "20 ★"),  # main
        ("compas_ffnn50",               "50"),
    ],
    "capture24": [
        ("capture24_ffnn10",                "10"),
        ("capture24_ep800ph200_5s_EP800",   "20 ★"),  # main
        ("capture24_ffnn50",                "50"),
    ],
}
FFNN_CHOSEN = 1  # epochs=20 always at index 1

fig, axes = plt.subplots(1, 3, figsize=(9, 3.8))
fig.suptitle("Beta classifier training epochs ablation (EO gap)", fontsize=11, fontweight="bold")

for ci, ds in enumerate(DATASETS):
    seeds = EXPECTED_SEEDS[ds]
    ax    = axes[ci]

    means, stds, labels = [], [], []
    for key, lbl in FFNN_CONFIGS[ds]:
        df = find_dir(key, ds, seeds)
        if df is not None:
            m, s = get_eo(df)
        else:
            m, s = float("nan"), 0.0
            print(f"  MISSING: ffnn {key} {ds}")
        means.append(m)
        stds.append(s)
        labels.append(lbl)

    col = DS_COLORS[ds]
    base_rgb = matplotlib.colors.to_rgb(col)
    colors   = [base_rgb + (CHOSEN_ALPHA if i == FFNN_CHOSEN else OTHER_ALPHA,)
                for i in range(len(labels))]
    bars = make_bar_ax(ax, labels, means, stds, colors,
                       ylabel="EO gap ↓", title=DS_LABELS[ds])
    bars[FFNN_CHOSEN].set_edgecolor("#111111")
    bars[FFNN_CHOSEN].set_linewidth(1.6)
    ax.set_xlabel("β epochs", fontsize=8)

    for bar, m in zip(bars, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=7)

fig.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_ffnn_ablation.png"),
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: fig_ffnn_ablation.png")

print("\nAll ablation figures done.")
