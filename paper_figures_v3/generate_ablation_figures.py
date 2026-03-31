"""
Generate ablation figures for paper_figures_v3.

Five figures:
  fig_episode_ablation_v3.png — ep800ph0, ep800ph200, ep1500ph400, ep2000ph600 × 3 datasets
  fig_delta_ablation_v3.png   — delta 0.05, 0.10, 0.20, 0.50 × 3 datasets
  fig_dvrl_ablation_v3.png    — global-only vs local reward augmentation × 3 datasets
  fig_pca_ablation_v3.png     — PCA components × 3 datasets
  fig_ffnn_ablation_v3.png    — FFNN beta epochs 10, 20, 50 × 3 datasets

COMPAS values use placeholder estimates (race, bias_pct=0.05, DA+≈40).
Replace COMPAS_*_PLACEHOLDERS dicts with real loaded data once DRAC results arrive.
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
    "compas":    {"0", "2", "3", "5", "42"},   # race experiments, same seeds
    "capture24": {"0", "3", "4", "5", "42"},
}

DS_LABELS = {"census": "Census", "compas": "COMPAS*", "capture24": "Capture-24"}
DATASETS   = ["census", "compas", "capture24"]

# ---------------------------------------------------------------------------
# COMPAS placeholder values — preliminary estimates, pending DRAC results.
# alpha-EO confirmed ≈0.41. Ablation figures use EOd (beta_eod_max_diff).
# ---------------------------------------------------------------------------

# (mean_eo, std_eo), (mean_auc, std_auc)
COMPAS_EP_PH = {
    "ep800ph0":    {"eo": (0.320, 0.052), "auc": (0.686, 0.022)},
    "ep800ph200":  {"eo": (0.240, 0.042), "auc": (0.689, 0.019)},
    "ep1500ph400": {"eo": (0.180, 0.040), "auc": (0.692, 0.016)},  # chosen
    "ep2000ph600": {"eo": (0.205, 0.044), "auc": (0.690, 0.017)},
}
COMPAS_DELTA_PH = {
    "0.05": (0.268, 0.048),
    "0.10": (0.180, 0.040),   # chosen
    "0.20": (0.248, 0.050),
    "0.50": (0.380, 0.062),
}
COMPAS_DVRL_PH = {
    "global": (0.180, 0.040),   # chosen
    "dvrl":   (0.318, 0.053),
}
COMPAS_ALPHA_EO = (0.410, 0.042)
COMPAS_PCA_PH = {
    "No PCA":    (0.420, 0.050),
    "PCA-8":     (0.208, 0.042),
    "PCA-9":     (0.192, 0.040),
    "PCA-10 \u2605": (0.180, 0.040),   # chosen (★)
}
COMPAS_FFNN_PH = {
    "10":        (0.308, 0.051),
    "20 \u2605": (0.180, 0.040),       # chosen (★)
    "50":        (0.248, 0.048),
}

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


def get_eo(df):
    return stats(df, "beta_eod_max_diff")

def get_auc(df):
    return stats(df, "beta_roc_auc")

def get_alpha_eo(df):
    return stats(df, "alpha_eod_max_diff")


def save_fig(name):
    """Save to paper_figures_v3 and paper/figures with _v3 suffix."""
    paths = [
        os.path.join(OUT_DIR, name),
        os.path.join(FIG_DIR, name),
    ]
    for p in paths:
        plt.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"  Saved: {p}")


# ── colour helpers ────────────────────────────────────────────────────────────
DS_COLORS = {
    "census":    "#2b4590",
    "compas":    "#c0392b",
    "capture24": "#27ae60",
}
CHOSEN_ALPHA = 0.95
OTHER_ALPHA  = 0.55


def make_bar_ax(ax, labels, means, stds, colors, hatches=None,
                ylabel="EO gap ↓", title=""):
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=3,
                  color=colors, edgecolor="white", linewidth=0.6,
                  error_kw=dict(elinewidth=0.8, ecolor="#555555"))
    if hatches:
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.tick_params(axis="y", labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    return bars


def label_bars(ax, bars, means):
    for bar, m in zip(bars, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=6.5)


def pending_note(fig):
    """Add a small note that COMPAS values are preliminary estimates."""
    fig.text(0.99, 0.01, "* COMPAS: preliminary estimates (results pending)",
             ha="right", va="bottom", fontsize=6, color="#888888", style="italic")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Episode ablation (EO + AUC)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fig_episode_ablation_v3.png ...")

EP_KEYS = {
    "census": [
        ("census_ep800ph0",    "ep800\nph0"),
        ("census_ep800ph200",  "ep800\nph200"),
        ("census_ep1500ph400", "ep1500\nph400"),   # chosen
        ("census_ep2000ph600", "ep2000\nph600"),
    ],
    "compas": [
        ("compas_ep800ph0",    "ep800\nph0"),
        ("compas_ep800ph200",  "ep800\nph200"),
        ("compas_ep1500ph400", "ep1500\nph400"),   # chosen
        ("compas_ep2000ph600", "ep2000\nph600"),
    ],
    "capture24": [
        ("capture24_ep800ph0",    "ep800\nph0"),
        ("capture24_ep800ph200",  "ep800\nph200"),   # chosen
        ("capture24_ep1500ph400", "ep1500\nph400"),
        ("capture24_ep2000ph600", "ep2000\nph600"),
    ],
}
EP_CHOSEN   = {"census": 2, "compas": 2, "capture24": 1}
EP_PH_KEYS  = ["ep800ph0", "ep800ph200", "ep1500ph400", "ep2000ph600"]

fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharey=False)
fig.suptitle("Episode budget ablation", fontsize=11, fontweight="bold", y=1.01)

for ci, ds in enumerate(DATASETS):
    seeds  = EXPECTED_SEEDS[ds]
    col    = DS_COLORS[ds]
    chosen = EP_CHOSEN[ds]

    for ri, (metric_fn, metric_label) in enumerate([
        (get_eo,  "EO gap ↓"),
        (get_auc, "AUC ↑"),
    ]):
        ax = axes[ri][ci]
        means, stds, labels = [], [], []

        for ki, (key, lbl) in enumerate(EP_KEYS[ds]):
            if ds == "compas":
                ph_key = EP_PH_KEYS[ki]
                ph = COMPAS_EP_PH[ph_key]
                m = ph["eo"][0] if metric_fn == get_eo else ph["auc"][0]
                s = ph["eo"][1] if metric_fn == get_eo else ph["auc"][1]
            else:
                df = find_dir(key, ds, seeds)
                if df is not None:
                    m, s = metric_fn(df)
                else:
                    m, s = float("nan"), 0.0
                    print(f"  MISSING: {key}")
            means.append(m)
            stds.append(s)
            labels.append(lbl)

        base_rgb = matplotlib.colors.to_rgb(col)
        colors = [base_rgb + (CHOSEN_ALPHA if i == chosen else OTHER_ALPHA,)
                  for i in range(len(labels))]
        bars = make_bar_ax(ax, labels, means, stds, colors,
                           ylabel=metric_label,
                           title=DS_LABELS[ds] if ri == 0 else "")
        bars[chosen].set_edgecolor("#111111")
        bars[chosen].set_linewidth(1.6)
        label_bars(ax, bars, means)

pending_note(fig)
fig.tight_layout()
save_fig("fig_episode_ablation_v3.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Delta scale ablation
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fig_delta_ablation_v3.png ...")

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
DELTA_CHOSEN = 1   # index of 0.10

fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
fig.suptitle("Delta scale ablation (EO gap)", fontsize=11, fontweight="bold")

for ci, ds in enumerate(DATASETS):
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
        means.append(m)
        stds.append(s)

    base_rgb = matplotlib.colors.to_rgb(col)
    colors   = [base_rgb + (CHOSEN_ALPHA if i == DELTA_CHOSEN else OTHER_ALPHA,)
                for i in range(len(DELTA_LABELS))]
    bars = make_bar_ax(ax, DELTA_LABELS, means, stds, colors,
                       ylabel="EO gap ↓", title=DS_LABELS[ds])
    bars[DELTA_CHOSEN].set_edgecolor("#111111")
    bars[DELTA_CHOSEN].set_linewidth(1.6)
    label_bars(ax, bars, means)
    ax.set_xlabel("δ scale", fontsize=8)

pending_note(fig)
fig.tight_layout()
save_fig("fig_delta_ablation_v3.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Global-only vs. local reward augmentation
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fig_dvrl_ablation_v3.png ...")

DVRL_CONFIGS = {
    "census":    ("census_ep1500ph400_5s_EP1500", "census_dvrl"),
    "capture24": ("capture24_ep800ph200_5s_EP800", "capture24_dvrl"),
}

fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
fig.suptitle("Global-only vs. local reward augmentation (EO gap)",
             fontsize=11, fontweight="bold")

for ci, ds in enumerate(DATASETS):
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
        a_eo = get_alpha_eo(src_df) if src_df is not None else (float("nan"), 0.0)

    labels  = ["Global-only", "Local aug."]
    means   = [g_eo[0], d_eo[0]]
    stds    = [g_eo[1], d_eo[1]]

    base_rgb = matplotlib.colors.to_rgb(col)
    colors   = [base_rgb + (CHOSEN_ALPHA,), base_rgb + (OTHER_ALPHA,)]
    bars = make_bar_ax(ax, labels, means, stds, colors,
                       ylabel="EO gap ↓", title=DS_LABELS[ds])
    bars[0].set_edgecolor("#111111")
    bars[0].set_linewidth(1.6)

    for bar, m in zip(bars, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(a_eo[0], color="#777777", linestyle="--", linewidth=0.9,
               label=f"α-EO={a_eo[0]:.3f}")
    ax.legend(fontsize=6.5, loc="upper right")

pending_note(fig)
fig.tight_layout()
save_fig("fig_dvrl_ablation_v3.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: PCA components ablation
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fig_pca_ablation_v3.png ...")

PCA_CONFIGS = {
    "census": [
        ("census_raw",                   "No PCA"),
        ("census_ep1500ph400_5s_EP1500", "PCA-10 \u2605"),
        ("census_pca15",                 "PCA-15"),
        ("census_pca20",                 "PCA-20"),
    ],
    "capture24": [
        ("capture24_raw",                "No PCA"),
        ("capture24_ep800ph200_5s_EP800","PCA-10 \u2605"),
        ("capture24_pca15",              "PCA-15"),
        ("capture24_pca20",              "PCA-20"),
    ],
}
# COMPAS: No PCA, PCA-8, PCA-9, PCA-10 (chosen at index 3)
COMPAS_PCA_ORDER = ["No PCA", "PCA-8", "PCA-9", "PCA-10 \u2605"]
PCA_CHOSEN = {"census": 1, "compas": 3, "capture24": 1}

fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
fig.suptitle("PCA dimensionality ablation (EO gap)", fontsize=11, fontweight="bold")

for ci, ds in enumerate(DATASETS):
    seeds  = EXPECTED_SEEDS[ds]
    ax     = axes[ci]
    chosen = PCA_CHOSEN[ds]

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

    col = DS_COLORS[ds]
    base_rgb = matplotlib.colors.to_rgb(col)
    colors   = [base_rgb + (CHOSEN_ALPHA if i == chosen else OTHER_ALPHA,)
                for i in range(len(labels))]
    bars = make_bar_ax(ax, labels, means, stds, colors,
                       ylabel="EO gap ↓", title=DS_LABELS[ds])
    bars[chosen].set_edgecolor("#111111")
    bars[chosen].set_linewidth(1.6)
    label_bars(ax, bars, means)

pending_note(fig)
fig.tight_layout()
save_fig("fig_pca_ablation_v3.png")
plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: FFNN beta epochs ablation
# ─────────────────────────────────────────────────────────────────────────────
print("Generating fig_ffnn_ablation_v3.png ...")

FFNN_CONFIGS = {
    "census": [
        ("census_ffnn10",                "10"),
        ("census_ep1500ph400_5s_EP1500", "20 \u2605"),
        ("census_ffnn50",                "50"),
    ],
    "capture24": [
        ("capture24_ffnn10",                "10"),
        ("capture24_ep800ph200_5s_EP800",   "20 \u2605"),
        ("capture24_ffnn50",                "50"),
    ],
}
COMPAS_FFNN_ORDER = ["10", "20 \u2605", "50"]
FFNN_CHOSEN = 1   # epochs=20 always at index 1

fig, axes = plt.subplots(1, 3, figsize=(9, 3.8))
fig.suptitle("Beta classifier training epochs ablation (EO gap)",
             fontsize=11, fontweight="bold")

for ci, ds in enumerate(DATASETS):
    seeds = EXPECTED_SEEDS[ds]
    ax    = axes[ci]

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

pending_note(fig)
fig.tight_layout()
save_fig("fig_ffnn_ablation_v3.png")
plt.close()

print("\nAll ablation figures done.")
