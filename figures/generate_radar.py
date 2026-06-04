"""
Generate radar_top_down_AB.png — 6-axis multi-metric trade-off profiles.

Methods shown: Alpha, FORGE, GroupDRO, FairTabDDPM.
Axes (3 utility, 3 fairness):
  AUC, F1 Weighted, Accuracy  (higher = better, direct)
  EO' Score, EOd' Score, DP' Score  (lower = better, inverted: 1 - normalised)

Normalisation: value / max across all 4 methods per dataset (no min subtraction),
so values fill [0, 1] with 1 = best observed for that metric on that dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

CSV_PATH = "/storage_1/epigou_storage/FORGE/experiment3/experiment3_radar_metrics.csv"
OUT_PATH = "/home/epigou/cs_9170_project/paper/figures/radar_top_down_AB.png"

# (csv_column, axis_label, invert)
METRICS = [
    ("beta_roc_auc",       "AUC",          False),
    ("beta_f1_weighted",   "F1 Weighted",  False),
    ("beta_acc",           "Accuracy",     False),
    ("beta_eo_tpr_diff",   "EO' Score",    True),
    ("beta_eod_avg_diff",  "EOd' Score",   True),
    ("beta_dp_diff",       "DP' Score",    True),
]

DATASETS     = ["Census", "Capture-24"]
PANEL_LABELS = ["a)", "b)"]

# method name in CSV → display style
METHOD_STYLES = {
    "Alpha":       {"color": "tab:red",    "ls": "-",  "lw": 1.5, "label": "Alpha"},
    "FORGE":       {"color": "tab:blue",   "ls": "-",  "lw": 2.0, "label": "FORGE"},
    "GroupDRO":    {"color": "tab:orange", "ls": "--", "lw": 1.5, "label": "GDRO"},
    "FairTabDDPM": {"color": "tab:green",  "ls": "--", "lw": 1.5, "label": "FairTabDDPM"},
}


def build_normed(df, dataset):
    cols    = [m[0] for m in METRICS]
    labels  = [m[1] for m in METRICS]
    methods = list(METHOD_STYLES.keys())
    sub     = df[(df["dataset"] == dataset) & (df["method"].isin(methods))].groupby("method")[cols].mean()

    normed = pd.DataFrame(index=sub.index)
    for col, label, invert in METRICS:
        mx = sub[col].max()
        n  = sub[col] / mx if mx > 0 else sub[col] * 0.0
        normed[label] = (1.0 - n) if invert else n
    return normed


# ── plot ──────────────────────────────────────────────────────────────────────

df     = pd.read_csv(CSV_PATH)
n      = len(METRICS)
labels = [m[1] for m in METRICS]
angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
angles += angles[:1]   # close polygon

fig, axes = plt.subplots(
    nrows=1, ncols=2,
    figsize=(10.0, 5.2),
    subplot_kw=dict(polar=True),
)

for ax, dataset, panel_label in zip(axes, DATASETS, PANEL_LABELS):
    normed = build_normed(df, dataset)

    for method, style in METHOD_STYLES.items():
        if method not in normed.index:
            continue
        row  = normed.loc[method]
        vals = row[labels].tolist() + [row[labels[0]]]
        ax.plot(angles, vals,
                color=style["color"], ls=style["ls"], lw=style["lw"])
        ax.fill(angles, vals, color=style["color"], alpha=0.07)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], size=8, color="gray")

    ax.text(-0.12, 1.14, panel_label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left")

# ── legend ────────────────────────────────────────────────────────────────────

handles = []
for method, style in METHOD_STYLES.items():
    h = mlines.Line2D([], [],
                      color=style["color"], ls=style["ls"], lw=style["lw"],
                      label=style["label"])
    handles.append(h)

fig.legend(
    handles=handles,
    loc="lower center",
    ncol=4,
    fontsize=11,
    frameon=True,
    bbox_to_anchor=(0.5, 0.01),
)

plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
