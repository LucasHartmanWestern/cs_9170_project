"""
Regenerate radar_top_down_AB.png from main_table_metrics_per_seed.csv.

Changes vs original:
  - Brier Inv. Score replaced by EOd Inv. Score
  - Panel labels A/B → a/b
  - Legend added
  - Max normalization (value / max across methods, no min subtraction)
  - Inverted metrics: EO, DP, EOd  (1 - normalized so high = good on radar)
  - Values are means over seeds per method/dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CSV_PATH = "/home/epigou/cs_9170_project/figures/main_table_metrics_per_seed.csv"
OUT_PATH = "/home/epigou/cs_9170_project/paper/figures/radar_top_down_AB.png"

# (csv_column, axis_label, invert)
# 3 axes: all available for every method from EXP-050 data
METRICS = [
    ("f1_weighted",  "F1 Weighted",  False),
    ("roc_auc",      "ROC-AUC",      False),
    ("eo_tpr_diff",  "EO' Score",    True),
]

DATASETS      = ["Census", "Capture-24"]
PANEL_LABELS  = ["a)", "b)"]

METHOD_STYLES = {
    "Alpha":       {"color": "tab:red",    "ls": "-",  "label": "Alpha"},
    "RL (ours)":   {"color": "tab:blue",   "ls": "-",  "label": "FORGE"},
    "GroupDRO":    {"color": "tab:orange", "ls": "--", "label": "GDRO"},
    "FairTabDDPM": {"color": "tab:green",  "ls": "--", "label": "FairTabDDPM"},
}

# ── helpers ──────────────────────────────────────────────────────────────────

def radar_values(means_row, labels):
    """Return normalized radar values (closed polygon) for one method."""
    return means_row[labels].tolist()


def build_normed(df, dataset):
    cols    = [m[0] for m in METRICS]
    labels  = [m[1] for m in METRICS]
    methods = list(METHOD_STYLES.keys())
    sub     = df[(df["dataset"] == dataset) & (df["method"].isin(methods))].groupby("method")[cols].mean()

    normed = pd.DataFrame(index=sub.index)
    for col, label, invert in METRICS:
        mx = sub[col].max()
        n  = sub[col] / mx if mx > 0 else sub[col] * 0.0
        normed[label] = (1 - n) if invert else n
    return normed


# ── plot ─────────────────────────────────────────────────────────────────────

df       = pd.read_csv(CSV_PATH)
n        = len(METRICS)
labels   = [m[1] for m in METRICS]
angles   = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
angles  += angles[:1]   # close the polygon

fig, axes = plt.subplots(
    nrows=1, ncols=2,
    figsize=(9.0, 5.0),
    subplot_kw=dict(polar=True),
)

for ax, dataset, panel_label in zip(axes, DATASETS, PANEL_LABELS):
    normed = build_normed(df, dataset)

    for method, row in normed.iterrows():
        style  = METHOD_STYLES.get(method, {"color": "black", "ls": "-"})
        vals   = row[labels].tolist() + [row[labels[0]]]   # close
        ax.plot(angles, vals, color=style["color"], ls=style["ls"], linewidth=1.5)
        ax.fill(angles, vals, color=style["color"], alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11)
    # Shift "Accuracy" right and "EO' Score" left so they aren't obscured
    # Nudge ROC-AUC and F1 Weighted labels outward a little
    for tick, lbl in zip(ax.xaxis.get_major_ticks(), ax.get_xticklabels()):
        tick.set_pad(14)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], size=9, color="gray")

    # panel label in upper-left (outside the radar)
    ax.text(-0.12, 1.12, panel_label, transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="top", ha="left")

# ── legend ───────────────────────────────────────────────────────────────────

handles = []
for method, style in METHOD_STYLES.items():
    patch = mpatches.Patch(color=style["color"], label=style["label"])
    handles.append(patch)

fig.legend(
    handles=handles,
    loc="lower center",
    ncol=2,
    fontsize=11,
    frameon=True,
    bbox_to_anchor=(0.5, 0.01),
    title="Method",
    title_fontsize=11,
)

plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
