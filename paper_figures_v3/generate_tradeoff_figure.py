"""
Generate EO vs AUC tradeoff scatter plot from main_table_metrics.csv.
Plots mean values only (no error bars), one marker per method per dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR  = "/home/epigou/cs_9170_project/paper_figures_v3"
CSV_PATH = os.path.join(OUT_DIR, "main_table_metrics.csv")

df = pd.read_csv(CSV_PATH)

DS_LABELS = {"Census": "Census", "COMPAS": "COMPAS", "Capture-24": "Capture-24"}
DATASETS  = ["Census", "COMPAS", "Capture-24"]

METHOD_ORDER = [
    "Alpha", "GroupDRO", "OT Repair", "FLB",
    "FairTabDDPM", "SMOTE", "CTGAN", "RL (ours)"
]

# Marker styles per method
MARKERS = {
    "Alpha":       ("o",  "#aaaaaa",  7, "Alpha",         False),
    "GroupDRO":    ("s",  "#e67e22",  7, "Group DRO",     False),
    "OT Repair":   ("^",  "#8e44ad",  7, "OT Repair",     False),
    "FLB":         ("D",  "#2980b9",  7, "FLB",           False),
    "FairTabDDPM": ("p",  "#16a085",  8, "FairTabDDPM",   False),
    "SMOTE":       ("h",  "#c0392b",  7, "SMOTE",         False),
    "CTGAN":       ("v",  "#7f8c8d",  7, "CTGAN",         False),
    "RL (ours)":   ("*",  "#f39c12", 13, "RL (ours)",     True),
}

DS_COLORS = {
    "Census":    "#2b4590",
    "COMPAS":    "#c0392b",
    "Capture-24": "#27ae60",
}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
fig.suptitle("Fairness-Utility Tradeoff: EO gap vs. AUC (mean, 5 seeds)",
             fontsize=11, fontweight="bold")

for ci, ds in enumerate(DATASETS):
    ax = axes[ci]
    ds_df = df[df["dataset"] == ds]

    for method in METHOD_ORDER:
        row = ds_df[ds_df["method"] == method]
        if len(row) == 0:
            continue
        eo  = float(row["eo_mean"])
        auc = float(row["auc_mean"])

        mark, color, size, label, is_rl = MARKERS[method]
        zorder = 5 if is_rl else 3

        ax.scatter(auc, eo, marker=mark, color=color, s=size**2,
                   zorder=zorder,
                   edgecolors="#111111" if is_rl else "none",
                   linewidths=1.2 if is_rl else 0,
                   label=label if ci == 0 else None)

        # Label RL and Alpha only inline; rest covered by legend
        if method in ("RL (ours)", "Alpha"):
            ax.annotate(label,
                        xy=(auc, eo),
                        xytext=(auc + 0.002, eo + 0.005),
                        fontsize=7,
                        color=color,
                        fontweight="bold" if is_rl else "normal",
                        ha="left", va="bottom",
                        clip_on=True)

    # Ideal direction arrow
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.annotate("", xy=(0.98, 0.05), xytext=(0.88, 0.15),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0))
    ax.text(0.93, 0.02, "ideal", transform=ax.transAxes,
            fontsize=6.5, color="#555555", ha="center")

    ax.set_xlabel("AUC ↑", fontsize=9)
    ax.set_ylabel("EO gap ↓", fontsize=9)
    ax.set_title(ds, fontsize=10, fontweight="bold", color=DS_COLORS[ds])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7.5)
    ax.grid(linewidth=0.3, alpha=0.5)

# Legend on first axis only
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=8,
           fontsize=7.5, frameon=True, bbox_to_anchor=(0.5, -0.06))

fig.tight_layout(rect=[0, 0.04, 1, 1])

out_path = os.path.join(OUT_DIR, "fig_tradeoff_v3.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out_path}")
