"""
Regenerate exp2_comparative_results.png.

Layout  : top-down (2 rows × 1 col) — AUC on top, EO Score below
Datasets: Census, Capture-24 (COMPAS dropped)
Methods : Alpha, FORGE (RL ours), OTR (OT Repair), GDRO (GroupDRO)
Colors  : Alpha=red, FORGE=blue, OTR=green, GDRO=orange
Data    : means over seeds from main_table_metrics_per_seed.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CSV_PATH = "/home/epigou/cs_9170_project/figures/main_table_metrics_per_seed.csv"
OUT_PATH = "/home/epigou/cs_9170_project/paper/figures/exp2_comparative_results.png"

DATASETS = ["Census", "Capture-24"]

METHOD_MAP = {
    "Alpha":       {"label": "Alpha",       "color": "tab:red",    "marker": "o", "lw": 1.5},
    "RL (ours)":   {"label": "FORGE",       "color": "#1f77b4",    "marker": "s", "lw": 2.5},
    "GroupDRO":    {"label": "GDRO",        "color": "#ff7f0e",    "marker": "D", "lw": 1.5},
    "OT Repair":   {"label": "OTR",         "color": "#2ca02c",    "marker": "^", "lw": 1.5},
    "FLB":         {"label": "FLB",         "color": "#9467bd",    "marker": "v", "lw": 1.5},
    "FairTabDDPM": {"label": "FairTabDDPM", "color": "#8c564b",    "marker": "P", "lw": 1.5},
    "SMOTE":       {"label": "SMOTE",       "color": "#e377c2",    "marker": "X", "lw": 1.5},
    "CTGAN":       {"label": "CTGAN",       "color": "#7f7f7f",    "marker": "*", "lw": 1.5},
}

PANELS = [
    {"col": "eo_tpr_diff", "ylabel": "EO' Score", "invert": True},
    {"col": "roc_auc",     "ylabel": "ROC-AUC",   "invert": False},
]

EPS = 1e-8

# ── load & aggregate ──────────────────────────────────────────────────────────

df   = pd.read_csv(CSV_PATH)
cols = [p["col"] for p in PANELS]
sub  = (
    df[df["dataset"].isin(DATASETS) & df["method"].isin(METHOD_MAP.keys())]
    .groupby(["method", "dataset"])[cols]
    .mean()
    .reset_index()
)

# Apply EO' inversion per dataset: 1 - EO / (max(EO) + eps)
for ds in DATASETS:
    mask    = sub["dataset"] == ds
    max_eo  = sub.loc[mask, "eo_tpr_diff"].max()
    sub.loc[mask, "eo_tpr_diff"] = 1 - sub.loc[mask, "eo_tpr_diff"] / (max_eo + EPS)

# ── plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4.5), sharex=False)

max_eo_per_ds = {ds: sub[sub["dataset"] == ds]["eo_tpr_diff"].max() for ds in DATASETS}

for ax, panel in zip(axes, PANELS):
    for csv_name, style in METHOD_MAP.items():
        row = sub[sub["method"] == csv_name].set_index("dataset")
        dsets = [ds for ds in DATASETS if ds in row.index]
        if panel["invert"]:
            vals = [1 - row.loc[ds, panel["col"]] / (max_eo_per_ds[ds] + EPS) for ds in dsets]
        else:
            vals = [row.loc[ds, panel["col"]] for ds in dsets]
        ax.plot(dsets, vals,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linewidth=style["lw"],
                markersize=8,
                ls="-",
                zorder=3 if csv_name == "RL (ours)" else 2)

    ax.set_ylabel(panel["ylabel"], fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_xlabel("Dataset", fontsize=11)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.03*(hi-lo), hi + 0.12*(hi-lo))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="lower center",
           fontsize=9,
           frameon=True,
           title="Method",
           title_fontsize=9,
           ncol=4,
           bbox_to_anchor=(0.5, 0.00))

plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
