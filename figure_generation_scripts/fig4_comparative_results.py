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

# Generated from experiment3 run dirs (census_forge, census_baselines,
# capture24_baselines) — see plot_results.ipynb cell 6 or regenerate manually.
CSV_PATH = "/storage_1/epigou_storage/FORGE/experiment3/main_table_metrics_per_seed.csv"
OUT_PATH = "/home/epigou/cs_9170_project/paper/figures/exp2_comparative_results.png"

DATASETS = ["Census", "Capture-24"]

METHOD_MAP = {
    "Alpha":       {"label": "Alpha",       "color": "tab:red",    "marker": "o"},
    "RL (ours)":   {"label": "FORGE",       "color": "tab:blue",   "marker": "s"},
    "OT Repair":   {"label": "OTR",         "color": "tab:green",  "marker": "^"},
    "GroupDRO":    {"label": "GDRO",        "color": "tab:orange", "marker": "D"},
    "FLB":         {"label": "FLB",         "color": "tab:purple", "marker": "v"},
    "FairTabDDPM": {"label": "FairTabDDPM", "color": "tab:brown",  "marker": "P"},
    "SMOTE":       {"label": "SMOTE",       "color": "tab:pink",   "marker": "X"},
    "CTGAN":       {"label": "CTGAN",       "color": "tab:gray",   "marker": "*"},
}

PANELS = [
    {"col": "eo_tpr_diff",  "title": "EO' Score",  "ylabel": "EO' Score", "invert": True},
    {"col": "roc_auc",      "title": "ROC-AUC",   "ylabel": "ROC-AUC",   "invert": False},
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

for ax, panel in zip(axes, PANELS):
    for csv_name, style in METHOD_MAP.items():
        row = sub[sub["method"] == csv_name].set_index("dataset")
        vals = [row.loc[ds, panel["col"]] for ds in DATASETS if ds in row.index]
        dsets = [ds for ds in DATASETS if ds in row.index]
        ax.plot(dsets, vals,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linewidth=2,
                markersize=7)

    ax.set_ylabel(panel["ylabel"], fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=10)

for ax in axes:
    ax.set_xlabel("Dataset", fontsize=11)

# Ensure EO' Score axis shows 1.0 as an upper reference tick
ax_eo = axes[0]
lo, _ = ax_eo.get_ylim()
ax_eo.set_ylim(lo, 1.04)
ticks = sorted(set(list(ax_eo.get_yticks()) + [1.0]))
ax_eo.set_yticks([t for t in ticks if lo <= t <= 1.04])

# shared legend below the plot
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc="lower center",
           fontsize=9,
           frameon=True,
           title="Method",
           title_fontsize=9,
           ncol=4,
           bbox_to_anchor=(0.5, -0.03))

plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
