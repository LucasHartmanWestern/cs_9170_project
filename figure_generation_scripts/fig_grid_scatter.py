"""
Fairness-utility scatter plot for FORGE grid search (Figure 2).

Each point is one confirmed (>=MIN_SEEDS seeds) grid configuration.
x-axis: beta-AUC (utility, higher is better)
y-axis: beta-EO (fairness, lower is better)

Horizontal dashed line marks the best baseline per dataset.
Points below the line beat the best baseline on fairness.
Selected configuration is highlighted with a star.

Once all grid seeds are complete, regenerate by setting MIN_SEEDS=1
or simply re-running — the filter becomes a no-op when every run has >=3 seeds.
"""


import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
FORGE = Path("/storage_1/epigou_storage/FORGE")

DIRS = {
    "census":    [FORGE / "experiment1/census_grid"],
    "capture24": [FORGE / "experiment1/capture24_grid"],
}

DATASET_FILTER = {
    "census":    lambda n: n.startswith("SPECcensus"),
    "capture24": lambda n: n.startswith("SPECforge") or n.startswith("SPECtest"),
}

MIN_SEEDS = 2  # set to 1 once all runs are complete

# Best baseline beta-EO per dataset — EXP-050 matched-epoch results
BEST_BASELINE = {"census": 0.057, "capture24": 0.122}
BASELINE_LABEL = {"census": "Best baseline (GDRO)", "capture24": "Best baseline (GDRO)"}

# Selected configuration per dataset
SELECTED = {
    "census":    {"k": 10.0, "pca": 10, "syn_pct": 40, "ep": 30},
    "capture24": {"k": 5.0, "pca": 15, "syn_pct": 20, "ep": 10},
}

DATASETS = [("census", "Census Income"), ("capture24", "Capture-24")]

# ── Colours ───────────────────────────────────────────────────────────────────
COL_BEAT  = "#4C72B0"   # beats baseline
COL_MISS  = "#AABDD4"   # does not beat baseline
COL_SEL   = "#2CA02C"   # selected config
COL_LINE  = "#D62728"   # baseline reference line

# ── Data loading ──────────────────────────────────────────────────────────────
from paper_figures.utils import read_run, load_dataset_runs


def load_dataset(dataset):
    return load_dataset_runs(DIRS[dataset], DATASET_FILTER[dataset], min_seeds=MIN_SEEDS)


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2),
                         gridspec_kw={"wspace": 0.32})

for ax, (key, ds_label) in zip(axes, DATASETS):
    df  = load_dataset(key)
    sel = SELECTED[key]
    baseline = BEST_BASELINE[key]

    beats = df["eo"] < baseline

    # Background (non-competitive) points
    ax.scatter(df.loc[~beats, "auc"], df.loc[~beats, "eo"],
               s=30, color=COL_MISS, alpha=0.6, zorder=2, linewidths=0)

    # Competitive points (beat baseline)
    ax.scatter(df.loc[beats, "auc"], df.loc[beats, "eo"],
               s=40, color=COL_BEAT, alpha=0.85, zorder=3, linewidths=0)

    # Selected config
    sel_mask = ((df["k"]       == sel["k"]) &
                (df["pca"]     == sel["pca"]) &
                (df["syn_pct"] == sel["syn_pct"]) &
                (df["ep"]      == sel["ep"]))
    if sel_mask.any():
        sx = df.loc[sel_mask, "auc"].values[0]
        sy = df.loc[sel_mask, "eo"].values[0]
        ax.scatter(sx, sy, s=220, color=COL_SEL, marker="*",
                   zorder=5, edgecolors="white", linewidths=0.6)

    # Best baseline reference line
    ax.axhline(baseline, color=COL_LINE, linestyle="--",
               linewidth=1.2, alpha=0.75, zorder=1)
    ax.text(df["auc"].min() + 0.002, baseline + 0.004,
            BASELINE_LABEL[key], color=COL_LINE, fontsize=7.5, va="bottom")

    # Shaded region below baseline (competitive zone)
    ax.axhspan(0, baseline, alpha=0.04, color=COL_BEAT, zorder=0)

    # Count label
    n_beat = beats.sum()
    n_total = len(df)
    ax.text(0.97, 0.97, f"{n_beat}/{n_total} configs\nbeat baseline",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="top",
            color=COL_BEAT)

    ax.set_title(ds_label, fontsize=10, pad=6)
    ax.set_xlabel("AUC" + r" $\uparrow$", fontsize=9)
    ax.set_ylabel(r"$\beta$-EO $\downarrow$", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="both", linewidth=0.4, alpha=0.35, color="gray")

    print(f"{ds_label}: {n_beat}/{n_total} configs beat baseline | "
          f"selected EO={sy:.3f} AUC={sx:.3f}")

# Legend
legend_handles = [
    mlines.Line2D([0], [0], marker="*", color="w", markerfacecolor=COL_SEL,
                  markersize=13, label="Selected configuration"),
    mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_BEAT,
                  markersize=8, label="Beats best baseline"),
    mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_MISS,
                  markersize=8, label="Does not beat best baseline"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.06))

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = Path("figs/grid")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "fig_grid_scatter.pdf", bbox_inches="tight")
fig.savefig(out_dir / "fig_grid_scatter.png", dpi=150, bbox_inches="tight")
print(f"Saved to {out_dir}/fig_grid_scatter.{{pdf,png}}")

paper_fig_dir = Path("paper/figures")
if paper_fig_dir.exists():
    shutil.copy(out_dir / "fig_grid_scatter.png",
                paper_fig_dir / "fig_grid_scatter.png")
    shutil.copy(out_dir / "fig_grid_scatter.pdf",
                paper_fig_dir / "fig_grid_scatter.pdf")
    print("Copied to paper/figures/")
