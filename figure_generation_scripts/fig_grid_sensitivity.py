"""
Main effects plot for FORGE hyperparameter sensitivity (Figure 2).

Layout: 2 rows (Census Income, Capture-24) x 4 cols (k | PCA | Synthetic % | Epochs).

For each panel: mean beta-EO +/- std across all runs that share that parameter value,
averaged over all other parameter dimensions. This is a standard one-way sensitivity
(main effects) decomposition: it answers "what happens to beta-EO when I vary this
parameter, regardless of other choices?"

Only runs with >= MIN_SEEDS completed seeds are included. This filters out partial
single- or double-seed runs that have not yet been confirmed. Once all seeds are
complete (all runs reach 3 seeds), set MIN_SEEDS = 1 to include everything, or
simply regenerate — the filter becomes a no-op when every run has >= 3 seeds.

The selected value per dataset is highlighted with a star marker and dashed vertical
line so the reader can immediately see why it was chosen.
"""


import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── Data roots ────────────────────────────────────────────────────────────────
FORGE = Path("/storage_1/epigou_storage/FORGE")

DIRS = {
    "census":    [FORGE / "experiment1/census_grid"],
    "capture24": [FORGE / "experiment1/capture24_grid"],
}

DATASET_FILTER = {
    "census":    lambda n: n.startswith("SPECcensus"),
    "capture24": lambda n: n.startswith("SPECforge") or n.startswith("SPECtest"),
}

# Minimum number of completed seeds required to include a run.
# Set to 3 while the grid is still running to exclude unconfirmed partial results.
# Once all runs are complete (every config has 3 seeds), this becomes a no-op.
MIN_SEEDS = 3

# Canonical selected configuration per dataset
SELECTED = {
    "census":    {"k": 10.0, "pca": 10, "syn_pct": 40, "ep": 30},
    "capture24": {"k": 5.0, "pca": 15, "syn_pct": 20, "ep": 10},
}

# ── Run loader ────────────────────────────────────────────────────────────────

from paper_figures.utils import read_run, load_dataset_runs


def load_dataset(dataset):
    df = load_dataset_runs(DIRS[dataset], DATASET_FILTER[dataset], min_seeds=MIN_SEEDS)
    if not df.empty:
        df["syn_pct"] = (df["traj"] / 5000 * 100).round().astype(int)
    return df


# ── Main effects computation ──────────────────────────────────────────────────

def main_effect(df, param):
    """
    For each value of `param`, compute the mean and std of beta_eo_mean across
    all runs that share that value (averaging over all other parameter dimensions).
    The std captures how sensitive results are to the other parameters at each level.
    """
    grouped = (
        df.groupby(param)["beta_eo_mean"]
        .agg(eo_mean="mean", eo_std="std", n="count")
        .reset_index()
        .sort_values(param)
    )
    grouped["eo_std"] = grouped["eo_std"].fillna(0)
    return grouped


# ── Plotting ──────────────────────────────────────────────────────────────────

COL_BASE = "#4C72B0"   # muted blue for non-selected points
COL_SEL  = "#2CA02C"   # green for the selected value

PARAMS = [
    ("k",       r"$k$"),
    ("pca",     "PCA dimensions"),
    ("syn_pct", "Synthetic data %"),
    ("ep",      "Classifier epochs"),
]

DATASETS = [
    ("census",    "Census Income"),
    ("capture24", "Capture-24"),
]

# ── Load ──────────────────────────────────────────────────────────────────────
dfs = {}
for key, label in DATASETS:
    df = load_dataset(key)
    dfs[key] = df
    print(f"{label}: {len(df)} completed runs")
    if not df.empty:
        print(
            df[["k", "pca", "syn_pct", "ep", "beta_eo_mean", "n_seeds"]]
            .sort_values(["k", "pca", "syn_pct", "ep"])
            .to_string(index=False)
        )
    print()

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 4, figsize=(12, 5),
    gridspec_kw={"hspace": 0.6, "wspace": 0.38},
)

for row, (key, ds_label) in enumerate(DATASETS):
    df  = dfs[key]
    sel = SELECTED[key]

    # Per-row y-limits: compute after all panels to share within row
    row_ymins, row_ymaxs = [], []

    for col, (param, col_title) in enumerate(PARAMS):
        ax = axes[row, col]

        if df.empty or param not in df.columns:
            ax.set_visible(False)
            continue

        effect  = main_effect(df, param)
        x       = effect[param].values.astype(float)
        y       = effect["eo_mean"].values
        e       = effect["eo_std"].values
        sel_val = float(sel[param])

        # Line connecting points
        ax.plot(x, y, color=COL_BASE, linewidth=1.5, zorder=2)

        # Shaded ±std band
        ax.fill_between(x, np.maximum(0, y - e), y + e,
                        alpha=0.18, color=COL_BASE, zorder=1)

        # Non-selected points
        for xi, yi in zip(x, y):
            if xi != sel_val:
                ax.scatter(xi, yi, s=50, color=COL_BASE, zorder=3,
                           edgecolors="white", linewidth=0.6)

        # Selected value: star + dashed vertical line
        sel_mask = effect[param].astype(float) == sel_val
        if sel_mask.any():
            sel_y = effect.loc[sel_mask, "eo_mean"].values[0]
            ax.scatter(sel_val, sel_y, s=160, color=COL_SEL, zorder=5,
                       marker="*", edgecolors="white", linewidth=0.6)
            ax.axvline(sel_val, color=COL_SEL, linestyle="--",
                       alpha=0.45, linewidth=1.2, zorder=1)

        # x-ticks at data values only
        ax.set_xticks(x)
        if param == "syn_pct":
            ax.set_xticklabels([f"{int(v)}%" for v in x], fontsize=8)
        else:
            ax.set_xticklabels([str(int(v)) for v in x], fontsize=8)

        # Column header on row 0 only
        if row == 0:
            ax.set_title(col_title, fontsize=9, pad=5)

        # y-axis label on leftmost panel only
        ax.set_ylabel(r"$\beta$-EO $\downarrow$" if col == 0 else "",
                      fontsize=8.5)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="y", labelsize=7.5)
        ax.tick_params(axis="x", labelsize=8)

        ax.grid(axis="y", linewidth=0.4, alpha=0.45, color="gray")
        ax.spines[["top", "right"]].set_visible(False)

        row_ymins.append(np.min(np.maximum(0, y - e)))
        row_ymaxs.append(np.max(y + e))

    # Shared y-axis range within each row
    if row_ymins and row_ymaxs:
        pad = (max(row_ymaxs) - min(row_ymins)) * 0.12
        ylo = max(0, min(row_ymins) - pad)
        yhi = max(row_ymaxs) + pad
        for ax in axes[row]:
            if ax.get_visible():
                ax.set_ylim(ylo, yhi)

    # Row label (dataset name) on the left of the row
    axes[row, 0].annotate(
        ds_label,
        xy=(-0.42, 0.5), xycoords="axes fraction",
        fontsize=10, fontweight="bold",
        ha="center", va="center", rotation=90,
    )

# Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [
    Line2D([0], [0], marker="*", color="w", markerfacecolor=COL_SEL,
           markersize=12, label="Selected value"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COL_BASE,
           markersize=8, label="Other values (mean ± std)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=2,
           fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.03))

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = Path("figs/grid")
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "fig_grid_sensitivity.pdf", bbox_inches="tight")
fig.savefig(out_dir / "fig_grid_sensitivity.png", dpi=150, bbox_inches="tight")
print(f"Saved to {out_dir}/fig_grid_sensitivity.{{pdf,png}}")

paper_fig_dir = Path("paper/figures")
if paper_fig_dir.exists():
    shutil.copy(out_dir / "fig_grid_sensitivity.png",
                paper_fig_dir / "fig_grid_sensitivity.png")
    shutil.copy(out_dir / "fig_grid_sensitivity.pdf",
                paper_fig_dir / "fig_grid_sensitivity.pdf")
    print("Copied to paper/figures/")
