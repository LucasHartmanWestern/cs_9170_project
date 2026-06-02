"""
Aggregate grid-search results across all completed runs for a dataset.

Reads final_test_metrics.csv + meta.json from each seed directory,
then produces:
  1. Ranked table of top configs by mean β-EO
  2. Marginal effect plots: β-EO vs each hyperparameter (k, pca, ep, ratio)
  3. Heatmaps for key parameter interactions (k×ep, k×pca)
  4. EO reduction bar chart (α-EO minus β-EO) per config group

Supports multiple storage roots (pass --storage multiple times).

Usage:
  # Capture24 from Lambda + Aulavik:
  python analyze_grid.py capture24 \
    --storage ~/cs_9170_project/training_runs \
    --output figs/grid_capture24

  # Census from Huron:
  python analyze_grid.py census_income \
    --storage /storage_1/epigou_storage/FORGE/training_runs \
    --output figs/grid_census
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── column names from final_test_metrics.csv ─────────────────────────────────
COL_ALPHA_EO = "alpha_eo_tpr_diff"
COL_BETA_EO  = "beta_eo_tpr_diff"
COL_ALPHA_F1 = "alpha_f1_weighted"
COL_BETA_F1  = "beta_f1_weighted"
COL_ALPHA_AUC = "alpha_roc_auc"
COL_BETA_AUC  = "beta_roc_auc"

PARAM_LABELS = {
    "k":     "Sigmoid sharpness (k)",
    "pca":   "PCA components",
    "ep":    "FFNN epochs per episode",
    "ratio": "Trajectory ratio",
}


# ── data collection ───────────────────────────────────────────────────────────

def collect(storage_paths: list[Path], dataset: str) -> pd.DataFrame:
    rows = []
    pattern = f"SPEC{dataset}_k*"
    for root in storage_paths:
        for run_dir in sorted(root.glob(pattern)):
            for seed_dir in sorted(run_dir.glob("seed_*")):
                meta_path = seed_dir / "meta.json"
                csv_path  = seed_dir / "final_test_metrics.csv"
                if not meta_path.exists() or not csv_path.exists():
                    continue
                try:
                    meta = json.loads(meta_path.read_text())
                    df   = pd.read_csv(csv_path)
                except Exception:
                    continue
                if df.empty:
                    continue
                r = df.iloc[-1].to_dict()
                r["k"]       = float(meta.get("global_sigmoid_k", 0))
                r["pca"]     = int(meta.get("pca_components", 10))
                r["ep"]      = int(meta.get("ffnn_epochs", 10))
                traj         = meta.get("TRAJ_LENGTH", meta.get("traj_length", 2000))
                real         = meta.get("REAL_DATA_SIZE", meta.get("real_data_size", 3000))
                r["ratio"]   = round(traj / (traj + real), 2)
                r["seed"]    = meta.get("seed")
                r["run_dir"] = run_dir.name
                rows.append(r)
    return pd.DataFrame(rows)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """If the same (k, pca, ep, ratio, seed) appears in multiple dirs, keep latest."""
    df = df.copy()
    df["_ts"] = df["run_dir"].str.extract(r"GG(\d{12})", expand=False)
    df = df.sort_values("_ts")
    df = df.drop_duplicates(subset=["k", "pca", "ep", "ratio", "seed"], keep="last")
    return df.drop(columns=["_ts"])


def require_n_seeds(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """Drop configs that have fewer than n completed seeds."""
    counts = df.groupby(["k", "pca", "ep", "ratio"])["seed"].count()
    valid  = counts[counts >= n].reset_index()[["k", "pca", "ep", "ratio"]]
    return df.merge(valid, on=["k", "pca", "ep", "ratio"])


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Group by config, compute mean ± std for key metrics."""
    df = df.copy()
    # Relative EO reduction per seed: (α-EO − β-EO) / α-EO, clipped to [-1, 1]
    # This removes the α-EO confound when comparing across PCA dimensions.
    alpha = df[COL_ALPHA_EO].clip(lower=1e-6)
    df["rel_eo_reduction"] = ((alpha - df[COL_BETA_EO]) / alpha).clip(-1, 1)

    agg = df.groupby(["k", "pca", "ep", "ratio"]).agg(
        n_seeds            = ("seed",             "count"),
        alpha_eo_mean      = (COL_ALPHA_EO,       "mean"),
        alpha_eo_std       = (COL_ALPHA_EO,       "std"),
        beta_eo_mean       = (COL_BETA_EO,        "mean"),
        beta_eo_std        = (COL_BETA_EO,        "std"),
        rel_eo_red_mean    = ("rel_eo_reduction",  "mean"),
        rel_eo_red_std     = ("rel_eo_reduction",  "std"),
        beta_f1_mean       = (COL_BETA_F1,        "mean"),
        beta_auc_mean      = (COL_BETA_AUC,       "mean"),
    ).reset_index()
    agg["delta_eo_mean"] = agg["alpha_eo_mean"] - agg["beta_eo_mean"]
    agg["delta_eo_std"]  = np.sqrt(agg["alpha_eo_std"]**2 + agg["beta_eo_std"]**2)
    return agg.sort_values("beta_eo_mean")


# ── plotting ──────────────────────────────────────────────────────────────────

BLUE  = "#2c7bb6"
GREEN = "#1a9641"
RED   = "#d7191c"


def _box_by_param(seed_df: pd.DataFrame, param: str, outpath: Path, title: str):
    """Box-plot of β-EO distributions grouped by one hyperparameter value."""
    levels = sorted(seed_df[param].unique())
    data   = [seed_df.loc[seed_df[param] == lv, COL_BETA_EO].values for lv in levels]

    fig, ax = plt.subplots(figsize=(max(4, 1.5 * len(levels)), 4))
    bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="black", lw=2))
    for patch in bp["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_alpha(0.6)

    ax.set_xticks(range(1, len(levels) + 1))
    ax.set_xticklabels([str(lv) for lv in levels])
    ax.set_xlabel(PARAM_LABELS.get(param, param))
    ax.set_ylabel("β-EO (lower = fairer)")
    ax.set_title(title)
    ax.yaxis.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_marginals(seed_df: pd.DataFrame, out_dir: Path, dataset: str):
    """One box plot per hyperparameter, marginalising over the others."""
    for param in ["k", "pca", "ep", "ratio"]:
        if seed_df[param].nunique() < 2:
            continue
        _box_by_param(
            seed_df, param,
            out_dir / f"marginal_{param}.png",
            title=f"{dataset}: β-EO by {PARAM_LABELS.get(param, param)}",
        )


def plot_heatmap(agg_df: pd.DataFrame, row_param: str, col_param: str,
                 out_dir: Path, dataset: str):
    """Heatmap of mean β-EO for a pair of parameters (averaged over the rest)."""
    sub = agg_df.groupby([row_param, col_param])["beta_eo_mean"].mean().reset_index()
    pivot = sub.pivot(index=row_param, columns=col_param, values="beta_eo_mean")

    fig, ax = plt.subplots(figsize=(max(4, 1.2 * pivot.shape[1]),
                                    max(3, 1.0 * pivot.shape[0])))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
                   vmin=pivot.values.min(), vmax=pivot.values.max())
    plt.colorbar(im, ax=ax, label="mean β-EO")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_xticklabels([str(v) for v in pivot.columns])
    ax.set_yticklabels([str(v) for v in pivot.index])
    ax.set_xlabel(PARAM_LABELS.get(col_param, col_param))
    ax.set_ylabel(PARAM_LABELS.get(row_param, row_param))
    ax.set_title(f"{dataset}: mean β-EO ({row_param} × {col_param})")

    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            val = pivot.values[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color="black")

    fig.tight_layout()
    fig.savefig(out_dir / f"heatmap_{row_param}_x_{col_param}.png", dpi=150)
    plt.close(fig)


def plot_delta_bars(agg_df: pd.DataFrame, param: str, out_dir: Path, dataset: str):
    """Bar chart of relative EO reduction grouped by one hyperparameter.

    Uses (α-EO − β-EO) / α-EO to remove the confound where different PCA
    dimensions or datasets have different baseline α-EO values.
    """
    sub = agg_df.groupby(param).agg(
        rel_mean=("rel_eo_red_mean", "mean"),
        rel_sem =("rel_eo_red_mean", lambda x: x.std() / np.sqrt(len(x))),
    ).reset_index()

    levels = sub[param].tolist()
    fig, ax = plt.subplots(figsize=(max(4, 1.5 * len(levels)), 4))
    colors = [GREEN if v >= 0 else RED for v in sub["rel_mean"]]
    ax.bar(range(len(levels)), sub["rel_mean"], yerr=sub["rel_sem"],
           capsize=4, color=colors, alpha=0.75, edgecolor="black")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([str(lv) for lv in levels])
    ax.set_xlabel(PARAM_LABELS.get(param, param))
    ax.set_ylabel("Relative EO reduction  (α-EO − β-EO) / α-EO")
    ax.set_title(f"{dataset}: Relative EO reduction by {PARAM_LABELS.get(param, param)}")
    ax.yaxis.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / f"delta_bar_{param}.png", dpi=150)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyse FORGE grid search results.")
    parser.add_argument("dataset", help="Dataset name prefix, e.g. capture24 or census_income")
    parser.add_argument("--storage", action="append", required=True,
                        help="Path(s) to training_runs directory (repeat for multiple servers)")
    parser.add_argument("--output",  default="figs/grid_analysis",
                        help="Output directory for figures and CSV")
    parser.add_argument("--min-seeds", type=int, default=2,
                        help="Minimum completed seeds to include a config (default: 2)")
    parser.add_argument("--top-n",   type=int, default=20,
                        help="Number of top configs to show in summary table")
    parser.add_argument("--min-alpha-eo", type=float, default=0.0,
                        help="Minimum α-EO per seed to include (filters projections "
                             "that don't preserve the group gap, e.g. --min-alpha-eo 0.10)")
    args = parser.parse_args()

    storage_paths = [Path(p).expanduser() for p in args.storage]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect and clean
    print(f"Collecting runs for '{args.dataset}' from {len(storage_paths)} storage path(s)...")
    raw = collect(storage_paths, args.dataset)
    if raw.empty:
        print("No completed runs found. Check --storage paths and dataset name.")
        return

    print(f"  Raw rows: {len(raw)}")
    raw = deduplicate(raw)
    print(f"  After dedup: {len(raw)}")
    raw = require_n_seeds(raw, n=args.min_seeds)
    print(f"  After min-seeds filter ({args.min_seeds}): {len(raw)} seed-rows")
    if args.min_alpha_eo > 0:
        before = len(raw)
        raw = raw[raw[COL_ALPHA_EO] >= args.min_alpha_eo]
        raw = require_n_seeds(raw, n=args.min_seeds)  # re-check after filter
        print(f"  After min-alpha-eo filter ({args.min_alpha_eo}): {len(raw)} seed-rows "
              f"(dropped {before - len(raw)})")

    # Params present in data
    for p in ["k", "pca", "ep", "ratio"]:
        print(f"  {p} levels: {sorted(raw[p].unique())}")

    # Aggregate to per-config stats
    agg = aggregate(raw)
    print(f"\nTop {args.top_n} configs by β-EO:")
    print(agg[["k","pca","ep","ratio","n_seeds","alpha_eo_mean","beta_eo_mean",
               "beta_eo_std","rel_eo_red_mean","delta_eo_mean","beta_f1_mean","beta_auc_mean"]]
          .head(args.top_n).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Save tables
    raw.to_csv(out_dir / "grid_seed_rows.csv", index=False)
    agg.to_csv(out_dir / "grid_config_summary.csv", index=False)
    print(f"\nSaved seed-level CSV → {out_dir / 'grid_seed_rows.csv'}")
    print(f"Saved config summary  → {out_dir / 'grid_config_summary.csv'}")

    # Plots
    print("\nGenerating plots...")
    plot_marginals(raw, out_dir, args.dataset)
    print("  Marginal box plots done.")

    for r, c in [("k", "ep"), ("k", "pca"), ("ep", "pca")]:
        if agg[r].nunique() > 1 and agg[c].nunique() > 1:
            plot_heatmap(agg, r, c, out_dir, args.dataset)
    print("  Heatmaps done.")

    for param in ["k", "pca", "ep", "ratio"]:
        if raw[param].nunique() > 1:
            plot_delta_bars(agg, param, out_dir, args.dataset)
    print("  Delta bar charts done.")

    print(f"\nAll outputs written to: {out_dir}/")


if __name__ == "__main__":
    main()
