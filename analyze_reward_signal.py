"""
Reward signal quality analysis across grid-search runs.

For each k value, reads metrics.csv from all matching seed directories and computes:
  - Deadzone fraction: fraction of phase-1 episodes where global_obj < 0.5
  - Episode return variance: std of per-episode cumulative return
  - Return distribution: box plots of global_obj by k

Outputs figures to --output directory.

Usage:
  python analyze_reward_signal.py census \
    --storage /storage_1/epigou_storage/FORGE/training_runs \
    --storage /storage_1/epigou_storage/FORGE/aulavik_runs \
    --output figs/reward_signal_census

  python analyze_reward_signal.py capture24 \
    --storage /path/to/lambda_runs \
    --output figs/reward_signal_capture24
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COL_GLOBAL_OBJ = "global.global_obj"
COL_RETURN     = "meta.episode_return"
COL_AVG_REWARD = "meta.avg_reward"

DEADZONE_THRESHOLD = 0.5  # global_obj < this => deadzone


def load_runs(dataset: str, storage_roots: list[Path]) -> pd.DataFrame:
    """Collect one row per seed-run with k, metrics arrays, and summary stats."""
    records = []
    prefix = f"SPEC{dataset}_k"
    seen = {}  # (k, pca, ep, ratio, seed) -> timestamp for dedup

    for root in storage_roots:
        for run_dir in sorted(root.glob(f"{prefix}*")):
            if not run_dir.is_dir():
                continue
            for seed_dir in sorted(run_dir.glob("seed_*")):
                meta_path = seed_dir / "meta.json"
                metrics_path = seed_dir / "metrics.csv"
                if not meta_path.exists() or not metrics_path.exists():
                    continue

                try:
                    meta = json.load(open(meta_path))
                    k   = float(meta.get("global_sigmoid_k", 0))
                    ep  = int(meta.get("ffnn_epochs") or 10)
                    pca = int(meta.get("pca_components") or 10)
                    traj = int(meta.get("TRAJ_LENGTH") or meta.get("traj_length") or 2000)
                    real = int(meta.get("REAL_DATA_SIZE") or meta.get("real_data_size") or 3000)
                    seed = int(meta.get("seed", seed_dir.name.split("_")[-1]))
                    ratio = round(traj / (traj + real), 3)
                    ts   = run_dir.name  # use dir name as timestamp proxy

                    key = (k, pca, ep, ratio, seed)
                    if key in seen and seen[key] >= ts:
                        continue
                    seen[key] = ts

                    df = pd.read_csv(metrics_path)
                    if COL_GLOBAL_OBJ not in df.columns:
                        continue

                    obj = df[COL_GLOBAL_OBJ].dropna().values
                    ret = df[COL_RETURN].dropna().values if COL_RETURN in df.columns else np.array([])

                    if len(obj) < 10:
                        continue

                    deadzone_frac = float((obj < DEADZONE_THRESHOLD).mean())
                    obj_mean = float(obj.mean())
                    obj_std  = float(obj.std())
                    ret_std  = float(ret.std()) if len(ret) > 1 else np.nan

                    records.append({
                        "k": k, "ep": ep, "pca": pca, "ratio": ratio, "seed": seed,
                        "deadzone_frac": deadzone_frac,
                        "obj_mean": obj_mean,
                        "obj_std": obj_std,
                        "ret_std": ret_std,
                        "n_episodes": len(obj),
                        "obj_values": obj,
                    })
                except Exception:
                    continue

    return pd.DataFrame(records)


def plot_deadzone_by_k(df: pd.DataFrame, out_dir: Path, dataset: str):
    """Bar chart: mean deadzone fraction per k, with std error bars."""
    summary = (
        df.groupby("k")["deadzone_frac"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["se"] = summary["std"] / np.sqrt(summary["count"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ks = summary["k"].values
    means = summary["mean"].values
    ses = summary["se"].values

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(ks)))
    bars = ax.bar(range(len(ks)), means * 100, yerr=ses * 100,
                  color=colors, edgecolor="black", capsize=4, linewidth=0.8)

    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"k={k:.0f}" for k in ks])
    ax.set_ylabel("Deadzone fraction (%)")
    ax.set_title(f"Reward deadzone by sigmoid sharpness — {dataset}")
    ax.axhline(20, color="red", linestyle="--", linewidth=0.8, label="20% threshold")
    ax.legend()
    ax.set_ylim(0, max(means.max() * 100 * 1.4, 25))

    for bar, m, n in zip(bars, means, summary["count"].values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{m*100:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = out_dir / "deadzone_by_k.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_obj_distribution_by_k(df: pd.DataFrame, out_dir: Path, dataset: str):
    """Box plots of episode global_obj values for each k."""
    ks = sorted(df["k"].unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(3 * len(ks), 4), sharey=True)
    if len(ks) == 1:
        axes = [axes]

    for ax, k in zip(axes, ks):
        sub = df[df["k"] == k]
        all_obj = np.concatenate(sub["obj_values"].values)
        ax.hist(all_obj, bins=40, orientation="horizontal", color="steelblue",
                edgecolor="none", alpha=0.75, density=True)
        ax.axhline(DEADZONE_THRESHOLD, color="red", linestyle="--",
                   linewidth=1, label="deadzone boundary")
        dz = float((all_obj < DEADZONE_THRESHOLD).mean())
        ax.set_title(f"k={k:.0f}\n{dz*100:.1f}% DZ", fontsize=10)
        ax.set_xlabel("Density")
        if ax == axes[0]:
            ax.set_ylabel("global_obj")

    axes[0].legend(fontsize=8)
    fig.suptitle(f"Episode reward distribution by k — {dataset}", fontsize=11)
    plt.tight_layout()
    path = out_dir / "obj_dist_by_k.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_return_variance_by_k(df: pd.DataFrame, out_dir: Path, dataset: str):
    """Box plot of per-seed return std grouped by k — shows learning stability."""
    ks = sorted(df["k"].unique())
    data = [df[df["k"] == k]["obj_std"].dropna().values for k in ks]
    labels = [f"k={k:.0f}" for k in ks]

    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2})

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(ks)))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_ylabel("Std of global_obj within run")
    ax.set_title(f"Reward variance (learning stability) by k — {dataset}")
    plt.tight_layout()
    path = out_dir / "reward_variance_by_k.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def print_summary_table(df: pd.DataFrame, dataset: str):
    print(f"\n{'='*60}")
    print(f"Reward signal summary — {dataset}")
    print(f"{'='*60}")
    print(f"{'k':>5}  {'runs':>5}  {'DZ mean':>9}  {'DZ std':>8}  {'obj_mean':>9}  {'obj_std':>9}")
    print("-" * 60)
    for k, g in df.groupby("k"):
        print(f"{k:>5.0f}  {len(g):>5}  {g['deadzone_frac'].mean()*100:>8.1f}%  "
              f"{g['deadzone_frac'].std()*100:>7.1f}%  "
              f"{g['obj_mean'].mean():>9.4f}  {g['obj_std'].mean():>9.4f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Dataset prefix, e.g. 'census' or 'capture24'")
    parser.add_argument("--storage", action="append", required=True, dest="storage",
                        help="Storage root(s) to scan (repeat for multiple)")
    parser.add_argument("--output", required=True, help="Output directory for figures")
    parser.add_argument("--min-episodes", type=int, default=100,
                        help="Minimum episodes for a run to be included")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage_roots = [Path(s) for s in args.storage]

    print(f"Collecting runs for '{args.dataset}' from {len(storage_roots)} storage root(s)...")
    df = load_runs(args.dataset, storage_roots)

    if df.empty:
        print("No runs found. Check --storage paths and dataset prefix.")
        return

    df = df[df["n_episodes"] >= args.min_episodes]
    print(f"Found {len(df)} seed-runs across k={sorted(df['k'].unique())}")

    print_summary_table(df, args.dataset)
    plot_deadzone_by_k(df, out_dir, args.dataset)
    plot_obj_distribution_by_k(df, out_dir, args.dataset)
    plot_return_variance_by_k(df, out_dir, args.dataset)
    print("\nDone.")


if __name__ == "__main__":
    main()
