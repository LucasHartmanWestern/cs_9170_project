"""
plot_centroid_drift.py — Quantify whether the FORGE policy learns to concentrate
synthetic generation toward the disadvantaged-positive cluster in PCA space.

Two metrics computed per snapshot episode, averaged across seeds:
  1. Centroid distance  — L2 distance from synthetic centroid to real
                          disadvantaged-positive centroid in PCA space.
                          Decreasing = policy drifts toward target group.
  2. Cosine similarity  — cosine between synthetic centroid and
                          disadvantaged-positive centroid vectors.
                          Increasing toward 1.0 = directional alignment.

Output: <run_dir>/analysis/fig_centroid_drift.png

Usage:
    python plot_centroid_drift.py training_runs/<run_dir> [--device cpu]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Helpers (mirrors check_run.py reconstruct_dataset)
# ---------------------------------------------------------------------------

def load_meta(seed_dir: Path) -> dict:
    with open(seed_dir / "meta.json") as f:
        return json.load(f)


def find_seed_dirs(run_dir: Path) -> list[Path]:
    return sorted(
        d for d in run_dir.iterdir()
        if d.is_dir() and d.name.startswith("seed_")
    )


def reconstruct_dataset(meta: dict, device: str):
    from dataset import Dataset
    ds = Dataset(
        dataset_name=meta["dataset_name"],
        multiclass=bool(meta.get("multiclass", False)),
        minority_id=int(meta.get("minority_id", 1)),
        majority_id=int(meta.get("majority_id", 0)),
        third_id=meta.get("third_id", None),
        pca_components=int(meta.get("pca_components", 10)),
        seed=int(meta["seed"]),
        device=device,
        use_pca=True,
    )
    kwargs = dict(
        train_size=meta.get("REAL_DATA_SIZE", None),
        bias_pct=meta.get("BIAS_PCT", None),
        da_pct=meta.get("DA_PCT", None),
        pca_components=int(meta.get("pca_components", 10)),
        drop_protected=False,
        protected_cols=ds.protected_attributes,
        win_seconds=meta.get("win_seconds", 1.0),
        step_seconds=meta.get("step_seconds", 0.5),
    )
    dp_col = meta.get("dp_protected_col", None)
    if dp_col is not None:
        kwargs["dp_protected_col"] = dp_col
    # kfold params — present in capture-24 fold runs
    if meta.get("n_folds"):
        kwargs["n_folds"]        = int(meta["n_folds"])
        kwargs["fold_idx"]       = int(meta["fold_idx"])
        kwargs["fold_rng_seed"]  = int(meta.get("fold_rng_seed", 6))
        kwargs["kfold_val_frac"] = float(meta.get("kfold_val_frac", 0.4))
    splits = ds.get_data_splits(**kwargs)
    x_tr, _, _, y_tr, _, _ = splits
    return ds, x_tr.cpu().numpy(), y_tr.cpu().numpy()


def snapshot_episodes(snap_dir: Path) -> list[int]:
    eps = []
    for f in snap_dir.glob("synthetic_ep*_phase1_class1.npz"):
        try:
            eps.append(int(f.stem.split("_ep")[1].split("_")[0]))
        except (IndexError, ValueError):
            pass
    return sorted(eps)


def compute_seed_metrics(seed_dir: Path, meta: dict, device: str) -> dict | None:
    snap_dir = seed_dir / "synthetic_snapshots"
    if not snap_dir.exists():
        return None

    eps = snapshot_episodes(snap_dir)
    if not eps:
        return None

    # Real data for this seed
    ds, X_tr, y_tr = reconstruct_dataset(meta, device)
    a_tr = ds.a_train.cpu().numpy()
    min_id = int(meta.get("minority_id", 1))

    # Disadvantaged-positive centroid
    disadv_pos_mask = (a_tr == min_id) & (y_tr == 1)
    n_dp = disadv_pos_mask.sum()
    if n_dp == 0:
        print(f"  [WARN] No disadvantaged positives in seed {meta['seed']} — skipping.")
        return None
    dp_centroid = X_tr[disadv_pos_mask].mean(axis=0)
    dp_norm = np.linalg.norm(dp_centroid)

    ep_list, dist_list, cos_list = [], [], []
    for ep in eps:
        snap_path = snap_dir / f"synthetic_ep{ep:04d}_phase1_class1.npz"
        if not snap_path.exists():
            continue
        pts = np.load(snap_path)["x"]          # (T, pca_dims)
        syn_centroid = pts.mean(axis=0)

        dist = np.linalg.norm(syn_centroid - dp_centroid)
        syn_norm = np.linalg.norm(syn_centroid)
        cos = (np.dot(syn_centroid, dp_centroid) / (syn_norm * dp_norm + 1e-8))

        ep_list.append(ep)
        dist_list.append(dist)
        cos_list.append(cos)

    return {
        "episodes": np.array(ep_list),
        "dist":     np.array(dist_list),
        "cos":      np.array(cos_list),
        "n_dp":     n_dp,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Plot centroid drift of synthetic points toward disadv-positive cluster.")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    assert run_dir.exists(), f"Run dir not found: {run_dir}"

    seed_dirs = find_seed_dirs(run_dir)
    assert seed_dirs, f"No seed_* subdirectories in {run_dir}"

    print(f"Run: {run_dir.name}")
    print(f"Seeds: {[s.name for s in seed_dirs]}")

    # Collect per-seed metrics
    all_eps, all_dist, all_cos = [], [], []
    for sd in seed_dirs:
        meta = load_meta(sd)
        print(f"  Processing {sd.name} (dataset={meta['dataset_name']}, seed={meta['seed']})…")
        result = compute_seed_metrics(sd, meta, args.device)
        if result is None:
            print(f"  Skipping {sd.name} — no snapshots.")
            continue
        print(f"    n_disadv_pos={result['n_dp']}, snapshots={len(result['episodes'])}, "
              f"dist: {result['dist'][0]:.3f} → {result['dist'][-1]:.3f}, "
              f"cos: {result['cos'][0]:.3f} → {result['cos'][-1]:.3f}")
        all_eps.append(result["episodes"])
        all_dist.append(result["dist"])
        all_cos.append(result["cos"])

    if not all_eps:
        print("No valid seeds found. Exiting.")
        return

    # Align all seeds to common episode grid (intersection)
    common_eps = all_eps[0]
    for ep_arr in all_eps[1:]:
        common_eps = np.intersect1d(common_eps, ep_arr)

    def align(arr_list, eps_list):
        out = []
        for arr, eps in zip(arr_list, eps_list):
            idx = np.isin(eps, common_eps)
            out.append(arr[idx])
        return np.array(out)   # (n_seeds, n_eps)

    dist_mat = align(all_dist, all_eps)
    cos_mat  = align(all_cos,  all_eps)

    dist_mean = dist_mat.mean(axis=0)
    dist_std  = dist_mat.std(axis=0)
    cos_mean  = cos_mat.mean(axis=0)
    cos_std   = cos_mat.std(axis=0)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    meta0 = load_meta(seed_dirs[0])
    dataset_name = meta0["dataset_name"].replace("_", " ").title()
    n_seeds = len(dist_mat)

    # Panel 1: centroid distance
    ax1.fill_between(common_eps, dist_mean - dist_std, dist_mean + dist_std,
                     alpha=0.20, color="steelblue")
    ax1.plot(common_eps, dist_mean, color="steelblue", lw=2.0,
             label=f"Mean $\\pm$ std ({n_seeds} seeds)")
    ax1.set_xlabel("Training episode")
    ax1.set_ylabel("L2 distance")
    ax1.set_title("Centroid distance")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    # Panel 2: cosine similarity
    ax2.fill_between(common_eps, cos_mean - cos_std, cos_mean + cos_std,
                     alpha=0.20, color="darkorange")
    ax2.plot(common_eps, cos_mean, color="darkorange", lw=2.0,
             label=f"Mean $\\pm$ std ({n_seeds} seeds)")
    ax2.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax2.set_xlabel("Training episode")
    ax2.set_ylabel("Cosine similarity")
    ax2.set_title("Directional alignment")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    plt.tight_layout()

    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fig_centroid_drift.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
