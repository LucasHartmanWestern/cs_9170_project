"""
Generate paper/figures/fig_centroid_drift_paper.png — combined 2×2 centroid drift figure.

Layout:
  Row 1 (Census Income):   subplot titles shown, x-axis hidden (shared with row 2)
  Row 2 (Capture-24):      no subplot titles, x-axis shown

Run from project root:  python figures/generate_centroid_drift_paper.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

CENSUS_RUN = Path(
    "/storage_1/epigou_storage/FORGE/aulavik_runs/census/k5/"
    "SPECcensus_k5_gpu1_EP5000_PCA10_REWwgl_minID0_majID1_"
    "TRJ2000_REAL3000_GG202604251803_9af13c63"
)
CAPTURE24_RUN = Path(
    "/storage_1/epigou_storage/FORGE/aulavik_runs/capture_24/k5/"
    "SPECcapture24_k5_gpu0_EP5000_PCA15_REWwgl_minID1_majID0_"
    "TRJ1000_REAL4000_GG202605041119_779cf9c5"
)

OUT_PATH = Path("paper/figures/fig_centroid_drift_paper.png")


# ---------------------------------------------------------------------------
# Data helpers (from plot_centroid_drift.py)
# ---------------------------------------------------------------------------

def load_meta(seed_dir: Path) -> dict:
    with open(seed_dir / "meta.json") as f:
        return json.load(f)


def find_seed_dirs(run_dir: Path) -> list[Path]:
    return sorted(d for d in run_dir.iterdir()
                  if d.is_dir() and d.name.startswith("seed_"))


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


def compute_run_metrics(run_dir: Path, device: str = "cpu") -> dict:
    seed_dirs = find_seed_dirs(run_dir)
    all_eps, all_dist, all_cos = [], [], []

    for sd in seed_dirs:
        meta = load_meta(sd)
        snap_dir = sd / "synthetic_snapshots"
        if not snap_dir.exists():
            continue
        eps = snapshot_episodes(snap_dir)
        if not eps:
            continue

        ds, X_tr, y_tr = reconstruct_dataset(meta, device)
        a_tr = ds.a_train.cpu().numpy()
        min_id = int(meta.get("minority_id", 1))
        mask = (a_tr == min_id) & (y_tr == 1)
        if mask.sum() == 0:
            continue

        dp_centroid = X_tr[mask].mean(axis=0)
        dp_norm = np.linalg.norm(dp_centroid)

        ep_list, dist_list, cos_list = [], [], []
        for ep in eps:
            f = snap_dir / f"synthetic_ep{ep:04d}_phase1_class1.npz"
            if not f.exists():
                continue
            pts = np.load(f)["x"]
            syn_centroid = pts.mean(axis=0)
            dist = np.linalg.norm(syn_centroid - dp_centroid)
            syn_norm = np.linalg.norm(syn_centroid)
            cos = float(np.dot(syn_centroid, dp_centroid) / (syn_norm * dp_norm + 1e-8))
            ep_list.append(ep)
            dist_list.append(dist)
            cos_list.append(cos)

        if ep_list:
            all_eps.append(np.array(ep_list))
            all_dist.append(np.array(dist_list))
            all_cos.append(np.array(cos_list))

    if not all_eps:
        return {}

    common_eps = all_eps[0]
    for e in all_eps[1:]:
        common_eps = np.intersect1d(common_eps, e)

    def align(arrays, eps_list):
        out = []
        for arr, ep in zip(arrays, eps_list):
            out.append(arr[np.isin(ep, common_eps)])
        return np.array(out)

    dist_mat = align(all_dist, all_eps)
    cos_mat  = align(all_cos,  all_eps)

    return {
        "episodes":   common_eps,
        "dist_mean":  dist_mat.mean(0),
        "dist_std":   dist_mat.std(0),
        "cos_mean":   cos_mat.mean(0),
        "cos_std":    cos_mat.std(0),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_row(ax_dist, ax_cos, data: dict,
             row_label: str,
             show_titles: bool,
             show_xlabel: bool):

    eps = data["episodes"]

    # Centroid distance panel
    ax_dist.fill_between(eps,
                         data["dist_mean"] - data["dist_std"],
                         data["dist_mean"] + data["dist_std"],
                         alpha=0.20, color="steelblue")
    ax_dist.plot(eps, data["dist_mean"], color="steelblue", lw=2.0,
                 label=r"Mean $\pm$ std")
    ax_dist.set_ylabel("L2 distance", fontsize=10)
    ax_dist.grid(alpha=0.25)
    ax_dist.legend(fontsize=9)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)

    # Cosine similarity panel
    ax_cos.fill_between(eps,
                        data["cos_mean"] - data["cos_std"],
                        data["cos_mean"] + data["cos_std"],
                        alpha=0.20, color="darkorange")
    ax_cos.plot(eps, data["cos_mean"], color="darkorange", lw=2.0,
                label=r"Mean $\pm$ std")
    ax_cos.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax_cos.set_ylabel("Cosine similarity", fontsize=10)
    ax_cos.grid(alpha=0.25)
    ax_cos.legend(fontsize=9)
    ax_cos.spines["top"].set_visible(False)
    ax_cos.spines["right"].set_visible(False)

    if show_titles:
        ax_dist.set_title("Centroid distance", fontsize=11)
        ax_cos.set_title("Directional alignment", fontsize=11)

    if show_xlabel:
        ax_dist.set_xlabel("Training episode", fontsize=10)
        ax_cos.set_xlabel("Training episode", fontsize=10)
    else:
        ax_dist.tick_params(labelbottom=False)
        ax_cos.tick_params(labelbottom=False)

    # Row label (a / b)
    ax_dist.text(-0.18, 0.5, row_label,
                 transform=ax_dist.transAxes,
                 fontsize=13, fontweight="bold",
                 va="center", ha="right")


def main():
    device = "cpu"

    print("Loading Census Income data...")
    census_data = compute_run_metrics(CENSUS_RUN, device)
    print("Loading Capture-24 data...")
    cap24_data  = compute_run_metrics(CAPTURE24_RUN, device)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6),
                             gridspec_kw={"hspace": 0.06, "wspace": 0.32})

    plot_row(axes[0, 0], axes[0, 1], census_data,
             row_label="a)",
             show_titles=True,
             show_xlabel=False)

    plot_row(axes[1, 0], axes[1, 1], cap24_data,
             row_label="b)",
             show_titles=False,
             show_xlabel=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved → {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
