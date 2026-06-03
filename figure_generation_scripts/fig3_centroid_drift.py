"""
fig3_centroid_drift.py — Two-panel centroid drift figure for Experiment 2.

Census:     5 seeds (0, 1, 2, 3, 42) from experiment3/census_forge/
Capture-24: 5 folds (0–4) from rand_0010 capture24 runs, each treated as a seed
            (seed_42 subdir from each fold dir).

Output: paper/figures/fig_centroid_drift_paper.png

Run from project root:  python paper_figures/fig3_centroid_drift.py [--device cpu]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from plot_centroid_drift import compute_seed_metrics, load_meta

# ── Run directories ───────────────────────────────────────────────────────────

CENSUS_RUN_DIR = Path(
    "/storage_1/epigou_storage/FORGE/experiment3/census_forge"
)

# rand_0010 fold dirs — seed_42 subdir in each is treated as one "seed"
_EXP1_C24 = Path("/storage_1/epigou_storage/FORGE/experiment1/capture24_random")
CAPTURE24_FOLD_DIRS = sorted(
    _EXP1_C24.glob(
        "SPECrand_0010_learning_rate0p00024106501268706255_"
        "lr0p00013646506337961615_delta_scale0p10965398215380505_"
        "optimizeradam_optimizeradamw_fold*_EP5000_PCA15_*"
    )
)

OUT_PATH = Path("paper/figures/fig_centroid_drift_paper.png")


# ── Computation helpers ───────────────────────────────────────────────────────

def collect_metrics(seed_dirs: list[Path], device: str, label: str):
    """Run compute_seed_metrics on each seed dir and return aligned arrays."""
    all_eps, all_dist, all_cos = [], [], []
    for sd in seed_dirs:
        meta = load_meta(sd)
        print(f"  [{label}] {sd.parent.name}/{sd.name} "
              f"(dataset={meta['dataset_name']}, seed={meta['seed']})")
        result = compute_seed_metrics(sd, meta, device)
        if result is None:
            print(f"    → skipped (no snapshots)")
            continue
        all_eps.append(result["episodes"])
        all_dist.append(result["dist"])
        all_cos.append(result["cos"])

    if not all_eps:
        return None

    common_eps = all_eps[0]
    for ep_arr in all_eps[1:]:
        common_eps = np.intersect1d(common_eps, ep_arr)

    def align(arrs, eps_list):
        return np.array([a[np.isin(e, common_eps)] for a, e in zip(arrs, eps_list)])

    dist_mat = align(all_dist, all_eps)
    cos_mat  = align(all_cos,  all_eps)

    return dict(
        episodes  = common_eps,
        dist_mean = dist_mat.mean(axis=0),
        dist_std  = dist_mat.std(axis=0),
        cos_mean  = cos_mat.mean(axis=0),
        cos_std   = cos_mat.std(axis=0),
        n         = len(dist_mat),
    )


def draw_panel(axes, metrics: dict, panel_label: str):
    """Draw centroid distance (left) and cosine similarity (right) into axes pair."""
    ax1, ax2 = axes
    eps = metrics["episodes"]
    n   = metrics["n"]

    ax1.fill_between(eps,
                     metrics["dist_mean"] - metrics["dist_std"],
                     metrics["dist_mean"] + metrics["dist_std"],
                     alpha=0.20, color="steelblue")
    ax1.plot(eps, metrics["dist_mean"], color="steelblue", lw=2.0,
             label=f"Mean $\\pm$ std ({n})")
    ax1.set_xlabel("Training episode")
    ax1.set_ylabel("L2 distance")
    ax1.set_title("Centroid distance")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)
    ax1.text(-0.12, 0.5, panel_label, transform=ax1.transAxes,
             fontsize=13, fontweight="bold", va="center", ha="right")

    ax2.fill_between(eps,
                     metrics["cos_mean"] - metrics["cos_std"],
                     metrics["cos_mean"] + metrics["cos_std"],
                     alpha=0.20, color="darkorange")
    ax2.plot(eps, metrics["cos_mean"], color="darkorange", lw=2.0,
             label=f"Mean $\\pm$ std ({n})")
    ax2.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax2.set_xlabel("Training episode")
    ax2.set_ylabel("Cosine similarity")
    ax2.set_title("Directional alignment")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    assert CENSUS_RUN_DIR.exists(), f"Census run dir not found: {CENSUS_RUN_DIR}"
    assert CAPTURE24_FOLD_DIRS, \
        f"No rand_0010 fold dirs found under {_RSRCH}"
    print(f"Capture-24 fold dirs ({len(CAPTURE24_FOLD_DIRS)}): "
          + ", ".join(d.name.split("_fold")[1].split("_EP")[0] for d in CAPTURE24_FOLD_DIRS))

    # Census: seed_* subdirs of the run dir
    census_seed_dirs = sorted(
        d for d in CENSUS_RUN_DIR.iterdir()
        if d.is_dir() and d.name.startswith("seed_")
    )
    print(f"\nCensus seeds: {[s.name for s in census_seed_dirs]}")
    print("Computing census centroid drift…")
    census_metrics = collect_metrics(census_seed_dirs, args.device, "Census")

    # Capture-24: seed_42 subdir from each fold dir acts as one seed
    c24_seed_dirs = [d / "seed_42" for d in CAPTURE24_FOLD_DIRS]
    missing = [p for p in c24_seed_dirs if not p.exists()]
    if missing:
        print(f"[WARN] Missing seed_42 in: {[str(m) for m in missing]}")
        c24_seed_dirs = [p for p in c24_seed_dirs if p.exists()]
    print(f"\nCapture-24 folds as seeds: {len(c24_seed_dirs)}")
    print("Computing Capture-24 centroid drift…")
    c24_metrics = collect_metrics(c24_seed_dirs, args.device, "Capture-24")

    # Build figure: 2 rows (one per dataset), 2 cols (dist | cos)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    fig.subplots_adjust(hspace=0.38, wspace=0.32)

    if census_metrics:
        draw_panel(axes[0], census_metrics, "a)")
    if c24_metrics:
        draw_panel(axes[1], c24_metrics, "b)")

    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved → {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
