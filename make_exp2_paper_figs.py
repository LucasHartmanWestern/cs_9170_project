"""
Combine per-dataset analysis figures into two-panel paper figures for Experiment 2.

Produces:
  paper/figures/fig_centroid_drift_paper.png  — 2-row: census (top) capture24 (bottom)
  paper/figures/fig_gen_curve_paper.png       — 2-row: census (top) capture24 (bottom)

Run from project root:  python make_exp2_paper_figs.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

CENSUS_ANALYSIS = Path(
    "/storage_1/epigou_storage/FORGE/aulavik_runs/census/k5/"
    "SPECcensus_k5_gpu1_EP5000_PCA10_REWwgl_minID0_majID1_"
    "TRJ2000_REAL3000_GG202604251803_9af13c63/analysis"
)
CAPTURE24_ANALYSIS = Path(
    "/storage_1/epigou_storage/FORGE/training_runs/"
    "SPECtest_fold0_EP5000_PCA15_REWwgl_minID1_majID0_"
    "TRJ1000_REAL4000_GG202605281522_4d35ee6c/analysis"
)
OUT_DIR = Path("paper/figures")


def stack_two(img_top_path: Path, img_bot_path: Path, out_path: Path,
              label_top: str, label_bot: str):
    top = mpimg.imread(img_top_path)
    bot = mpimg.imread(img_bot_path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))
    fig.subplots_adjust(hspace=0.08, left=0.07)

    for ax, img, label in zip(axes, [top, bot], [label_top, label_bot]):
        ax.imshow(img)
        ax.axis("off")
        ax.text(-0.04, 0.5, label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="center", ha="right", rotation=0)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stack_two(
        CENSUS_ANALYSIS / "fig_centroid_drift.png",
        CAPTURE24_ANALYSIS / "fig_centroid_drift.png",
        OUT_DIR / "fig_centroid_drift_paper.png",
        label_top="a)",
        label_bot="b)",
    )

    stack_two(
        CENSUS_ANALYSIS / "fig_gen_curve.png",
        CAPTURE24_ANALYSIS / "fig_gen_curve.png",
        OUT_DIR / "fig_gen_curve_paper.png",
        label_top="a)",
        label_bot="b)",
    )


if __name__ == "__main__":
    main()
