"""
Val vs test EO tracking figure.

At each checkpoint episode (every ~150 episodes), evaluates the saved beta
model on both the validation set (which the reward signal is computed from)
and the held-out test set.

If val EO and test EO track together, the policy is not overfitting to the
validation partition — the learned strategy generalizes to unseen data.

Two panels: Census Income (left), Capture-24 (right).
Each panel: val EO (dashed) and test EO (solid), mean +/- seed range across
3 seeds.

Output: figs/exp2_norm/fig_val_vs_test.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Run directories
# ---------------------------------------------------------------------------

CENSUS_RUN = Path(
    "/storage_1/epigou_storage/FORGE/training_runs_k10/"
    "SPECcensus_k10_r04_e30_EP5000_PCA10_REWwgl_minID0_majID1_"
    "TRJ2000_REAL3000_GG202605121151_249e76f9"
)
CAPTURE24_RUN = Path(
    "/storage_1/epigou_storage/FORGE/aulavik_runs/capture_24/k5/"
    "SPECcapture24_k5_gpu0_EP5000_PCA15_REWwgl_minID1_majID0_"
    "TRJ1000_REAL4000_GG202605041119_779cf9c5"
)

DATASETS = [
    (CENSUS_RUN,    "Census Income",  "#1a4f8a"),
    (CAPTURE24_RUN, "Capture-24",     "#27ae60"),
]

OUT_DIR = Path("figs/exp2_norm")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_meta(seed_dir: Path) -> dict:
    with open(seed_dir / "meta.json") as f:
        return json.load(f)


def find_seed_dirs(run_dir: Path) -> list[Path]:
    return sorted(d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("seed_"))


def snapshot_episodes_npz(snap_dir: Path) -> list[int]:
    eps = []
    for f in snap_dir.glob("synthetic_ep*_phase1_class1.npz"):
        try:
            eps.append(int(f.stem.split("_ep")[1].split("_")[0]))
        except Exception:
            pass
    return sorted(eps)


def snapshot_episodes_pt(snap_dir: Path) -> list[int]:
    eps = []
    for f in snap_dir.glob("beta_ep*_phase1_class1.pt"):
        try:
            eps.append(int(f.stem.split("_ep")[1].split("_")[0]))
        except Exception:
            pass
    return sorted(eps)


class FFNNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def compute_eo(model, x, y, a, device):
    """EO (TPR gap) for a loaded model on a given split."""
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        output_size = logits.shape[-1] if logits.ndim > 1 else 1
        if output_size == 1:
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        else:
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    y_np = y.cpu().numpy()
    a_np = a.cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    groups = np.unique(a_np)
    tprs = [preds[(a_np == g) & (y_np == 1)].mean()
            for g in groups if ((a_np == g) & (y_np == 1)).sum() > 0]
    if len(tprs) == 2:
        return abs(tprs[0] - tprs[1])
    return float("nan")


def reconstruct_splits(meta: dict, device: str):
    """Returns (ds, x_val, y_val, a_val, x_test, y_test, a_test)."""
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
    if meta.get("dp_protected_col") is not None:
        kwargs["dp_protected_col"] = meta["dp_protected_col"]
    splits = ds.get_data_splits(**kwargs)
    _, x_val, x_test, _, y_val, y_test = splits
    a_val  = ds.a_val
    a_test = ds.a_test
    return x_val, y_val, a_val, x_test, y_test, a_test


# ---------------------------------------------------------------------------
# Compute val and test EO across checkpoints for all seeds
# ---------------------------------------------------------------------------

def compute_val_test_curves(run_dir: Path, device: str = "cpu") -> list[dict]:
    results = []
    for sd in find_seed_dirs(run_dir):
        meta  = load_meta(sd)
        snap_dir = sd / "synthetic_snapshots"

        target_eps = snapshot_episodes_npz(snap_dir)  # every ~150 eps
        pt_eps     = snapshot_episodes_pt(snap_dir)
        if not target_eps or not pt_eps:
            continue

        print(f"  {sd.name}: reconstructing dataset...")
        x_val, y_val, a_val, x_test, y_test, a_test = reconstruct_splits(meta, device)
        pca_k = int(meta.get("pca_components", 10))

        ep_list, val_eo_list, test_eo_list = [], [], []
        for tgt in target_eps:
            closest = min(pt_eps, key=lambda e: abs(e - tgt))
            if abs(closest - tgt) > 100:
                continue
            pt_path = snap_dir / f"beta_ep{closest:04d}_phase1_class1.pt"
            if not pt_path.exists():
                continue

            sd_weights = torch.load(pt_path, map_location=device, weights_only=True)
            out_size = sd_weights["network.4.weight"].shape[0]
            model = FFNNModel(pca_k, [32, 16], out_size).to(device)
            model.load_state_dict(sd_weights)

            val_eo  = compute_eo(model, x_val,  y_val,  a_val,  device)
            test_eo = compute_eo(model, x_test, y_test, a_test, device)

            ep_list.append(tgt)
            val_eo_list.append(val_eo)
            test_eo_list.append(test_eo)

        print(f"    {len(ep_list)} points | "
              f"val_EO: {np.nanmean(val_eo_list):.3f} | "
              f"test_EO: {np.nanmean(test_eo_list):.3f}")
        results.append({
            "seed":     sd.name,
            "episodes": np.array(ep_list),
            "val_eo":   np.array(val_eo_list),
            "test_eo":  np.array(test_eo_list),
        })
    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

COL_TEST = "#1a4f8a"   # dark blue — test set (held-out)
COL_VAL  = "#999999"   # neutral gray — validation set (optimised against)


def plot_panel(ax, seed_data, label, panel_label):
    if not seed_data:
        return

    common_eps = seed_data[0]["episodes"]
    for s in seed_data[1:]:
        common_eps = np.intersect1d(common_eps, s["episodes"])

    def align(field):
        out = []
        for s in seed_data:
            idx = np.isin(s["episodes"], common_eps)
            out.append(s[field][idx])
        return np.array(out)

    val_mat  = align("val_eo")
    test_mat = align("test_eo")

    val_mean  = np.nanmean(val_mat,  axis=0)
    test_mean = np.nanmean(test_mat, axis=0)
    val_min   = np.nanmin(val_mat,   axis=0)
    val_max   = np.nanmax(val_mat,   axis=0)
    test_min  = np.nanmin(test_mat,  axis=0)
    test_max  = np.nanmax(test_mat,  axis=0)

    # Val EO — gray dashed, drawn first so test sits on top
    ax.fill_between(common_eps, val_min, val_max,
                    alpha=0.12, color=COL_VAL, zorder=2)
    ax.plot(common_eps, val_mean, color=COL_VAL, linewidth=1.8,
            linestyle="--", zorder=3, label="Validation EO")

    # Test EO — blue solid
    ax.fill_between(common_eps, test_min, test_max,
                    alpha=0.18, color=COL_TEST, zorder=4)
    ax.plot(common_eps, test_mean, color=COL_TEST, linewidth=2.2,
            linestyle="-", zorder=5, label="Test EO (held-out)")

    ax.text(-0.20, 0.97, panel_label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left", clip_on=False)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel(r"$\beta$-EO $\downarrow$", fontsize=12)
    ax.set_ylim(bottom=0, top=0.5)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.tick_params(labelsize=11)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    panel_labels = ["a)", "b)"]

    for ax, (run_dir, label, _color), panel_label in zip(axes, DATASETS, panel_labels):
        print(f"\n{label}...")
        seed_data = compute_val_test_curves(run_dir, device)
        plot_panel(ax, seed_data, label, panel_label)

    plt.tight_layout()
    out = OUT_DIR / "fig_val_vs_test.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
