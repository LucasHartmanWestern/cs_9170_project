"""
Normalized Figures 4, 5, 6 for Experiment 2.

Normalization removes per-seed variation due to different initial conditions
(different alpha-EO gaps, different initial L2 distances in PCA space):

  Fig 4 — Training curves:  r_norm_t = (r_t - r_0) / (1 - r_0)
           Shows return improvement above the per-seed starting point.

  Fig 5 — Centroid drift:
    L2:      L2_norm_t = L2_t / L2_0   (1.0 at start; <1 = closer to target)
    Cosine:  Raw similarity (naturally bounded; absolute value interpretable)

  Fig 6 — Generalizability curve (EO only):
           EO_reduction_t = (alpha_EO - beta_EO_t) / alpha_EO
           Shows fraction of per-seed fairness gap closed.  0=no improvement, 1=fully closed.
           Uses saved beta_ep*.pt models (avoids expensive re-training).

Output:
  figs/exp2_norm/fig4_training_curves_norm.png
  figs/exp2_norm/fig5_centroid_drift_norm.png
  figs/exp2_norm/fig6_gen_curve_norm.png
"""

import json
import sys
import re
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
    (CENSUS_RUN,    "Census Income",  "#1a4f8a", "#6699cc"),
    (CAPTURE24_RUN, "Capture-24",     "#27ae60", "#7ecfa0"),
]

SEEDS = ["0", "1", "42"]
SMOOTH_WINDOW = 40
OUT_DIR = Path("figs/exp2_norm")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_meta(seed_dir: Path) -> dict:
    with open(seed_dir / "meta.json") as f:
        return json.load(f)


def find_seed_dirs(run_dir: Path) -> list[Path]:
    return sorted(d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("seed_"))


def smooth(arr, w=SMOOTH_WINDOW):
    s = pd.Series(arr)
    return s.rolling(window=w, min_periods=1, center=True).mean().values


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


# ---------------------------------------------------------------------------
# Fig 4 — Normalized training curves
# ---------------------------------------------------------------------------

def compute_training_curves(run_dir: Path) -> list[dict]:
    """Returns list of {seed, episodes, r_norm} dicts."""
    results = []
    for sd in find_seed_dirs(run_dir):
        csv = sd / "metrics.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv, usecols=["episode", "meta.phase", "meta.episode_return"])
        ph1 = df[df["meta.phase"] == "phase1_class1"].copy()
        ph1 = ph1[~ph1["episode"].duplicated(keep="last")].sort_values("episode")
        r = ph1["meta.episode_return"].values
        if len(r) < 2:
            continue
        r0 = r[0]
        denom = 1.0 - r0
        r_norm = (r - r0) / denom if abs(denom) > 1e-6 else r - r0
        results.append({"seed": sd.name, "episodes": ph1["episode"].values, "r_norm": r_norm})
    return results


def plot_fig4(ax, curves, color_main, color_seed, label, panel_label):
    ax.text(-0.20, 0.97, panel_label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top", ha="left", clip_on=False)

    smoothed_all = []
    for c in curves:
        sm = smooth(c["r_norm"])
        smoothed_all.append((c["episodes"], sm))
        ax.plot(c["episodes"], sm, color=color_seed, linewidth=0.5, alpha=0.35, zorder=2)

    if not smoothed_all:
        return

    min_len = min(len(s) for _, s in smoothed_all)
    arr = np.array([s[:min_len] for _, s in smoothed_all])
    eps = smoothed_all[0][0][:min_len]

    ax.plot(eps, arr.mean(0), color=color_main, linewidth=2.2, zorder=5, label="Mean")
    ax.fill_between(eps, arr.min(0), arr.max(0),
                    color=color_main, alpha=0.20, zorder=4, label="Seed range")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)
    ax.set_xlabel("Episode", fontsize=14)
    ax.set_ylabel("Normalised return\n" + r"$(r_t - r_0)/(1 - r_0)$", fontsize=11)
    ax.set_title(label, fontsize=12)
    ax.tick_params(labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=11, loc="lower right")


# ---------------------------------------------------------------------------
# Fig 5 — Normalized centroid drift
# ---------------------------------------------------------------------------

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
    x_tr, _, x_te, y_tr, _, y_te = splits
    return ds, x_tr.cpu().numpy(), y_tr.cpu().numpy(), x_te, y_te


def compute_centroid_drift(run_dir: Path, device: str = "cpu") -> list[dict]:
    results = []
    for sd in find_seed_dirs(run_dir):
        meta = load_meta(sd)
        snap_dir = sd / "synthetic_snapshots"
        eps = snapshot_episodes_npz(snap_dir)
        if not eps:
            continue

        print(f"  Centroid drift: {sd.name}...")
        ds, X_tr, y_tr, _, _ = reconstruct_dataset(meta, device)
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

        if not ep_list:
            continue

        results.append({
            "seed": sd.name,
            "episodes": np.array(ep_list),
            "dist": np.array(dist_list),
            "cos": np.array(cos_list),
        })
    return results


def plot_fig5_row(axes, seed_data, color_main, color_seed, ds_label, row_idx):
    panel_labels = ["a)", "b)"] if row_idx == 0 else ["c)", "d)"]

    # Align to common episode grid
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

    dist_mat = align("dist")
    cos_mat  = align("cos")

    # Normalise L2 by per-seed initial value
    dist_norm = dist_mat / dist_mat[:, 0:1]

    for col, (mat, ax) in enumerate([(dist_norm, axes[0]), (cos_mat, axes[1])]):
        for row_s in mat:
            ax.plot(common_eps, row_s, color=color_seed, linewidth=0.6, alpha=0.35, zorder=2)

        mean = mat.mean(0)
        std  = mat.std(0)
        ax.plot(common_eps, mean, color=color_main, linewidth=2.0, zorder=5)
        ax.fill_between(common_eps, mean - std, mean + std,
                        color=color_main, alpha=0.18, zorder=4)

        # Reference lines
        if col == 0:
            ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        else:
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.4)

        ax.text(-0.18, 0.97, panel_labels[col], transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left", clip_on=False)
        ax.set_xlabel("Episode", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linewidth=0.4, alpha=0.4)

    axes[0].set_ylabel(f"{ds_label}\n" + r"$L2_t / L2_0$", fontsize=10)
    axes[1].set_ylabel("Cosine similarity", fontsize=10)


# ---------------------------------------------------------------------------
# Fig 6 — Normalized gen curve (using saved beta .pt checkpoints)
# ---------------------------------------------------------------------------

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


def eval_beta_pt(pt_path: Path, x_test, y_test, a_test,
                 input_size: int, device: str = "cpu") -> float:
    """Load saved beta state_dict, evaluate EO on test set."""
    from sklearn.metrics import roc_auc_score, f1_score
    sd = torch.load(pt_path, map_location=device, weights_only=True)
    output_size = sd["network.4.weight"].shape[0]
    model = FFNNModel(input_size, [32, 16], output_size).to(device)
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        logits = model(x_test.to(device))
        if output_size == 1:
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        else:
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    y_np = y_test.cpu().numpy()
    a_np = a_test.cpu().numpy()
    preds = (probs >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_np, probs)
    except Exception:
        auc = float("nan")
    f1w = f1_score(y_np, preds, average="weighted", zero_division=0)

    eo = float("nan")
    groups = np.unique(a_np)
    if len(groups) >= 2:
        tprs = [preds[(a_np == g) & (y_np == 1)].mean()
                for g in groups if ((a_np == g) & (y_np == 1)).sum() > 0]
        if len(tprs) == 2:
            eo = abs(tprs[0] - tprs[1])

    return {"eo": eo, "f1w": f1w, "auc": auc}


def compute_gen_curve(run_dir: Path, device: str = "cpu") -> list[dict]:
    """
    For each seed, evaluate saved beta .pt checkpoints at .npz snapshot intervals.
    Normalises EO by per-seed alpha_EO from final_test_metrics.csv.
    """
    # Get per-seed alpha_EO from final_test_metrics.csv
    alpha_eo_by_seed = {}
    final_csv = run_dir / "final_test_metrics.csv"
    if final_csv.exists():
        df_f = pd.read_csv(final_csv)
        if "seed" in df_f.columns and "alpha_eo_tpr_diff" in df_f.columns:
            for _, row in df_f.iterrows():
                alpha_eo_by_seed[int(row["seed"])] = float(row["alpha_eo_tpr_diff"])

    results = []
    for sd in find_seed_dirs(run_dir):
        meta = load_meta(sd)
        snap_dir = sd / "synthetic_snapshots"

        target_eps = snapshot_episodes_npz(snap_dir)
        if not target_eps:
            continue

        pt_eps = snapshot_episodes_pt(snap_dir)
        if not pt_eps:
            print(f"  [warn] No .pt checkpoints in {sd.name}")
            continue

        print(f"  Gen curve: {sd.name} ({len(target_eps)} eval points)...")
        ds, _, _, x_te, y_te = reconstruct_dataset(meta, device)
        a_test = ds.a_test

        seed_val = int(meta["seed"])
        alpha_eo = alpha_eo_by_seed.get(seed_val, float("nan"))
        pca_k    = int(meta.get("pca_components", 10))

        ep_list, eo_list, eo_norm_list, f1w_list, auc_list = [], [], [], [], []
        for tgt in target_eps:
            closest = min(pt_eps, key=lambda e: abs(e - tgt))
            if abs(closest - tgt) > 100:
                continue
            pt_path = snap_dir / f"beta_ep{closest:04d}_phase1_class1.pt"
            if not pt_path.exists():
                continue
            m = eval_beta_pt(pt_path, x_te, y_te, a_test, pca_k, device)
            ep_list.append(tgt)
            eo_list.append(m["eo"])
            f1w_list.append(m["f1w"])
            auc_list.append(m["auc"])
            if not np.isnan(alpha_eo) and alpha_eo > 0:
                eo_norm_list.append((alpha_eo - m["eo"]) / alpha_eo)
            else:
                eo_norm_list.append(float("nan"))

        results.append({
            "seed": sd.name,
            "alpha_eo": alpha_eo,
            "episodes": np.array(ep_list),
            "eo": np.array(eo_list),
            "eo_norm": np.array(eo_norm_list),
            "f1w": np.array(f1w_list),
            "auc": np.array(auc_list),
        })
    return results


def plot_fig6_row(axes, seed_data, color_main, color_seed, ds_label, row_idx):
    panel_labels = ["a)", "b)", "c)"] if row_idx == 0 else ["d)", "e)", "f)"]
    titles = [r"EO reduction (normalised)", "F1-Weighted (test)", "AUC (test)"] if row_idx == 0 else [None, None, None]
    fields = ["eo_norm", "f1w", "auc"]

    if not seed_data:
        return

    common_eps = seed_data[0]["episodes"]
    for s in seed_data[1:]:
        common_eps = np.intersect1d(common_eps, s["episodes"])

    for col, (field, ax) in enumerate(zip(fields, axes)):
        mat = []
        for s in seed_data:
            idx = np.isin(s["episodes"], common_eps)
            mat.append(s[field][idx])
        mat = np.array(mat)

        valid = ~np.all(np.isnan(mat), axis=0)
        if not valid.any():
            continue

        for row_s in mat:
            ax.plot(common_eps[valid], row_s[valid],
                    color=color_seed, linewidth=0.6, alpha=0.35, zorder=2)

        mean = np.nanmean(mat[:, valid], axis=0)
        std  = np.nanstd(mat[:, valid], axis=0)
        ax.plot(common_eps[valid], mean, color=color_main, linewidth=2.0,
                marker="o", markersize=3, zorder=5)
        ax.fill_between(common_eps[valid], mean - std, mean + std,
                        color=color_main, alpha=0.18, zorder=4)

        if field == "eo_norm":
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.axhline(1, color="green", linewidth=0.8, linestyle=":", alpha=0.5)

        ax.text(-0.18, 0.97, panel_labels[col], transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="top", ha="left", clip_on=False)
        if titles[col]:
            ax.set_title(titles[col], fontsize=10, pad=4)
        ax.set_xlabel("Episode", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linewidth=0.4, alpha=0.4)

    axes[0].set_ylabel(f"{ds_label}\n" + r"$(α\text{-EO} - β\text{-EO}_t)/α\text{-EO}$", fontsize=9)
    axes[1].set_ylabel("F1-weighted", fontsize=10)
    axes[2].set_ylabel("AUC", fontsize=10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cpu"

    # ── Fig 4 ─────────────────────────────────────────────────────────────────
    print("\n=== Fig 4: Normalized training curves ===")
    fig4, axes4 = plt.subplots(1, 2, figsize=(9.0, 3.75), sharey=False)

    for ax, (run_dir, label, color_main, color_seed) in zip(axes4, DATASETS):
        panel_label = "a)" if "Census" in label else "b)"
        curves = compute_training_curves(run_dir)
        print(f"  {label}: {len(curves)} seeds")
        plot_fig4(ax, curves, color_main, color_seed, label, panel_label)

    plt.tight_layout()
    out4 = OUT_DIR / "fig4_training_curves_norm.png"
    fig4.savefig(out4, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out4}")
    plt.close(fig4)

    # ── Fig 5 ─────────────────────────────────────────────────────────────────
    print("\n=== Fig 5: Normalized centroid drift ===")
    fig5, axes5 = plt.subplots(2, 2, figsize=(10, 7),
                               gridspec_kw={"hspace": 0.48, "wspace": 0.40})

    for row_idx, (run_dir, label, color_main, color_seed) in enumerate(DATASETS):
        print(f"  {label}...")
        seed_data = compute_centroid_drift(run_dir, device)
        plot_fig5_row(axes5[row_idx], seed_data, color_main, color_seed, label, row_idx)

    out5 = OUT_DIR / "fig5_centroid_drift_norm.png"
    fig5.savefig(out5, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out5}")
    plt.close(fig5)

    # ── Fig 6 ─────────────────────────────────────────────────────────────────
    print("\n=== Fig 6: Normalized gen curve ===")
    fig6, axes6 = plt.subplots(2, 3, figsize=(13, 7),
                               gridspec_kw={"hspace": 0.50, "wspace": 0.42})

    for row_idx, (run_dir, label, color_main, color_seed) in enumerate(DATASETS):
        print(f"  {label}...")
        seed_data = compute_gen_curve(run_dir, device)
        plot_fig6_row(axes6[row_idx], seed_data, color_main, color_seed, label, row_idx)

    out6 = OUT_DIR / "fig6_gen_curve_norm.png"
    fig6.savefig(out6, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out6}")
    plt.close(fig6)

    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
