#!/usr/bin/env python3
"""
eval_raw_space.py  --  Post-hoc raw-feature-space evaluation for FORGE.

For each seed in a completed training run:
  1. Re-load the dataset in raw feature space (use_pca=False) to get raw real train/test data.
  2. Re-fit PCA on the same training data (use_pca=True) to recover pca_transform.
  3. Load the best synthetic dataset from best_synthetic_phase1_class1.npz (PCA space).
  4. Apply pca_transform.inverse_transform() to map synthetic data to raw feature space.
  5. Train a fresh beta classifier on raw D_aug = raw real train + inverse-transformed synthetic.
  6. Evaluate beta on the raw test set; report EO gap, DP gap, AUC, F1w.
  7. Save per-seed results to <seed_dir>/raw_space_eval/metrics.json and a run-level
     summary to <run_dir>/raw_space_eval_summary.json.

Usage:
    python eval_raw_space.py <run_dir> [--device cpu]

The RL training itself is unchanged. This script only performs the final evaluation step
in raw feature space, as an alternative to the default PCA-space evaluation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))
from dataset import Dataset
from agents.ffnn_agent2 import FFNNAgent


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _make_dataset(meta: dict, use_pca: bool, device: str) -> Dataset:
    return Dataset(
        dataset_name=meta["dataset_name"],
        multiclass=meta.get("multiclass", False),
        minority_id=meta.get("minority_id"),
        majority_id=meta.get("majority_id"),
        third_id=meta.get("third_id"),
        pca_components=meta.get("pca_components", 10),
        seed=meta["seed"],
        device=device,
        use_pca=use_pca,
    )


def _get_splits(ds: Dataset, meta: dict):
    # Build kwargs from meta; get_data_splits filters irrelevant ones per dataset.
    kwargs = dict(
        train_size=meta.get("REAL_DATA_SIZE"),
        bias_pct=meta.get("BIAS_PCT"),
        da_pct=meta.get("DA_PCT"),
        pca_components=meta.get("pca_components", 10),
        drop_protected=False,
        win_seconds=meta.get("win_seconds"),
        step_seconds=meta.get("step_seconds"),
    )
    # Drop None-valued optional params so they don't override dataset defaults.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return ds.get_data_splits(**kwargs)


def _compute_metrics(
    model: FFNNAgent,
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
) -> dict:
    model.model.eval()
    with torch.no_grad():
        logits = model.model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    y_np = y.cpu().numpy()
    a_np = a.cpu().numpy()
    preds = (probs >= 0.5).astype(int)

    try:
        auc = float(roc_auc_score(y_np, probs))
    except Exception:
        auc = float("nan")

    f1w = float(f1_score(y_np, preds, average="weighted", zero_division=0))

    def tpr(mask):
        pos = y_np[mask] == 1
        return float(preds[mask][pos].mean()) if pos.sum() > 0 else float("nan")

    def pos_rate(mask):
        return float(preds[mask].mean()) if mask.sum() > 0 else float("nan")

    groups = np.unique(a_np)
    tprs = {int(g): tpr(a_np == g) for g in groups}
    rates = {int(g): pos_rate(a_np == g) for g in groups}

    valid_tprs = [v for v in tprs.values() if not np.isnan(v)]
    valid_rates = [v for v in rates.values() if not np.isnan(v)]

    eo_gap = float(max(valid_tprs) - min(valid_tprs)) if len(valid_tprs) >= 2 else float("nan")
    dp_gap = float(max(valid_rates) - min(valid_rates)) if len(valid_rates) >= 2 else float("nan")

    return {
        "eo_gap": eo_gap,
        "dp_gap": dp_gap,
        "auc": auc,
        "f1w": f1w,
        "tpr_by_group": tprs,
        "pos_rate_by_group": rates,
    }


def eval_seed(seed_dir: Path, device: str) -> dict | None:
    meta = _load_json(seed_dir / "meta.json")
    ffnn_meta = _load_json(seed_dir / "ffnn_meta.json")

    syn_path = seed_dir / "best_synthetic_phase1_class1.npz"
    if not syn_path.exists():
        print(f"  [SKIP] {syn_path.name} not found")
        return None

    syn = np.load(syn_path)
    syn_x_pca = syn["x"]   # (T, d) — PCA space
    syn_y = syn["y"]       # (T,)
    print(f"  Synthetic shape: {syn_x_pca.shape}  (PCA d={syn_x_pca.shape[1]})")

    # Step 1: recover pca_transform by re-fitting PCA on same training data
    ds_pca = _make_dataset(meta, use_pca=True, device=device)
    _get_splits(ds_pca, meta)
    pca_transform = getattr(ds_pca, "pca_transform", None)
    if pca_transform is None:
        print("  [SKIP] pca_transform not available (run may have use_pca=False)")
        return None

    # Step 2: load dataset in raw feature space
    ds_raw = _make_dataset(meta, use_pca=False, device=device)
    splits = _get_splits(ds_raw, meta)
    x_train_raw, _, x_test_raw, y_train, _, y_test = splits[:6]
    a_test = ds_raw.a_test

    raw_dim = x_train_raw.shape[1]
    print(f"  Raw feature dim: {raw_dim}")

    # Step 3: inverse-transform synthetic data to raw feature space
    syn_x_raw_np = pca_transform.inverse_transform(syn_x_pca)   # (T, raw_dim)
    syn_x_raw = torch.tensor(syn_x_raw_np, dtype=torch.float32, device=device)
    syn_y_tensor = torch.tensor(syn_y, dtype=torch.long, device=device)

    # Step 4: augmented training set in raw space
    x_aug = torch.cat([x_train_raw, syn_x_raw], dim=0)
    y_aug = torch.cat([y_train, syn_y_tensor], dim=0)

    # Step 5: train fresh beta in raw space with the same FFNN config
    beta_cfg = {
        "input_size": raw_dim,
        "hidden_sizes": ffnn_meta.get("hidden_sizes", [32, 16]),
        "output_size": ffnn_meta.get("output_size", 2),
        "learning_rate": ffnn_meta.get("learning_rate", 1e-3),
        "batch_size": ffnn_meta.get("batch_size", 64),
        "epochs": ffnn_meta.get("epochs", 10),
        "optimizer": ffnn_meta.get("optimizer", "adam"),
        "type": "classification",
        "classes": [0, 1],
        "device": device,
        "seed": meta["seed"],
    }
    beta = FFNNAgent(**beta_cfg)
    loader = DataLoader(
        TensorDataset(x_aug, y_aug),
        batch_size=beta_cfg["batch_size"],
        shuffle=True,
    )
    beta.train(loader)

    # Step 6: evaluate on raw test set
    metrics = _compute_metrics(beta, x_test_raw, y_test, a_test)
    print(
        f"  EO gap: {metrics['eo_gap']:.4f}  "
        f"AUC: {metrics['auc']:.4f}  "
        f"F1w: {metrics['f1w']:.4f}"
    )

    out_dir = seed_dir / "raw_space_eval"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run directory containing seed_* subdirs")
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    args = parser.parse_args()

    run_dir = args.run_dir
    seed_dirs = sorted(d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("seed_"))

    if not seed_dirs:
        print(f"No seed_* directories found in {run_dir}")
        sys.exit(1)

    print(f"Run: {run_dir.name}  ({len(seed_dirs)} seed(s))\n")

    all_metrics = {}
    for sd in seed_dirs:
        print(f"[{sd.name}]")
        m = eval_seed(sd, args.device)
        if m is not None:
            all_metrics[sd.name] = m

    if not all_metrics:
        print("No results produced.")
        return

    eo_vals = [m["eo_gap"] for m in all_metrics.values() if not np.isnan(m["eo_gap"])]
    auc_vals = [m["auc"] for m in all_metrics.values() if not np.isnan(m["auc"])]
    f1_vals = [m["f1w"] for m in all_metrics.values() if not np.isnan(m["f1w"])]

    summary = {
        "eo_gap_mean": float(np.mean(eo_vals)) if eo_vals else float("nan"),
        "eo_gap_std": float(np.std(eo_vals)) if eo_vals else float("nan"),
        "auc_mean": float(np.mean(auc_vals)) if auc_vals else float("nan"),
        "f1w_mean": float(np.mean(f1_vals)) if f1_vals else float("nan"),
        "n_seeds": len(all_metrics),
        "per_seed": all_metrics,
    }

    print(f"\n--- Summary ({len(all_metrics)} seeds) ---")
    print(f"EO gap : {summary['eo_gap_mean']:.4f} ± {summary['eo_gap_std']:.4f}")
    print(f"AUC    : {summary['auc_mean']:.4f}")
    print(f"F1w    : {summary['f1w_mean']:.4f}")

    out_path = run_dir / "raw_space_eval_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
