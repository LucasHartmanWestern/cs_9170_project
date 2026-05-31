"""
Retrain with FORGE's best synthetic data using a 20-epoch classifier,
evaluated via TestSuite (same metrics as normal FORGE runs).

Usage:
    python eval_ep20_classifier.py <run_dir> [--device cuda:0] [--epochs 20]

Example:
    python eval_ep20_classifier.py \\
        /storage_1/epigou_storage/FORGE/training_runs/SPECtest_fold0_... \\
        --device cuda:0
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from dataset import Dataset
from agents.ffnn_agent2 import FFNNAgent
from test_suite import TestSuite


def _nearest_snapshot(seed_dir: Path, best_ep: int) -> Path:
    snaps = sorted(seed_dir.glob("synthetic_snapshots/synthetic_ep*_phase1_class1.npz"))
    if not snaps:
        raise FileNotFoundError(f"No synthetic snapshots in {seed_dir}")
    def ep_num(p):
        return int(p.stem.split("_ep")[1].split("_")[0])
    return min(snaps, key=lambda p: abs(ep_num(p) - best_ep))


def _make_ffnn(input_dim, meta_cfg, epochs, device):
    return FFNNAgent(
        input_dim,
        hidden_sizes=meta_cfg.get("hidden_sizes", [32, 16]),
        output_size=2,
        learning_rate=meta_cfg.get("learning_rate", 0.001),
        batch_size=meta_cfg.get("batch_size", 64),
        epochs=epochs,
        optimizer=meta_cfg.get("optimizer", "adam"),
        type="classification",
        classes=[0, 1],
        device=device,
    )


def eval_run(run_dir: Path, device: str, epochs: int = 20):
    seed_dir = run_dir / "seed_42"

    with open(run_dir / "meta.json") as f:
        meta = json.load(f).get("config", {})

    with open(seed_dir / "best_beta_meta_phase1_class1.json") as f:
        best_ep = json.load(f)["episode"]

    snap_path = _nearest_snapshot(seed_dir, best_ep)
    snap_ep = int(snap_path.stem.split("_ep")[1].split("_")[0])
    print(f"Best ep: {best_ep}  |  Snapshot: ep {snap_ep}  ({snap_path.name})")

    # Synthetic data (PCA space)
    snap = np.load(snap_path)
    x_syn = torch.tensor(snap["x"], dtype=torch.float32)
    y_syn = torch.tensor(snap["y"], dtype=torch.long)

    # Real kfold data — same call as training.py
    dp_col   = meta.get("dp_protected_col")
    ds = Dataset(
        meta["dataset_name"],
        multiclass=meta.get("multiclass", False),
        minority_id=meta["minority_id"],
        majority_id=meta["majority_id"],
        third_id=meta.get("third_id"),
        pca_components=meta["pca_components"],
        seed=42,
        use_pca=True,
    )
    split_kwargs = dict(
        train_size=meta.get("REAL_DATA_SIZE") or meta.get("total_data_size"),
        bias_pct=meta.get("BIAS_PCT") or meta.get("bias_pct"),
        da_pct=meta.get("DA_PCT") or meta.get("da_pct"),
        pca_components=meta["pca_components"],
        fold_idx=meta["fold_idx"],
        n_folds=meta["n_folds"],
        fold_rng_seed=meta.get("fold_rng_seed"),
        win_seconds=meta.get("win_seconds", 5.0),
        step_seconds=meta.get("step_seconds", 2.5),
    )
    if dp_col:
        split_kwargs["dp_protected_col"] = dp_col
    x_tr, x_val, x_te, y_tr, y_val, y_te = ds.get_data_splits(**split_kwargs)
    a_te = torch.as_tensor(ds.a_test, dtype=torch.long)

    # Convert to tensors
    x_tr = torch.as_tensor(x_tr, dtype=torch.float32)
    y_tr = torch.as_tensor(y_tr, dtype=torch.long)
    x_te = torch.as_tensor(x_te, dtype=torch.float32)
    y_te = torch.as_tensor(y_te, dtype=torch.long)

    print(f"Real train: {x_tr.shape[0]}  |  Synthetic: {x_syn.shape[0]}")

    # Train alpha on real data only (10 ep, same as FORGE)
    ffnn_cfg = {"hidden_sizes": [32, 16], "learning_rate": 0.001,
                "batch_size": 64, "optimizer": "adam"}
    alpha = _make_ffnn(x_tr.shape[1], ffnn_cfg, epochs=10, device=device)
    alpha.train(DataLoader(TensorDataset(x_tr, y_tr), batch_size=64, shuffle=True))

    # Train beta on real + synthetic (20 ep)
    x_hybrid = torch.cat([x_tr, x_syn], dim=0)
    y_hybrid = torch.cat([y_tr, y_syn], dim=0)
    print(f"Hybrid train: {x_hybrid.shape[0]}  |  Training beta with {epochs} epochs...")
    beta = _make_ffnn(x_hybrid.shape[1], ffnn_cfg, epochs=epochs, device=device)
    beta.train(DataLoader(TensorDataset(x_hybrid, y_hybrid), batch_size=64, shuffle=True))

    # Evaluate using TestSuite — write to a temp dir so we don't touch original outputs
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        suite = TestSuite(
            seed_dir=tmp_path,
            experiment_dir=tmp_path,
            seed=42,
            run_id=f"ep{epochs}_retrain",
        )
        suite.log_final_test(
            alpha_model=alpha,
            beta_model=beta,
            x_test=x_te.to(device),
            y_test=y_te.to(device),
            a_test=a_te.to(device),
            prefer_best_beta=False,
            run_alpha_raw_original=False,
            run_alpha_plus_real=False,
            run_alpha_plus_ctgan=False,
            run_ctabgan=False,
        )
        # Read back the CSV TestSuite wrote
        import pandas as pd
        row = pd.read_csv(tmp_path / "final_test_metrics.csv").iloc[0]

    print(f"\n=== Fold {meta['fold_idx']} — {epochs}-epoch beta on FORGE synthetic ===")
    print(f"  α-EO  = {row['alpha_eo_tpr_diff']:.4f}")
    print(f"  β-EO  = {row['beta_eo_tpr_diff']:.4f}   (vs 10-ep β-EO from run)")
    print(f"  β-F1w = {row['beta_f1_weighted']:.4f}")
    print(f"  β-AUC = {row['beta_roc_auc']:.4f}")

    return row['beta_eo_tpr_diff']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    results = {}
    for run_dir in args.run_dirs:
        eo = eval_run(run_dir, args.device, args.epochs)
        results[run_dir.name] = eo

    if len(results) > 1:
        import statistics
        mean_eo = statistics.mean(results.values())
        print(f"\n=== Mean β-EO across {len(results)} folds: {mean_eo:.4f} ===")
