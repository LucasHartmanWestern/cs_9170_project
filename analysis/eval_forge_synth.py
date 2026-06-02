"""
eval_forge_synth.py — Evaluate FORGE's best synthetic dataset with a fresh 20-epoch FFNN.

Takes a completed FORGE run directory (kfold), loads the best synthetic snapshot,
combines with real training data, trains a fresh FFNN with the specified config,
and evaluates on val + test. Used to isolate the contribution of synthetic data
quality from the RL-optimised checkpoint.

Usage:
    python analysis/eval_forge_synth.py <forge_run_dir> [--lr LR] [--epochs N] [--optimizer adam] [--device cpu]

Output:
    Prints a row matching final_test_metrics.csv format, plus saves
    <forge_run_dir>/seed_42/synth_eval_metrics.json
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import Dataset
from agents.ffnn_agent2 import FFNNAgent
import reward_helpers as rh


def eo_tpr_diff(model, x, y, a, device, minority_id):
    model.model.eval()
    with torch.no_grad():
        logits = model.model(torch.tensor(x, dtype=torch.float32).to(device))
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    tpr = {}
    for g in np.unique(a):
        mask = (a == g) & (y == 1)
        tpr[g] = preds[mask].mean() if mask.sum() > 0 else 0.0
    groups = sorted(tpr.keys())
    if len(groups) < 2:
        return 0.0
    return abs(tpr[groups[0]] - tpr[groups[1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--lr",        type=float, default=0.00024106501268706255)
    ap.add_argument("--epochs",    type=int,   default=20)
    ap.add_argument("--optimizer", type=str,   default="adam")
    ap.add_argument("--device",    type=str,   default="cpu")
    args = ap.parse_args()

    seed_dir = args.run_dir / "seed_42"
    meta = json.load(open(seed_dir / "meta.json"))
    device = torch.device(args.device)

    # Load dataset split for this fold
    ds = Dataset(
        meta["dataset_name"],
        meta.get("multiclass", False),
        minority_id=meta.get("minority_id"),
        majority_id=meta.get("majority_id"),
        third_id=meta.get("third_id"),
        pca_components=meta["pca_components"],
        seed=42,
        device=device,
        use_pca=True,
    )
    x_train, x_val, x_test, y_train, y_val, y_test = ds.get_data_splits(
        da_pct=meta.get("DA_PCT") or meta.get("da_pct"),
        real_data_size=meta["REAL_DATA_SIZE"],
        n_folds=meta.get("n_folds", 5),
        fold_idx=meta.get("fold_idx", 0),
        fold_rng_seed=meta.get("fold_rng_seed", 6),
        kfold_val_frac=meta.get("kfold_val_frac", 0.4),
        pca_components=meta["pca_components"],
    )
    # Group labels stored as side-effects on ds
    a_train = ds.a_train.cpu().numpy()
    a_val   = ds.a_val.cpu().numpy()
    a_test  = ds.a_test.cpu().numpy()
    x_train = x_train.cpu().numpy()
    x_val   = x_val.cpu().numpy()
    x_test  = x_test.cpu().numpy()
    y_train = y_train.cpu().numpy()
    y_val   = y_val.cpu().numpy()
    y_test  = y_test.cpu().numpy()

    # Load best synthetic snapshot
    ckpt = json.load(open(seed_dir / "best_beta_meta_phase1_class1.json"))
    best_ep = ckpt["episode"]
    snap_path = seed_dir / "synthetic_snapshots" / f"synthetic_ep{best_ep:04d}_phase1_class1.npz"
    if not snap_path.exists():
        # Find nearest available snapshot
        snaps = sorted((seed_dir / "synthetic_snapshots").glob("synthetic_ep*_phase1_class1.npz"))
        snap_path = min(snaps, key=lambda p: abs(int(p.stem.split("_ep")[1].split("_")[0]) - best_ep))
        print(f"  [warn] ep{best_ep} snapshot missing, using {snap_path.name}")
    synth = np.load(snap_path)
    x_syn = synth["x"].astype(np.float32)
    y_syn = synth["y"].astype(np.float32)

    # Combine real train + synthetic
    x_aug = np.concatenate([x_train, x_syn], axis=0)
    y_aug = np.concatenate([y_train, y_syn], axis=0)

    # Train fresh FFNN
    beta = FFNNAgent(
        input_size=x_aug.shape[1],
        hidden_sizes=[32, 16],
        classes=[0, 1],
        type="classification",
        learning_rate=args.lr,
        batch_size=64,
        epochs=args.epochs,
        device=device,
        seed=42,
        optimizer=args.optimizer,
    )
    ds_aug = TensorDataset(
        torch.tensor(x_aug, dtype=torch.float32).to(device),
        torch.tensor(y_aug, dtype=torch.long).to(device),
    )
    loader = DataLoader(ds_aug, batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(42))
    for _ in range(args.epochs):
        beta.train(loader)

    # Evaluate
    def get_metrics(x, y, a, label):
        beta.model.eval()
        with torch.no_grad():
            logits = beta.model(torch.tensor(x, dtype=torch.float32).to(device))
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        f1w  = f1_score(y, preds, average="weighted", zero_division=0)
        auc  = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.0
        eo   = eo_tpr_diff(beta, x, y, a, device, meta.get("minority_id", 1))
        return {"f1w": f1w, "auc": auc, "eo": eo}

    val_m  = get_metrics(x_val,  y_val,  a_val,  "val")
    test_m = get_metrics(x_test, y_test, a_test, "test")

    result = {
        "fold_idx": meta.get("fold_idx"),
        "lr": args.lr, "epochs": args.epochs, "optimizer": args.optimizer,
        "best_ep": best_ep, "n_synthetic": len(x_syn),
        "val_eo":   val_m["eo"],  "val_f1w":  val_m["f1w"],  "val_auc":  val_m["auc"],
        "test_eo": test_m["eo"], "test_f1w": test_m["f1w"], "test_auc": test_m["auc"],
    }

    out_path = seed_dir / "synth_eval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"fold={result['fold_idx']}  val_eo={val_m['eo']:.4f}  test_eo={test_m['eo']:.4f}  "
          f"test_f1w={test_m['f1w']:.4f}  test_auc={test_m['auc']:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
