"""
Evaluate the best saved checkpoint for a completed or in-progress run.

Computes metrics on TWO sets:
  VAL (full, uncapped) — use this for config selection / hyperparameter ranking.
      Val is the 40% of held-out subjects' windows NOT used for training.
      Passing real_data_size=None removes the 4k training-time cap so EO is
      estimated from ~138k windows (~3500 female positives) rather than 4k (~100).
  TEST (full) — for final sanity check only. Do NOT use for config selection.

Usage:
    python eval_checkpoint.py <run_dir> [--device cpu]

Batch mode (all runs in a directory):
    for d in training_runs/SPECfold0_*; do
        python eval_checkpoint.py $d --device cpu
    done
"""

import argparse, json, sys
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def compute_metrics(model, x, y, a, device, label):
    model.model.eval()
    with torch.no_grad():
        logits = model.model(x.to(device))
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = (probs >= 0.5).astype(int)

    y_np = y.cpu().numpy()
    a_np = a.cpu().numpy()

    f1w = f1_score(y_np, preds, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y_np, probs)
    except Exception:
        auc = float("nan")

    g0, g1 = (a_np == 0), (a_np == 1)
    def tpr(mask):
        denom = int((y_np[mask] == 1).sum())
        return preds[mask & (y_np == 1)].sum() / max(denom, 1)
    tpr0, tpr1 = tpr(g0), tpr(g1)
    eo = abs(tpr0 - tpr1)

    n_pos_g1 = int((y_np[g1] == 1).sum())
    print(f"  {label}: EO={eo:.4f}  TPR_g0={tpr0:.4f}  TPR_g1={tpr1:.4f}  "
          f"F1w={f1w:.4f}  AUC={auc:.4f}  (g1_pos={n_pos_g1})")
    return {"eo": eo, "tpr0": tpr0, "tpr1": tpr1, "f1w": f1w, "auc": auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        sys.exit(f"Run dir not found: {run_dir}")

    meta  = json.loads((run_dir / "meta.json").read_text())
    cfg   = meta["config"]
    seeds = json.loads((run_dir / "seeds.json").read_text())["seeds"]

    print(f"\nRun: {run_dir.name}")
    print(f"Config: pca={cfg['pca_components']} k={cfg['global_sigmoid_k']} "
          f"epochs={cfg['ffnn_epochs']} traj={cfg['TRAJ_LENGTH']} "
          f"fold={cfg.get('fold_idx','?')} seed(s)={seeds}")

    sys.path.insert(0, str(run_dir.parents[1]))
    from dataset import Dataset
    from agents.ffnn_agent import FFNNAgent

    device = torch.device(args.device)

    for seed_str in seeds:
        seed = int(seed_str)
        seed_dir = run_dir / f"seed_{seed}"
        print(f"\n{'='*60}\nSeed {seed}")

        alpha_path     = seed_dir / "alpha_state_dict.pt"
        beta_path      = seed_dir / "best_beta_state_dict_phase1_class1.pt"
        best_meta_path = seed_dir / "best_beta_meta_phase1_class1.json"

        if not beta_path.exists():
            print("  No best_beta checkpoint yet — skipping.")
            continue

        best_ep = json.loads(best_meta_path.read_text()) if best_meta_path.exists() else {}
        print(f"  Best checkpoint: ep={best_ep.get('episode','?')}  "
              f"global_obj={best_ep.get('metric_value','?'):.4f}")

        # --- Build dataset with NO val cap (real_data_size=None) ---
        # This gives the full ~138k-window val set for stable EO estimation.
        # The training-time 4k cap was for reward-computation speed only.
        ds = Dataset(
            dataset_name=cfg["dataset_name"],
            seed=seed,
            device=str(device),
            multiclass=cfg.get("MULTICLASS", cfg.get("multiclass", False)),
            minority_id=cfg["minority_id"],
            majority_id=cfg["majority_id"],
            third_id=None,
            use_pca=cfg.get("USE_PCA", cfg.get("use_pca", True)),
        )
        splits = ds.get_data_splits(
            real_data_size=None,          # no cap → full uncapped val
            da_pct=cfg.get("DA_PCT", cfg.get("da_pct")),
            bias_pct=cfg.get("BIAS_PCT", cfg.get("bias_pct")),
            pca_components=cfg["pca_components"],
            use_pca=cfg.get("USE_PCA", cfg.get("use_pca", True)),
            win_seconds=cfg.get("win_seconds"),
            step_seconds=cfg.get("step_seconds"),
            n_folds=cfg.get("n_folds"),
            fold_idx=cfg.get("fold_idx"),
            kfold_val_frac=cfg.get("kfold_val_frac", 0.4),
        )
        x_train, x_val, x_test, y_train, y_val, y_test = splits
        a_val  = ds.a_val
        a_test = ds.a_test

        print(f"  Val  (uncapped): {x_val.shape[0]:,} windows  "
              f"pos={int((y_val==1).sum()):,}  g1_pos={int(((y_val==1)&(a_val==1)).sum()):,}")
        print(f"  Test (full):     {x_test.shape[0]:,} windows  "
              f"pos={int((y_test==1).sum()):,}  g1_pos={int(((y_test==1)&(a_test==1)).sum()):,}")

        # --- Load models ---
        ffnn_kw = dict(
            input_size=cfg["pca_components"],
            hidden_sizes=cfg.get("ffnn_hidden_sizes", [32, 16]),
            output_size=2, classes=[0, 1], type="classification",
            learning_rate=cfg.get("ffnn_lr", 0.001),
            batch_size=cfg.get("ffnn_batch_size", 64),
            epochs=cfg.get("ffnn_epochs", 10),
            optimizer=cfg.get("ffnn_optimizer", "adam"),
            device=str(device), seed=seed,
        )
        alpha = FFNNAgent(**ffnn_kw)
        beta  = FFNNAgent(**ffnn_kw)

        if alpha_path.exists():
            alpha.model.load_state_dict(
                torch.load(str(alpha_path), map_location=device, weights_only=False))
        beta.model.load_state_dict(
            torch.load(str(beta_path), map_location=device, weights_only=False))

        # --- VAL metrics (use for config ranking) ---
        print("\n  -- VAL (use for config selection) --")
        val_alpha = compute_metrics(alpha, x_val,  y_val,  a_val,  device, "alpha")
        val_beta  = compute_metrics(beta,  x_val,  y_val,  a_val,  device, "beta ")
        print(f"  Val EO: {val_alpha['eo']:.4f} -> {val_beta['eo']:.4f}  "
              f"({'improved' if val_beta['eo'] < val_alpha['eo'] else 'DEGRADED'})")

        # --- TEST metrics (sanity check only — do not use for selection) ---
        print("\n  -- TEST (sanity check only) --")
        tst_alpha = compute_metrics(alpha, x_test, y_test, a_test, device, "alpha")
        tst_beta  = compute_metrics(beta,  x_test, y_test, a_test, device, "beta ")
        print(f"  Test EO: {tst_alpha['eo']:.4f} -> {tst_beta['eo']:.4f}  "
              f"({'improved' if tst_beta['eo'] < tst_alpha['eo'] else 'DEGRADED'})")

        # --- Save results ---
        out = {
            "run": run_dir.name, "seed": seed,
            "best_ep": best_ep.get("episode"),
            "global_obj": best_ep.get("metric_value"),
            "config": {
                "pca": cfg["pca_components"], "k": cfg["global_sigmoid_k"],
                "epochs": cfg["ffnn_epochs"], "traj": cfg["TRAJ_LENGTH"],
                "fold": cfg.get("fold_idx"),
            },
            "val_uncapped": {
                "n_windows": int(x_val.shape[0]),
                "g1_pos": int(((y_val==1)&(a_val==1)).sum()),
                "alpha_eo": val_alpha["eo"], "beta_eo": val_beta["eo"],
                "alpha_f1w": val_alpha["f1w"], "beta_f1w": val_beta["f1w"],
                "alpha_auc": val_alpha["auc"], "beta_auc": val_beta["auc"],
            },
            "test": {
                "n_windows": int(x_test.shape[0]),
                "g1_pos": int(((y_test==1)&(a_test==1)).sum()),
                "alpha_eo": tst_alpha["eo"], "beta_eo": tst_beta["eo"],
                "alpha_f1w": tst_alpha["f1w"], "beta_f1w": tst_beta["f1w"],
                "alpha_auc": tst_alpha["auc"], "beta_auc": tst_beta["auc"],
            },
        }
        out_path = seed_dir / "eval_checkpoint_results.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
