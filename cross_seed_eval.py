#!/usr/bin/env python3
"""
cross_seed_eval.py — Cross-seed generalization test.

For each model seed M: load beta model + M's preprocessing pipeline (encoder, scaler, PCA).
For each test seed N: apply M's pipeline to N's raw test data and evaluate EO.

Diagonal = standard same-seed evaluation.
Off-diagonal = cross-seed generalization: if the agent overfits to its val set,
off-diagonal EO should degrade relative to the diagonal.

Usage:
    python cross_seed_eval.py training_runs/<run_dir>
"""

import sys
import os
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import Dataset
from agents.ffnn_agent2 import FFNNAgent


SEEDS   = [0, 1, 42]
DEVICE  = "cpu"
DA_PCT  = 0.014
PCA_N   = 10


def compute_eo(y_true: torch.Tensor, y_pred: torch.Tensor, a: torch.Tensor):
    """EO gap = |TPR(a=0) - TPR(a=1)|. Returns (eo, tpr_a0, tpr_a1)."""
    eos = {}
    for g in [0, 1]:
        pos = (a == g) & (y_true == 1)
        eos[g] = (y_pred[pos] == 1).float().mean().item() if pos.sum() > 0 else float("nan")
    return abs(eos[0] - eos[1]), eos[0], eos[1]


def load_dataset(seed):
    ds = Dataset(
        "census_income", multiclass=False,
        minority_id=0, majority_id=1, third_id=None,
        pca_components=PCA_N, seed=seed, device=DEVICE, use_pca=True,
    )
    result = ds.get_data_splits(
        da_pct=DA_PCT, pca_components=PCA_N,
        dp_protected_col="sex", return_raw=True,
    )
    # result: X_train, X_val, X_test, y_train, y_val, y_test, X_test_raw_df, y_test_raw_np
    return ds, result


def apply_pipeline(X_raw_df, encoder, scaler, pca, cat_cols, num_cols):
    """Apply a fixed encoder+scaler+PCA to a raw test DataFrame."""
    X_cat = encoder.transform(X_raw_df[cat_cols]) if cat_cols else np.empty((len(X_raw_df), 0))
    X_num = scaler.transform(X_raw_df[num_cols])  if num_cols else np.empty((len(X_raw_df), 0))
    X_all = np.hstack([X_num, X_cat])
    X_pca = pca.transform(X_all) if pca is not None else X_all
    return torch.tensor(X_pca, dtype=torch.float32)


def load_beta(seed_dir):
    with open(os.path.join(seed_dir, "ffnn_meta.json")) as f:
        meta = json.load(f)
    agent = FFNNAgent(
        input_size=meta["input_size"],
        hidden_sizes=meta["hidden_sizes"],
        output_size=meta["output_size"],
        type="classification",
        classes=meta["classes"],
        device=DEVICE,
    )
    state_dict = torch.load(
        os.path.join(seed_dir, "best_beta_state_dict_phase1_class1.pt"),
        map_location=DEVICE,
    )
    agent.model.load_state_dict(state_dict)
    agent.model.eval()
    return agent


def main():
    if len(sys.argv) < 2:
        print("Usage: python cross_seed_eval.py <run_dir>")
        sys.exit(1)

    run_dir = sys.argv[1]
    print(f"Run: {run_dir}\n")

    print("Loading datasets and models...")
    datasets, models = {}, {}
    for s in SEEDS:
        ds, result = load_dataset(s)
        datasets[s] = (ds, result)
        models[s]   = load_beta(os.path.join(run_dir, f"seed_{s}"))
        print(f"  seed={s} loaded — test size={result[2].shape[0]}")

    print("\n" + "="*65)
    print("EO gap (beta model) — rows=model seed, cols=test seed")
    print("  diagonal = standard same-seed | off-diagonal = cross-seed")
    print("="*65)

    header = f"{'':14}" + "".join(f"  test_s={n}" for n in SEEDS)
    print(header)

    eo_matrix = {}
    for ms in SEEDS:
        ds_M, _ = datasets[ms]
        cache   = ds_M._gan_view_cache
        beta_M  = models[ms]

        row = f"  model_s={ms:>2}  "
        eo_matrix[ms] = {}
        for ts in SEEDS:
            ds_N, result_N = datasets[ts]
            X_raw_N = result_N[6]   # raw test DataFrame from seed N
            y_test  = result_N[5]   # torch tensor
            a_test  = ds_N.a_test

            X_pca = apply_pipeline(
                X_raw_N,
                cache["encoder"], cache["scaler"], cache["pca"],
                cache["cat_cols"], cache["num_cols"],
            )

            y_pred = beta_M.predict(X_pca)
            eo, tpr0, tpr1 = compute_eo(y_test, y_pred, a_test)
            eo_matrix[ms][ts] = eo
            marker = " *" if ms == ts else "  "
            row += f"  {eo:.3f}{marker}"
        print(row)

    print("\n* = same-seed (diagonal)")

    # Summary: mean off-diagonal vs diagonal
    diag     = [eo_matrix[s][s] for s in SEEDS]
    offdiag  = [eo_matrix[ms][ts] for ms in SEEDS for ts in SEEDS if ms != ts]
    print(f"\nMean diagonal EO:     {np.mean(diag):.3f}")
    print(f"Mean off-diagonal EO: {np.mean(offdiag):.3f}")
    diff = np.mean(offdiag) - np.mean(diag)
    print(f"Δ (off - diag):       {diff:+.3f}  "
          f"({'worse — possible overfitting signal' if diff > 0.02 else 'stable — no clear overfitting'})")


if __name__ == "__main__":
    main()
