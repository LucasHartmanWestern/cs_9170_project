"""
Quick diagnostic: COMPAS with race as protected attribute.
Sweeps bias_pct values and seeds to find a regime with alpha-EO >= 0.10
and DA+ in a useful range.

Usage: python compas_race_diagnostic.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from dataset import Dataset
from agents.ffnn_agent2 import FFNNAgent
from reward_helpers import fairness_classification_metrics

SEEDS     = [0, 2, 3, 5, 42]          # same as Census
# bias_pct now = target Caucasian positive rate within Caucasian subgroup
# (group-specific bias: AA positives kept at natural rate)
BIAS_PCTS = [0.05, 0.07, 0.10, 0.14]
DEVICE    = "cpu"

def run_alpha(bias_pct, seed):
    ds = Dataset(
        dataset_name="compas",
        multiclass=False,
        minority_id=0,       # 0 = Caucasian (disadvantaged)
        majority_id=1,       # 1 = African-American
        third_id=None,
        pca_components=10,
        seed=seed,
        device=DEVICE,
        use_pca=True,
    )
    X_train, X_val, X_test, y_train, y_val, y_test = ds.get_data_splits(
        bias_pct=bias_pct,
        dp_protected_col="race",
        drop_protected=False,
    )
    # Protected attributes stored on ds after call
    a_train = ds.a_train
    a_test  = ds.a_test

    # DA+: disadvantaged (a=0, Caucasian) positives in training
    a_train_np = a_train.cpu().numpy() if torch.is_tensor(a_train) else a_train
    y_train_np = y_train.cpu().numpy() if torch.is_tensor(y_train) else y_train
    mask_disadv = (a_train_np == 0)
    mask_adv    = (a_train_np == 1)
    da_plus     = int((y_train_np[mask_disadv] == 1).sum())
    aa_plus     = int((y_train_np[mask_adv]    == 1).sum())

    # Train alpha on real data only
    input_dim = X_train.shape[1]
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import roc_auc_score, f1_score

    alpha = FFNNAgent(
        input_size=input_dim,
        hidden_sizes=[32, 16],
        output_size=2,
        classes=[0, 1],
        learning_rate=0.001,
        batch_size=64,
        epochs=20,
        type="classification",
        device=DEVICE,
        seed=seed,
    )
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    alpha.train(loader)

    # Evaluate on test set — use raw logits (predict() returns argmax, not probs)
    alpha.model.eval()
    with torch.no_grad():
        logits = alpha.model(X_test.to(DEVICE))   # shape (N, 2)
        p1 = torch.softmax(logits, dim=1)[:, 1]   # class-1 probability

    a_t = a_test if torch.is_tensor(a_test) else torch.tensor(a_test.astype(int))
    y_t = y_test if torch.is_tensor(y_test) else torch.tensor(y_test.astype(int))

    p1_np = p1.cpu().numpy()
    y_np  = y_t.cpu().numpy()
    a_np  = a_t.cpu().numpy()

    # Optimal threshold (Youden's J) for EO evaluation
    from sklearn.metrics import roc_curve
    fpr_c, tpr_c, thresholds = roc_curve(y_np, p1_np)
    youden = tpr_c - fpr_c
    opt_thresh = float(thresholds[np.argmax(youden)])
    opt_thresh = max(0.01, min(0.99, opt_thresh))

    preds_05  = (p1_np >= 0.50).astype(int)
    preds_opt = (p1_np >= opt_thresh).astype(int)
    alpha_auc = roc_auc_score(y_np, p1_np)
    alpha_f1w = f1_score(y_np, preds_05, average="weighted")
    frac_pos  = float(preds_05.mean())

    # EO at both thresholds
    metrics    = fairness_classification_metrics(a_t, y_t, p1, threshold=0.5)
    metrics_op = fairness_classification_metrics(a_t, y_t, p1, threshold=opt_thresh)
    alpha_eo    = float(metrics["eo_tpr_diff"])
    alpha_eo_op = float(metrics_op["eo_tpr_diff"])

    # Per-group TPR at 0.5 for debugging (first run only)
    if seed == 0:
        for g, gname in [(0, "Caucasian"), (1, "AA")]:
            m = (a_np == g)
            pos = m & (y_np == 1)
            tpr_g = preds_05[pos].mean() if pos.sum() > 0 else float("nan")
            tpr_g_op = preds_opt[pos].mean() if pos.sum() > 0 else float("nan")
            print(f"    [{gname}] n_pos={pos.sum()}, TPR@0.5={tpr_g:.3f}, TPR@opt({opt_thresh:.2f})={tpr_g_op:.3f}, frac_pred_pos={preds_05[m].mean():.3f}")

    return {
        "bias_pct":    bias_pct,
        "seed":        seed,
        "da_plus":     da_plus,
        "aa_plus":     aa_plus,
        "alpha_eo":    alpha_eo,
        "alpha_eo_op": alpha_eo_op,
        "alpha_auc":   alpha_auc,
        "alpha_f1w":   alpha_f1w,
        "frac_pos":    frac_pos,
    }

print(f"{'bias':>6} {'seed':>5} {'DA+':>5} {'AA+':>5} {'EO@0.5':>8} {'EO@opt':>8} {'AUC':>7} {'F1w':>7} {'frac+':>7}")
print("-" * 75)

results_all = []
for bias_pct in BIAS_PCTS:
    for seed in SEEDS:
        try:
            r = run_alpha(bias_pct, seed)
            results_all.append(r)
            print(f"{r['bias_pct']:>6.2f} {r['seed']:>5} {r['da_plus']:>5} {r['aa_plus']:>5} "
                  f"{r['alpha_eo']:>8.4f} {r['alpha_eo_op']:>8.4f} {r['alpha_auc']:>7.4f} {r['alpha_f1w']:>7.4f} {r['frac_pos']:>7.3f}")
        except Exception as e:
            print(f"{bias_pct:>6.2f} {seed:>5}  ERROR: {e}")

print()
print("Summary — mean alpha_EO per bias_pct:")
import collections
by_bias = collections.defaultdict(list)
for r in results_all:
    by_bias[r['bias_pct']].append(r['alpha_eo'])
for b, eos in sorted(by_bias.items()):
    print(f"  bias={b:.2f}: mean_EO={np.mean(eos):.4f}  DA+ range approx from above")
