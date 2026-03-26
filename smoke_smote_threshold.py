"""
smoke_smote_threshold.py

Smoke test: two simple alternatives to the RL approach on census bias=0.10.

1. SMOTE — oversample (minority_group, y=1) cell in preprocessed feature space,
   train same FFNN as v18.
2. Threshold calibration — train alpha model on real data, then find the
   per-group threshold that minimises EO gap on the validation set, report
   test performance.

Run: python smoke_smote_threshold.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE

from dataset import Dataset
from agents.ffnn_agent2 import FFNNAgent

# ── Config ────────────────────────────────────────────────────────────────────
SEEDS        = [42, 0, 1, 2, 3]
BIAS_PCT     = 0.10
# Census: _map_protected_census → Male=1, Female=0.
# Female is the DISADVANTAGED group (fewer positive-class outcomes).
MINORITY_ID  = 0   # female
MAJORITY_ID  = 1   # male
HIDDEN       = [32, 16]
LR           = 0.001
BATCH        = 64
EPOCHS       = 20        # same as v18 alpha/beta
N_SMOTE      = 2000      # synthetic samples to add (matches RL traj_length scale)
DEVICE       = "cpu"

# ── Helpers ───────────────────────────────────────────────────────────────────

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_loader(X, y, batch_size, shuffle=True, seed=42):
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)

def train_ffnn(X_train, y_train, input_size, seed):
    agent = FFNNAgent(
        input_size=input_size,
        hidden_sizes=HIDDEN,
        output_size=2,
        learning_rate=LR,
        batch_size=BATCH,
        epochs=EPOCHS,
        type="classification",
        device=DEVICE,
        seed=seed,
    )
    loader = make_loader(X_train, y_train, BATCH, shuffle=True, seed=seed)
    agent.train(loader)
    return agent

def get_probs(agent, X):
    agent.model.eval()
    with torch.no_grad():
        logits = agent.model(torch.tensor(X, dtype=torch.float32, device=DEVICE))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs

def metrics_at_threshold(y_true, p1, a, thr_maj, thr_min, minority_id):
    """Compute EO gap, F1-weighted, AUC with group-specific thresholds."""
    y_hat = np.where(a == minority_id, (p1 >= thr_min).astype(int),
                                       (p1 >= thr_maj).astype(int))
    # EO: TPR gap
    def tpr(mask):
        pos = mask & (y_true == 1)
        if pos.sum() == 0:
            return float("nan")
        return float((y_hat[pos] == 1).sum() / pos.sum())

    tpr0 = tpr(a == MAJORITY_ID)
    tpr1 = tpr(a == MINORITY_ID)
    eo   = abs(tpr0 - tpr1) if not (np.isnan(tpr0) or np.isnan(tpr1)) else float("nan")

    # F1-weighted and AUC use the probs (threshold-independent for AUC)
    f1w = f1_score(y_true, y_hat, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y_true, p1)
    except Exception:
        auc = float("nan")
    return eo, f1w, auc

def metrics_default(y_true, p1, a, minority_id):
    return metrics_at_threshold(y_true, p1, a, 0.5, 0.5, minority_id)

def tune_threshold(p1_val, y_val, a_val, minority_id):
    """Grid-search minority threshold on validation set to minimise EO gap."""
    best_thr, best_eo = 0.5, float("inf")
    for thr in np.arange(0.05, 0.96, 0.05):
        eo, _, _ = metrics_at_threshold(y_val, p1_val, a_val, 0.5, thr, minority_id)
        if not np.isnan(eo) and eo < best_eo:
            best_eo, best_thr = eo, thr
    return best_thr, best_eo

# ── Per-seed loop ─────────────────────────────────────────────────────────────

alpha_results  = []
smote_results  = []
thresh_results = []

print(f"\n{'='*68}")
print(f"  census_income  bias_pct={BIAS_PCT}  seeds={SEEDS}")
print(f"{'='*68}")
print(f"{'Method':<22} {'EO↓':>8} {'F1w↑':>8} {'AUC↑':>8}  (per seed)")

for seed in SEEDS:
    seed_everything(seed)

    # ── Load data ─────────────────────────────────────────────────────────────
    ds = Dataset(
        dataset_name="census_income",
        multiclass=False,
        minority_id=1,   # spec convention: 1=male (majority_id in fairness terms)
        majority_id=0,   # 0=female (disadvantaged group — our MINORITY_ID target)
        third_id=None,
        pca_components=10,
        seed=seed,
        device=DEVICE,
        use_pca=False,      # raw preprocessed features for SMOTE
    )
    X_tr, X_val, X_test, y_tr, y_val, y_test = ds.get_data_splits(
        bias_pct=BIAS_PCT,
        val_frac=0.15,
        test_frac=0.15,
        train_size=3000,
    )
    X_tr   = X_tr.numpy();   y_tr   = y_tr.numpy()
    X_val  = X_val.numpy();  y_val  = y_val.numpy()
    X_test = X_test.numpy(); y_test = y_test.numpy()
    a_tr   = ds.a_train.numpy()
    a_val  = ds.a_val.numpy()
    a_test = ds.a_test.numpy()

    n_feat = X_tr.shape[1]
    min_pos_count = int(((a_tr == MINORITY_ID) & (y_tr == 1)).sum())
    print(f"\n  seed={seed}  min-pos training examples: {min_pos_count}")

    # ── Alpha (real data only, baseline) ──────────────────────────────────────
    alpha = train_ffnn(X_tr, y_tr, n_feat, seed)
    p1_alpha_test = get_probs(alpha, X_test)
    eo_a, f1_a, auc_a = metrics_default(y_test, p1_alpha_test, a_test, MINORITY_ID)
    alpha_results.append((eo_a, f1_a, auc_a))
    print(f"  alpha (real only)    EO={eo_a:.3f}  F1w={f1_a:.3f}  AUC={auc_a:.3f}")

    # ── SMOTE ─────────────────────────────────────────────────────────────────
    # Determine how many synthetic samples to generate so the total added ≈ N_SMOTE
    # We oversample the (minority, y=1) class by adding N_SMOTE examples.
    # SMOTE requires the final count of each class label; we work at the
    # (group, label) cell level by synthesising directly.
    mask_min_pos = (a_tr == MINORITY_ID) & (y_tr == 1)
    X_min_pos = X_tr[mask_min_pos]
    n_original = X_min_pos.shape[0]

    # Use SMOTE on the full training set but only synthetically upsampling y=1
    # Create a combined label that encodes (a, y) so SMOTE targets only minority-pos
    combined_label = a_tr * 2 + y_tr   # 0=(maj,0), 1=(maj,1), 2=(min,0), 3=(min,1)
    n_target = n_original + N_SMOTE
    # Current counts
    from collections import Counter
    cnts = Counter(combined_label.tolist())
    # Make sure we only upsample class 3=(minority,y=1); leave others at their count
    sampling_strategy = {3: max(n_target, cnts[3])}
    k_neighbors = min(5, n_original - 1)
    smote = SMOTE(sampling_strategy=sampling_strategy,
                  k_neighbors=k_neighbors,
                  random_state=seed)
    try:
        X_smote, combined_smote = smote.fit_resample(X_tr, combined_label)
        y_smote = combined_smote % 2       # recover y
        print(f"  SMOTE generated {(combined_smote==3).sum() - cnts[3]} new minority-pos samples")
    except Exception as e:
        print(f"  SMOTE failed: {e}; falling back to random oversampling")
        n_gen = N_SMOTE
        idx = np.random.choice(np.where(mask_min_pos)[0], size=n_gen, replace=True)
        X_smote = np.vstack([X_tr, X_tr[idx]])
        y_smote = np.concatenate([y_tr, y_tr[idx]])

    smote_agent = train_ffnn(X_smote, y_smote, n_feat, seed)
    p1_smote_test = get_probs(smote_agent, X_test)
    eo_s, f1_s, auc_s = metrics_default(y_test, p1_smote_test, a_test, MINORITY_ID)
    smote_results.append((eo_s, f1_s, auc_s))
    print(f"  SMOTE               EO={eo_s:.3f}  F1w={f1_s:.3f}  AUC={auc_s:.3f}")

    # ── Threshold calibration ─────────────────────────────────────────────────
    p1_alpha_val = get_probs(alpha, X_val)
    best_thr, best_eo_val = tune_threshold(p1_alpha_val, y_val, a_val, MINORITY_ID)
    eo_t, f1_t, auc_t = metrics_at_threshold(
        y_test, p1_alpha_test, a_test, 0.5, best_thr, MINORITY_ID
    )
    thresh_results.append((eo_t, f1_t, auc_t))
    print(f"  Threshold cal.      EO={eo_t:.3f}  F1w={f1_t:.3f}  AUC={auc_t:.3f}  (thr={best_thr:.2f})")


# ── Summary ───────────────────────────────────────────────────────────────────
import statistics

def summary(results, name):
    eos  = [r[0] for r in results]
    f1s  = [r[1] for r in results]
    aucs = [r[2] for r in results]
    def fmt(vals):
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{m:.3f}±{s:.3f}"
    print(f"  {name:<22} EO={fmt(eos)}  F1w={fmt(f1s)}  AUC={fmt(aucs)}")

print(f"\n{'='*68}")
print(f"  SUMMARY  ({len(SEEDS)} seeds)")
print(f"{'='*68}")
summary(alpha_results,  "Alpha (real only)")
summary(smote_results,  "SMOTE")
summary(thresh_results, "Threshold cal.")
print(f"\n  v18 reference (from paper):  EO=0.063±0.059  F1w=0.811  AUC=0.877")
print(f"{'='*68}\n")
