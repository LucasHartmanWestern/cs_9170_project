"""
PTB-XL age-based diagnostic.

Protected attribute: age (young=0 for age<50, old=1 for age>=60; drop 50-60 middle).
Young patients have naturally low MI rate (~8%), old patients high (~47%).
This mirrors the census sex structure without any bias injection.

Tests:
  1. PCA variance explained curve → optimal n_components
  2. Discriminant vector cosine similarity: age-based vs sex-based
  3. Alpha-EO scan: no bias, then increasing bias until significant EO gap
"""
import ast
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader

from agents.ffnn_agent2 import FFNNAgent
from reward_helpers import p1_from_agent, eo_gap_from_probs

SEED   = 42
DEVICE = "cpu"
DATA_DIR = Path("datasets/ptb-xl")

np.random.seed(SEED)


# ============================================================
# Helper: load and filter PTB-XL to MI/NORM, build age groups
# ============================================================
def load_ptbxl(age_young_max=50, age_old_min=60, mi_threshold=50.0):
    db = pd.read_csv(DATA_DIR / "ptbxl_database.csv")
    scp = pd.read_csv(DATA_DIR / "scp_statements.csv", index_col=0)
    mi_codes   = set(scp[scp["diagnostic_class"] == "MI"].index.tolist())
    norm_codes = set(scp[scp["diagnostic_class"] == "NORM"].index.tolist())

    def classify(s):
        try: codes = ast.literal_eval(s)
        except: return -1
        is_mi   = any(codes.get(c, 0) >= mi_threshold for c in mi_codes)
        is_norm = any(codes.get(c, 0) >= mi_threshold for c in norm_codes)
        if is_mi and not is_norm: return 1
        if is_norm and not is_mi: return 0
        return -1

    db["y"] = db["scp_codes"].apply(classify)
    db = db[(db["y"] >= 0) & (db["age"] < 120)].reset_index(drop=True)

    # Age-based groups: 0=young (<50), 1=old (>=60); drop 50-59 middle
    db["a_age"] = -1
    db.loc[db["age"] < age_young_max,  "a_age"] = 0  # young (disadvantaged)
    db.loc[db["age"] >= age_old_min,   "a_age"] = 1  # old   (advantaged)
    db_age = db[db["a_age"] >= 0].copy().reset_index(drop=True)

    # Sex-based groups (0=male, 1=female) — keep for comparison
    db_age["a_sex"] = db_age["sex"].astype(int)

    return db_age


def load_features(db):
    cache = np.load(DATA_DIR / "ptbxl_features_cache.npz", allow_pickle=True)
    cached_ids = cache["ecg_ids"].tolist()
    cached_X   = cache["X"]
    id_to_row  = {eid: i for i, eid in enumerate(cached_ids)}
    valid_mask = db["ecg_id"].isin(set(cached_ids))
    db = db[valid_mask].reset_index(drop=True)
    row_indices = [id_to_row[eid] for eid in db["ecg_id"]]
    X = cached_X[row_indices]
    return db, X


def train_val_test_split(db, X):
    """Use official strat_fold: 1-6 train, 7-8 val, 9-10 test."""
    fold = db["strat_fold"].to_numpy()
    tr = fold <= 6
    va = (fold == 7) | (fold == 8)
    te = (fold == 9) | (fold == 10)
    return (X[tr], X[va], X[te],
            db["y"].to_numpy()[tr], db["y"].to_numpy()[va], db["y"].to_numpy()[te],
            db["a_age"].to_numpy()[tr], db["a_age"].to_numpy()[va], db["a_age"].to_numpy()[te],
            db["a_sex"].to_numpy()[tr], db["a_sex"].to_numpy()[va], db["a_sex"].to_numpy()[te])


def apply_bias_age(X_tr, y_tr, a_tr, bias_pct):
    """Reduce young (a=0) positives to target_pct rate within young group."""
    is_young = (a_tr == 0)
    idx_old  = np.where(~is_young)[0]
    idx_yneg = np.where(is_young & (y_tr == 0))[0]
    idx_ypos = np.where(is_young & (y_tr == 1))[0]

    n_yneg = len(idx_yneg)
    keep = int(np.floor((bias_pct * n_yneg) / (1 - bias_pct)))
    keep = max(1, min(len(idx_ypos), keep))
    rng = np.random.RandomState(SEED)
    keep_idx = rng.choice(idx_ypos, size=keep, replace=False)

    sel = np.concatenate([idx_old, idx_yneg, keep_idx])
    rng2 = np.random.RandomState(SEED)
    perm = rng2.permutation(len(sel))
    sel = sel[perm]
    return X_tr[sel], y_tr[sel], a_tr[sel]


def train_ffnn(X_tr, y_tr):
    t = torch.tensor(X_tr, dtype=torch.float32)
    l = torch.tensor(y_tr, dtype=torch.long)
    model = FFNNAgent(
        input_size=X_tr.shape[1], hidden_sizes=[32, 16], output_size=2,
        learning_rate=0.001, epochs=20, batch_size=64, device=DEVICE, type="classification",
    )
    loader = DataLoader(TensorDataset(t, l), batch_size=64, shuffle=True)
    model.train(loader)
    return model


def group_tpr(a, y, p1, grp):
    m = (a == grp)
    y_g, p_g = y[m], p1[m]
    pos = y_g == 1
    if pos.sum() == 0: return float("nan"), 0
    return float((p_g[pos] >= 0.5).mean()), int(pos.sum())


def eo_from_arrays(a, y, p1):
    t0, n0 = group_tpr(a, y, p1, 0)
    t1, n1 = group_tpr(a, y, p1, 1)
    return abs(t0 - t1), t0, t1, n0, n1


# ============================================================
# Load data
# ============================================================
print("Loading PTB-XL with age grouping (young<50 vs old>=60)...")
db, X_raw = load_features(load_ptbxl())

print(f"Records: {len(db)}  MI={int((db['y']==1).sum())}  NORM={int((db['y']==0).sum())}")
print(f"Young (a=0): n={int((db['a_age']==0).sum())}  MI={int(((db['a_age']==0)&(db['y']==1)).sum())}")
print(f"Old   (a=1): n={int((db['a_age']==1).sum())}  MI={int(((db['a_age']==1)&(db['y']==1)).sum())}")
print()

(Xtr, Xva, Xte,
 ytr, yva, yte,
 atr_age, ava_age, ate_age,
 atr_sex, ava_sex, ate_sex) = train_val_test_split(db, X_raw)

print(f"Train: {len(Xtr)}  young={int((atr_age==0).sum())} (MI={int(((atr_age==0)&(ytr==1)).sum())})"
      f"  old={int((atr_age==1).sum())} (MI={int(((atr_age==1)&(ytr==1)).sum())})")
print(f"Val:   {len(Xva)}")
print(f"Test:  {len(Xte)}")
print()


# ============================================================
# PART 1: PCA explained variance curve
# ============================================================
print("=" * 65)
print("PART 1: PCA explained variance (fit on train only, age subset)")
print("=" * 65)

scaler = StandardScaler()
Xtr_z = scaler.fit_transform(Xtr)

pca_full = PCA(n_components=min(96, len(Xtr_z))).fit(Xtr_z)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]
for t in thresholds:
    n = int(np.argmax(cumvar >= t)) + 1
    print(f"  {int(t*100)}% variance explained: {n} components")

print()
print("  Component-by-component (first 50):")
for i in range(0, 50, 5):
    vals = pca_full.explained_variance_ratio_[i:i+5]
    cumv = cumvar[i:i+5]
    line = "  " + "  ".join(f"PC{i+j+1}:{v:.3f}(cum:{c:.3f})"
                             for j, (v, c) in enumerate(zip(vals, cumv)))
    print(line)
print()


# ============================================================
# PART 2: Discriminant vector analysis — age vs sex
# ============================================================
print("=" * 65)
print("PART 2: MI discriminant vector similarity (age-based vs sex-based)")
print("  Using unbiased test set, 30 PCA components")
print("=" * 65)

Xva_z = scaler.transform(Xva)
Xte_z = scaler.transform(Xte)
pca30 = PCA(n_components=30, random_state=SEED).fit(Xtr_z)
Xtr30 = pca30.transform(Xtr_z)
Xva30 = pca30.transform(Xva_z)
Xte30 = pca30.transform(Xte_z)

def discriminant_analysis(X_te, y_te, a_te, label):
    pos0 = X_te[(a_te==0) & (y_te==1)]
    neg0 = X_te[(a_te==0) & (y_te==0)]
    pos1 = X_te[(a_te==1) & (y_te==1)]
    neg1 = X_te[(a_te==1) & (y_te==0)]

    d0 = pos0.mean(0) - neg0.mean(0)   # MI direction for group 0
    d1 = pos1.mean(0) - neg1.mean(0)   # MI direction for group 1

    cos = float(np.dot(d0, d1) / (np.linalg.norm(d0) * np.linalg.norm(d1) + 1e-8))
    between = float(np.linalg.norm(pos0.mean(0) - pos1.mean(0)))
    within  = float(np.mean([pos0.std(0).mean(), pos1.std(0).mean()]))
    print(f"  [{label}] cosine(d0, d1)={cos:.3f}  "
          f"MI_centroid_dist={between:.3f}  within_spread={within:.3f}")
    print(f"    group0: n_pos={len(pos0)}, n_neg={len(neg0)}")
    print(f"    group1: n_pos={len(pos1)}, n_neg={len(neg1)}")

discriminant_analysis(Xte30, yte, ate_age, "age-based (young=0, old=1)")
discriminant_analysis(Xte30, yte, ate_sex, "sex-based  (male=0, female=1)")
print()


# ============================================================
# PART 3: Alpha-EO scan — age-based, no bias then increasing bias
# ============================================================
print("=" * 65)
print("PART 3: Alpha-EO at different DA+ levels (age-based, 30 PCA)")
print("  Group 0 = young (<50), disadvantaged; group 1 = old (>=60)")
print("=" * 65)

def run_alpha_eo(bias_pct_val, n_pca, label):
    pca_n = PCA(n_components=n_pca, random_state=SEED).fit(Xtr_z)
    Xtr_p = pca_n.transform(Xtr_z)
    Xva_p = pca_n.transform(Xva_z)
    Xte_p = pca_n.transform(Xte_z)

    if bias_pct_val is not None:
        Xtr_b, ytr_b, atr_b = apply_bias_age(Xtr_p, ytr, atr_age, bias_pct_val)
    else:
        Xtr_b, ytr_b, atr_b = Xtr_p, ytr, atr_age.copy()

    da_plus = int(np.sum((atr_b == 0) & (ytr_b == 1)))
    n_young = int(np.sum(atr_b == 0))
    n_old_pos = int(np.sum((atr_b == 1) & (ytr_b == 1)))

    alpha = train_ffnn(Xtr_b, ytr_b)
    p1_te = p1_from_agent(alpha, torch.tensor(Xte_p, dtype=torch.float32))
    eo_te, t0_te, t1_te, n0_te, n1_te = eo_from_arrays(ate_age, yte, p1_te.numpy())

    p1_va = p1_from_agent(alpha, torch.tensor(Xva_p, dtype=torch.float32))
    eo_va, t0_va, t1_va, _, _ = eo_from_arrays(ava_age, yva, p1_va.numpy())

    print(f"  {label}: DA+={da_plus:4d} ({da_plus}/{n_young} young train, "
          f"old_pos={n_old_pos})")
    print(f"    val_EO={eo_va:.3f}  test_EO={eo_te:.3f}  "
          f"[young_TPR={t0_te:.3f}(n={n0_te})  old_TPR={t1_te:.3f}(n={n1_te})]")

# Unbiased (natural rates)
run_alpha_eo(None, 30, "no_bias       ")

# Bias scan: reduce young MI to target positive rates
for bp in [0.06, 0.04, 0.03, 0.02, 0.01]:
    run_alpha_eo(bp, 30, f"bias_pct={bp:.2f}")

print()


# ============================================================
# PART 4: PCA component sensitivity at best bias level
# ============================================================
print("=" * 65)
print("PART 4: PCA components sensitivity at bias_pct=0.04")
print("=" * 65)

for n_pca in [10, 15, 20, 30, 50]:
    run_alpha_eo(0.04, n_pca, f"pca={n_pca:2d}")
