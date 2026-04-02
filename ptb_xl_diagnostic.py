"""
PTB-XL diagnostic: investigate whether ECG features support sex-based EO gap.

Key question: does the biased alpha classifier fail on the UNBIASED test set?
If not, PTB-XL lacks the structural group disparity needed for our paper setting.

Tests:
  1. Unbiased training → test EO  (does natural data show EO gap?)
  2. Biased training   → test EO at several bias_pct levels
  3. Effect of PCA components (10 vs 20 vs 30) with optimal bias_pct
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from dataset import Dataset
from agents.ffnn_agent2 import FFNNAgent
from reward_helpers import p1_from_agent, eo_gap_from_probs

SEED   = 42
DEVICE = "cpu"


def make_ds(bias_pct, pca_components=10, bias_val=False):
    ds = Dataset(
        dataset_name="ptb_xl",
        multiclass=False,
        minority_id=1,   # female
        majority_id=0,   # male
        third_id=None,
        seed=SEED,
        device=DEVICE,
        use_pca=True,
    )
    splits = ds.get_data_splits(
        train_size=None,
        bias_pct=bias_pct,
        pca_components=pca_components,
        dp_protected_col="sex",
        bias_val=bias_val,
    )
    return ds, splits


def train_ffnn(X_tr, y_tr, input_dim):
    model = FFNNAgent(
        input_size=input_dim,
        hidden_sizes=[32, 16],
        output_size=2,
        learning_rate=0.001,
        epochs=20,
        batch_size=64,
        device=DEVICE,
        type="classification",
    )
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    model.train(loader)
    return model


def group_tpr(a_tensor, y_tensor, p1_tensor, group_val):
    mask = (a_tensor == group_val).numpy().astype(bool)
    y_g  = y_tensor.numpy()[mask]
    p_g  = p1_tensor.numpy()[mask]
    pos  = y_g == 1
    if pos.sum() == 0:
        return float("nan"), 0
    tpr = float((p_g[pos] >= 0.5).mean())
    return tpr, int(pos.sum())


# ============================================================
# PART 1: Unbiased baseline — does a fair training set show EO?
# ============================================================
print("=" * 65)
print("PART 1: Unbiased training → test EO")
print("  (tests whether ECG features have natural sex-based EO gap)")
print("=" * 65)

ds_u, (X_tr, X_va, X_te, y_tr, y_va, y_te) = make_ds(bias_pct=None, pca_components=10)
alpha_u = train_ffnn(X_tr, y_tr, X_tr.shape[1])
p1_te = p1_from_agent(alpha_u, X_te)
eo_te = eo_gap_from_probs(ds_u.a_test, y_te, p1_te).item()

tpr_m, nm = group_tpr(ds_u.a_test, y_te, p1_te, 0)
tpr_f, nf = group_tpr(ds_u.a_test, y_te, p1_te, 1)
print(f"  Test EO = {eo_te:.3f}  |  male TPR={tpr_m:.3f} (n={nm})  female TPR={tpr_f:.3f} (n={nf})")

p1_va = p1_from_agent(alpha_u, X_va)
eo_va = eo_gap_from_probs(ds_u.a_val, y_va, p1_va).item()
print(f"  Val  EO = {eo_va:.3f}")
print()


# ============================================================
# PART 2: Bias scan — test EO at each bias_pct
# ============================================================
print("=" * 65)
print("PART 2: Biased training → test EO  (bias_val=False, 10 PCA)")
print("=" * 65)

for bp in [0.04, 0.03, 0.02, 0.015, 0.01]:
    try:
        ds_b, (Xtr, Xva, Xte, ytr, yva, yte) = make_ds(bias_pct=bp, pca_components=10, bias_val=False)
    except Exception as e:
        print(f"  bias_pct={bp}: ERROR — {e}")
        continue

    a_tr_np = ds_b.a_train.numpy()
    y_tr_np = ytr.numpy()
    da_plus  = int(np.sum((y_tr_np == 1) & (a_tr_np == 1)))  # female MI in train
    n_disadv = int(np.sum(a_tr_np == 1))

    alpha_b = train_ffnn(Xtr, ytr, Xtr.shape[1])
    p1_te = p1_from_agent(alpha_b, Xte)
    eo_te = eo_gap_from_probs(ds_b.a_test, yte, p1_te).item()
    tpr_m, nm = group_tpr(ds_b.a_test, yte, p1_te, 0)
    tpr_f, nf = group_tpr(ds_b.a_test, yte, p1_te, 1)

    p1_va = p1_from_agent(alpha_b, Xva)
    eo_va = eo_gap_from_probs(ds_b.a_val, yva, p1_va).item()

    print(f"  bias_pct={bp:.3f}: DA+={da_plus:3d}  "
          f"val_EO={eo_va:.3f}  test_EO={eo_te:.3f}  "
          f"[male_TPR={tpr_m:.3f} n={nm}  female_TPR={tpr_f:.3f} n={nf}]")

print()


# ============================================================
# PART 3: PCA components scan at bias_pct=0.015
# ============================================================
print("=" * 65)
print("PART 3: PCA components scan at bias_pct=0.015 (bias_val=False)")
print("=" * 65)

for n_pca in [10, 15, 20, 30, 50]:
    try:
        ds_p, (Xtr, Xva, Xte, ytr, yva, yte) = make_ds(bias_pct=0.015, pca_components=n_pca, bias_val=False)
    except Exception as e:
        print(f"  pca={n_pca}: ERROR — {e}")
        continue

    alpha_p = train_ffnn(Xtr, ytr, Xtr.shape[1])
    p1_te = p1_from_agent(alpha_p, Xte)
    eo_te = eo_gap_from_probs(ds_p.a_test, yte, p1_te).item()
    tpr_m, nm = group_tpr(ds_p.a_test, yte, p1_te, 0)
    tpr_f, nf = group_tpr(ds_p.a_test, yte, p1_te, 1)

    p1_va = p1_from_agent(alpha_p, Xva)
    eo_va = eo_gap_from_probs(ds_p.a_val, yva, p1_va).item()

    a_tr_np = ds_p.a_train.numpy()
    da_plus = int(np.sum((ytr.numpy() == 1) & (a_tr_np == 1)))
    print(f"  pca={n_pca:2d}: DA+={da_plus:3d}  val_EO={eo_va:.3f}  test_EO={eo_te:.3f}  "
          f"[male_TPR={tpr_m:.3f} n={nm}  female_TPR={tpr_f:.3f} n={nf}]")

print()


# ============================================================
# PART 4: Direct feature analysis — group-conditional MI means
# ============================================================
print("=" * 65)
print("PART 4: Feature separability — group-conditional MI statistics")
print("  (check if male MI and female MI look different in raw 96-d space)")
print("=" * 65)

ds_raw, (X_tr_r, X_va_r, X_te_r, y_tr_r, y_va_r, y_te_r) = make_ds(bias_pct=None, pca_components=10)

# Reconstruct full standardized features (before PCA) by calling Dataset.split_ptb_xl
# with return_raw=True is not implemented; instead compare PCA projections.
# Compare PCA-space centroid of male-MI vs female-MI on test (unbiased).
X_te_np = X_te_r.numpy()
y_te_np = y_te_r.numpy()
a_te_np = ds_raw.a_test.numpy()

male_mi   = X_te_np[(y_te_np == 1) & (a_te_np == 0)]
female_mi = X_te_np[(y_te_np == 1) & (a_te_np == 1)]
male_neg  = X_te_np[(y_te_np == 0) & (a_te_np == 0)]
female_neg= X_te_np[(y_te_np == 0) & (a_te_np == 1)]

print(f"  Test subgroup sizes: male_MI={len(male_mi)}, female_MI={len(female_mi)}, "
      f"male_neg={len(male_neg)}, female_neg={len(female_neg)}")

# L2 distance between group MI centroids vs within-group MI std
c_male  = male_mi.mean(axis=0)
c_female= female_mi.mean(axis=0)
between = float(np.linalg.norm(c_male - c_female))
within_m = float(np.mean(male_mi.std(axis=0)))
within_f = float(np.mean(female_mi.std(axis=0)))

print(f"  Centroid distance (male_MI vs female_MI): {between:.3f}")
print(f"  Within-group spread: male_MI={within_m:.3f}  female_MI={within_f:.3f}")
print(f"  Separability ratio (between/within): {between / max(within_m, within_f):.3f}")
print()

# Also check: cosine similarity between class centroids per group
male_neg_c  = male_neg.mean(axis=0)
female_neg_c= female_neg.mean(axis=0)
c_m_disc   = c_male   - male_neg_c
c_f_disc   = c_female - female_neg_c
cos = float(np.dot(c_m_disc, c_f_disc) / (np.linalg.norm(c_m_disc) * np.linalg.norm(c_f_disc) + 1e-8))
print(f"  Cosine similarity of MI-discriminant vectors (male vs female): {cos:.3f}")
print("  (1.0 = same direction; <0.5 = very different MI signatures)")
