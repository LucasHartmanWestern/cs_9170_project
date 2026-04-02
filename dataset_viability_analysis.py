"""
Systematic viability analysis for PTB-XL and MEPS against census structural criteria.

Census reference values:
  - Val disadv positives (biased): ~42
  - Test disadv positives (unbiased): ~670
  - Alpha-EO: ~0.40+
  - Discriminant cosine similarity: low (strong group divergence)
  - Total dataset: ~48k

Criteria to pass:
  1. Val disadv positives (biased) >= 40
  2. Test disadv positives (unbiased) >= 200
  3. Alpha-EO >= 0.25 at target DA+
  4. Discriminant cosine < 0.80
"""
import ast, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset, DataLoader
from agents.ffnn_agent2 import FFNNAgent
from reward_helpers import p1_from_agent, eo_gap_from_probs

SEED   = 42
DEVICE = "cpu"
np.random.seed(SEED)

def train_ffnn(X_tr, y_tr):
    model = FFNNAgent(input_size=X_tr.shape[1], hidden_sizes=[32,16], output_size=2,
                      learning_rate=0.001, epochs=20, batch_size=64, device=DEVICE, type="classification")
    loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                      torch.tensor(y_tr, dtype=torch.long)), batch_size=64, shuffle=True)
    model.train(loader)
    return model

def eo_gap(a, y, p1):
    for g in [0, 1]:
        m = (a == g) & (y == 1)
        if m.sum() == 0: return float("nan")
    t0 = float((p1[(a==0)&(y==1)] >= 0.5).mean())
    t1 = float((p1[(a==1)&(y==1)] >= 0.5).mean())
    return abs(t0 - t1), t0, t1

def cosine_discriminant(X, y, a):
    d0 = X[(a==0)&(y==1)].mean(0) - X[(a==0)&(y==0)].mean(0)
    d1 = X[(a==1)&(y==1)].mean(0) - X[(a==1)&(y==0)].mean(0)
    return float(np.dot(d0,d1)/(np.linalg.norm(d0)*np.linalg.norm(d1)+1e-8))

def apply_bias(X, y, a, disadv_code, bias_pct, seed=SEED):
    is_disadv = (a == disadv_code)
    idx_adv   = np.where(~is_disadv)[0]
    idx_dneg  = np.where(is_disadv & (y==0))[0]
    idx_dpos  = np.where(is_disadv & (y==1))[0]
    keep = int(np.floor(bias_pct * len(idx_dneg) / (1 - bias_pct)))
    keep = max(1, min(len(idx_dpos), keep))
    rng = np.random.RandomState(seed)
    keep_idx = rng.choice(idx_dpos, size=keep, replace=False)
    sel = np.concatenate([idx_adv, idx_dneg, keep_idx])
    perm = np.random.RandomState(seed).permutation(len(sel))
    return X[sel[perm]], y[sel[perm]], a[sel[perm]]

def pca_transform(X_tr, X_va, X_te, n=25):
    sc = StandardScaler().fit(X_tr)
    pca = PCA(n_components=n, random_state=SEED).fit(sc.transform(X_tr))
    return pca.transform(sc.transform(X_tr)), pca.transform(sc.transform(X_va)), pca.transform(sc.transform(X_te))

def check_framing(label, X_tr, y_tr, a_tr, X_va, y_va, a_va, X_te, y_te, a_te,
                  disadv_code, bias_pct, n_pca=25):
    Xtr_p, Xva_p, Xte_p = pca_transform(X_tr, X_va, X_te, n_pca)
    Xtr_b, ytr_b, atr_b = apply_bias(Xtr_p, y_tr, a_tr, disadv_code, bias_pct)
    Xva_b, yva_b, ava_b = apply_bias(Xva_p, y_va, a_va, disadv_code, bias_pct)

    val_pos  = int(np.sum((yva_b==1) & (ava_b==disadv_code)))
    test_pos = int(np.sum((y_te==1)  & (a_te==disadv_code)))
    da_plus  = int(np.sum((ytr_b==1) & (atr_b==disadv_code)))

    alpha = train_ffnn(Xtr_b, ytr_b)
    p1 = p1_from_agent(alpha, torch.tensor(Xte_p, dtype=torch.float32)).numpy()
    eo, t0, t1 = eo_gap(a_te, y_te, p1)

    cos = cosine_discriminant(Xte_p, y_te, a_te)

    ok1 = "✓" if val_pos  >= 40  else "✗"
    ok2 = "✓" if test_pos >= 200 else "✗"
    ok3 = "✓" if eo       >= 0.25 else "✗"
    ok4 = "✓" if cos      <  0.80 else "✗"

    print(f"\n  {label}")
    print(f"    bias_pct={bias_pct}  DA+={da_plus}  pca={n_pca}")
    print(f"    {ok1} val_disadv_pos={val_pos:4d} (need>=40)")
    print(f"    {ok2} test_disadv_pos={test_pos:4d} (need>=200)")
    print(f"    {ok3} alpha_EO={eo:.3f} (need>=0.25)  disadv_TPR={t0:.3f}  adv_TPR={t1:.3f}")
    print(f"    {ok4} discriminant_cosine={cos:.3f} (need<0.80)")
    passes = sum([val_pos>=40, test_pos>=200, eo>=0.25, cos<0.80])
    print(f"    PASS: {passes}/4 criteria")
    return passes

# ==============================================================
# PTB-XL ANALYSIS
# ==============================================================
print("=" * 70)
print("PTB-XL: Testing all viable diagnostic class framings")
print("=" * 70)

DATA_DIR = Path("datasets/ptb-xl")
db = pd.read_csv(DATA_DIR / "ptbxl_database.csv")
scp = pd.read_csv(DATA_DIR / "scp_statements.csv", index_col=0)

# Load feature cache
cache = np.load(DATA_DIR / "ptbxl_features_cache.npz", allow_pickle=True)
cached_ids = cache["ecg_ids"].tolist()
cached_X   = cache["X"]
id_to_row  = {eid: i for i, eid in enumerate(cached_ids)}

db = db[db["ecg_id"].isin(set(cached_ids))].reset_index(drop=True)
db = db[db["age"] < 120].reset_index(drop=True)
row_indices = [id_to_row[eid] for eid in db["ecg_id"]]
X_all = cached_X[row_indices]

# All diagnostic classes
diag_classes = scp["diagnostic_class"].dropna().unique().tolist()
print(f"\nDiagnostic classes: {diag_classes}")

# For each class, check MI-equivalent (positive) vs NORM (negative)
for diag_pos in ["MI", "STTC", "HYP", "CD"]:  # CD=conduction disturbance
    pos_codes  = set(scp[scp["diagnostic_class"] == diag_pos].index.tolist())
    norm_codes = set(scp[scp["diagnostic_class"] == "NORM"].index.tolist())

    def classify(s, thresh=50.0):
        try: codes = ast.literal_eval(s)
        except: return -1
        is_pos  = any(codes.get(c,0) >= thresh for c in pos_codes)
        is_norm = any(codes.get(c,0) >= thresh for c in norm_codes)
        if is_pos and not is_norm: return 1
        if is_norm and not is_pos: return 0
        return -1

    db["y_tmp"] = db["scp_codes"].apply(classify)
    sub = db[db["y_tmp"] >= 0].copy().reset_index(drop=True)
    sub_X = X_all[[list(db.index).index(i) if i < len(db) else 0
                   for i in sub.index]]
    # Reindex properly
    orig_idx = sub.index.tolist()
    sub_X = np.array([X_all[list(db.index).index(i)] for i in sub.index
                      if i in list(db.index)])

    # Actually let's rebuild cleanly
    mask = db["y_tmp"] >= 0
    sub2 = db[mask].copy().reset_index(drop=True)
    sub2_X = X_all[mask.values]

    n_pos = int((sub2["y_tmp"]==1).sum())
    n_neg = int((sub2["y_tmp"]==0).sum())
    print(f"\n{diag_pos} vs NORM: total={len(sub2)}, pos={n_pos} ({100*n_pos/len(sub2):.1f}%), neg={n_neg}")

    fold = sub2["strat_fold"].to_numpy()
    tr_m = fold <= 6; va_m = (fold==7)|(fold==8); te_m = (fold==9)|(fold==10)

    y_all2 = sub2["y_tmp"].to_numpy()
    X_tr, X_va, X_te = sub2_X[tr_m], sub2_X[va_m], sub2_X[te_m]
    y_tr, y_va, y_te = y_all2[tr_m], y_all2[va_m], y_all2[te_m]

    # Sex framing: male=0 (adv), female=1 (disadv)
    a_sex = sub2["sex"].to_numpy()
    a_tr_s, a_va_s, a_te_s = a_sex[tr_m], a_sex[va_m], a_sex[te_m]
    print(f"  Sex framing — female_pos_total={int(((sub2['sex']==1)&(sub2['y_tmp']==1)).sum())}")

    # Age framing: young<50=0 (disadv), old>=60=1 (adv)
    sub2["a_age"] = -1
    sub2.loc[sub2["age"] <  50, "a_age"] = 0
    sub2.loc[sub2["age"] >= 60, "a_age"] = 1
    age_valid = sub2["a_age"] >= 0
    sub2_age = sub2[age_valid].copy().reset_index(drop=True)
    sub2_age_X = sub2_X[age_valid.values]
    fold_age = sub2_age["strat_fold"].to_numpy()
    tr_a = fold_age<=6; va_a=(fold_age==7)|(fold_age==8); te_a=(fold_age==9)|(fold_age==10)
    y_age = sub2_age["y_tmp"].to_numpy()
    a_age = sub2_age["a_age"].to_numpy()
    print(f"  Age framing — young_pos_total={int(((sub2_age['a_age']==0)&(sub2_age['y_tmp']==1)).sum())}"
          f"  old_pos_total={int(((sub2_age['a_age']==1)&(sub2_age['y_tmp']==1)).sum())}")

    # Find target bias_pct (~DA+~43-100 for sex; just check natural for age)
    for bias in [0.04, 0.02]:
        check_framing(
            f"{diag_pos} vs NORM | sex-based | bias_pct={bias}",
            X_tr, y_tr, a_tr_s, X_va, y_va, a_va_s, X_te, y_te, a_te_s,
            disadv_code=1, bias_pct=bias, n_pca=25)

    check_framing(
        f"{diag_pos} vs NORM | age-based | bias_pct=0.04",
        sub2_age_X[tr_a], y_age[tr_a], a_age[tr_a],
        sub2_age_X[va_a], y_age[va_a], a_age[va_a],
        sub2_age_X[te_a], y_age[te_a], a_age[te_a],
        disadv_code=0, bias_pct=0.04, n_pca=25)


# ==============================================================
# MEPS ANALYSIS
# ==============================================================
print("\n\n" + "=" * 70)
print("MEPS: Testing alternative framings")
print("=" * 70)

from dataset import Dataset

def meps_framing(label, dp_col, minority_id, bias_pct, real_data_size=3000):
    try:
        ds = Dataset("meps", multiclass=False, minority_id=minority_id, majority_id=1-minority_id,
                     third_id=None, seed=SEED, device=DEVICE, use_pca=True)
        X_tr, X_va, X_te, y_tr, y_va, y_te = ds.get_data_splits(
            train_size=real_data_size, bias_pct=bias_pct, pca_components=25, dp_protected_col=dp_col)

        a_tr = ds.a_train.numpy(); a_va = ds.a_val.numpy(); a_te = ds.a_test.numpy()
        disadv = minority_id

        val_pos  = int(np.sum((y_va.numpy()==1) & (a_va==disadv)))
        test_pos = int(np.sum((y_te.numpy()==1) & (a_te==disadv)))
        da_plus  = int(np.sum((y_tr.numpy()==1) & (a_tr==disadv)))

        alpha = train_ffnn(X_tr.numpy(), y_tr.numpy())
        p1 = p1_from_agent(alpha, X_te).numpy()
        eo, t0, t1 = eo_gap(a_te, y_te.numpy(), p1)

        # Discriminant cosine on PCA test set
        cos = cosine_discriminant(X_te.numpy(), y_te.numpy(), a_te)

        ok1 = "✓" if val_pos  >= 40  else "✗"
        ok2 = "✓" if test_pos >= 200 else "✗"
        ok3 = "✓" if eo       >= 0.25 else "✗"
        ok4 = "✓" if cos      <  0.80 else "✗"

        print(f"\n  {label}")
        print(f"    bias_pct={bias_pct}  DA+={da_plus}")
        print(f"    {ok1} val_disadv_pos={val_pos:4d} (need>=40)")
        print(f"    {ok2} test_disadv_pos={test_pos:4d} (need>=200)")
        print(f"    {ok3} alpha_EO={eo:.3f} (need>=0.25)  disadv_TPR={t0:.3f}  adv_TPR={t1:.3f}")
        print(f"    {ok4} discriminant_cosine={cos:.3f} (need<0.80)")
        passes = sum([val_pos>=40, test_pos>=200, eo>=0.25, cos<0.80])
        print(f"    PASS: {passes}/4 criteria")
        return passes
    except Exception as e:
        print(f"\n  {label}: ERROR — {e}")
        return 0

# Ethnicity (Hispanic=0 vs non-Hispanic=1) — current framing
meps_framing("Ethnicity: Hispanic(0) vs non-Hisp(1) | bias_pct=0.08", "ethnicity", 0, 0.08)
meps_framing("Ethnicity: Hispanic(0) vs non-Hisp(1) | bias_pct=0.05", "ethnicity", 0, 0.05)

# Race (Black=0 vs White=1)
meps_framing("Race: Black(0) vs White(1) | bias_pct=0.08", "race", 0, 0.08)
meps_framing("Race: Black(0) vs White(1) | bias_pct=0.05", "race", 0, 0.05)

print("\n\nDone.")
