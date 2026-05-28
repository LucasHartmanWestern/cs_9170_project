#!/usr/bin/env python3
"""
census_fold_preflight.py

For each candidate k-fold split strategy on census_income, trains alpha on
each fold's training split and reports α-EO + α-WGL on both val and test.
Val and test share the same held-out fold (stratified 40/60 split by y×a).

Goal: confirm that for each fold rotation:
  1. α-EO > EO_THRESHOLD on val and test (meaningful signal exists)
  2. val α-EO ≈ test α-EO (val-test alignment, so RL reward targets the
     same fairness problem as the final evaluation)
  3. WGL is dominated by the same group that drives EO (female, a=0)

Usage:
    python census_fold_preflight.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split

DATA_PATH    = "datasets/census+income/adult.data"
DA_PCT       = 0.01433    # → DA+ ≈ 43 in a 3000-sample training set
TRAIN_SIZE   = 3000
PCA_COMP     = 10
FFNN_EPOCHS  = 30
BATCH_SIZE   = 64
LR           = 0.001
HIDDEN       = [32, 16]
VAL_FRAC     = 0.4        # fraction of held-out fold → val; remainder → test
SEED         = 42
EO_THRESHOLD = 0.05
ALIGN_TOL    = 0.10       # max |val_EO − test_EO| to declare "aligned"

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]
CAT_COLS = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "native-country",
]
NUM_COLS = [
    "age", "fnlwgt", "education-num", "capital-gain",
    "capital-loss", "hours-per-week",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_census():
    """Return (X_df, y, a) with raw (unencoded) X_df, no income/sex cols."""
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES,
                     na_values="?", skipinitialspace=True).dropna()
    y = np.where(df["income"].isin([">50K", ">50K."]), 1, 0).astype(int)
    a = (df["sex"].str.strip().str.lower() == "male").astype(int).to_numpy()
    X = df.drop(columns=["income", "sex"]).reset_index(drop=True)
    return X, y, a


# ---------------------------------------------------------------------------
# Scarcity injection
# ---------------------------------------------------------------------------

def apply_da_pct(X_df, y, a, da_pct, n_total, rng_seed):
    """
    Keep exactly round(da_pct * n_total) disadvantaged-group positives
    (female, a=0). Keep all advantaged positives and all negatives, then
    subsample the combined remainder to reach n_total total rows.
    """
    rng = np.random.RandomState(rng_seed)
    target_da_plus = max(1, round(da_pct * n_total))

    disadv_pos = np.where((a == 0) & (y == 1))[0]
    rest       = np.where(~((a == 0) & (y == 1)))[0]

    keep_n = min(len(disadv_pos), target_da_plus)
    da_idx = rng.choice(disadv_pos, keep_n, replace=False)

    n_rest = n_total - keep_n
    if len(rest) >= n_rest:
        rest_idx = rng.choice(rest, n_rest, replace=False)
    else:
        rest_idx = rest

    idx = np.sort(np.concatenate([da_idx, rest_idx]))
    return X_df.iloc[idx].reset_index(drop=True), y[idx], a[idx]


# ---------------------------------------------------------------------------
# Feature encoding (fit on train only)
# ---------------------------------------------------------------------------

def encode_splits(X_tr, X_va, X_te):
    """Fit OneHotEncoder + StandardScaler on train; transform all three."""
    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
    scaler = StandardScaler()
    pca    = PCA(n_components=PCA_COMP, random_state=SEED)

    X_tr_cat = enc.fit_transform(X_tr[CAT_COLS])
    X_tr_num = scaler.fit_transform(X_tr[NUM_COLS])
    X_tr_enc = np.hstack([X_tr_num, X_tr_cat])
    X_tr_pca = pca.fit_transform(X_tr_enc)

    def _transform(X_df):
        cat = enc.transform(X_df[CAT_COLS])
        num = scaler.transform(X_df[NUM_COLS])
        return pca.transform(np.hstack([num, cat]))

    return X_tr_pca, _transform(X_va), _transform(X_te)


# ---------------------------------------------------------------------------
# Alpha model
# ---------------------------------------------------------------------------

class SimpleFFNN(nn.Module):
    def __init__(self, input_dim, hidden, n_classes=2):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_and_eval(X_tr_p, y_tr, X_va_p, y_va, a_va, X_te_p, y_te, a_te):
    torch.manual_seed(SEED)
    model = SimpleFFNN(PCA_COMP, HIDDEN)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    ce    = nn.CrossEntropyLoss()
    Xt    = torch.FloatTensor(X_tr_p)
    yt    = torch.LongTensor(y_tr)

    for _ in range(FFNN_EPOCHS):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            ce(model(Xt[idx]), yt[idx]).backward()
            opt.step()

    model.eval()
    eps = 1e-7

    def eval_split(X_p, y, a):
        with torch.no_grad():
            probs = torch.softmax(model(torch.FloatTensor(X_p)), dim=1)[:, 1].numpy()
        preds = (probs >= 0.5).astype(int)

        def tpr(mask):
            pos = y[mask] == 1
            return float(preds[mask][pos].mean()) if pos.sum() > 0 else 0.0

        def bce(mask):
            p = np.clip(probs[mask], eps, 1 - eps)
            l = y[mask]
            return float(-np.mean(l * np.log(p) + (1 - l) * np.log(1 - p)))

        tpr_f, tpr_m = tpr(a == 0), tpr(a == 1)
        bce_f, bce_m = bce(a == 0), bce(a == 1)
        eo   = abs(tpr_f - tpr_m)
        wgl  = max(bce_f, bce_m)
        # WGL group matches EO group if both point to the same disadvantaged side
        wgl_group = 0 if bce_f >= bce_m else 1   # group with higher loss
        eo_group  = 0 if tpr_f <= tpr_m  else 1   # group with lower TPR
        return dict(eo=eo, wgl=wgl, tpr_f=tpr_f, tpr_m=tpr_m,
                    bce_f=bce_f, bce_m=bce_m,
                    wgl_group=wgl_group, eo_group=eo_group,
                    wgl_aligned=(wgl_group == eo_group))

    val_r  = eval_split(X_va_p, y_va, a_va)
    test_r = eval_split(X_te_p, y_te, a_te)
    return val_r, test_r


# ---------------------------------------------------------------------------
# Preflight runner
# ---------------------------------------------------------------------------

def run_preflight(label, fold_indices_list, X, y, a):
    """
    fold_indices_list: list of k arrays, each containing row indices for
    one fold. The fold at position fi is held-out when fi == rotation index.
    """
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    val_eos, test_eos, wgl_aligned_all = [], [], []
    fold_ok = []

    for fi, held_idx in enumerate(fold_indices_list):
        rng = np.random.RandomState(SEED + fi)
        train_idx = np.concatenate([fold_indices_list[fj]
                                    for fj in range(len(fold_indices_list))
                                    if fj != fi])

        X_held, y_held, a_held = X.iloc[held_idx], y[held_idx], a[held_idx]

        # Stratified val/test split within held-out fold (by y×a)
        strat = y_held * 2 + a_held
        try:
            idx_va, idx_te = train_test_split(
                np.arange(len(held_idx)),
                train_size=max(1, int(len(held_idx) * VAL_FRAC)),
                stratify=strat, random_state=SEED + fi,
            )
        except ValueError:
            n_va = max(1, int(len(held_idx) * VAL_FRAC))
            idx_va, idx_te = np.arange(n_va), np.arange(n_va, len(held_idx))

        X_va = X_held.iloc[idx_va].reset_index(drop=True)
        y_va = y_held[idx_va]
        a_va = a_held[idx_va]
        X_te = X_held.iloc[idx_te].reset_index(drop=True)
        y_te = y_held[idx_te]
        a_te = a_held[idx_te]

        X_tr_raw = X.iloc[train_idx]
        y_tr_raw = y[train_idx]
        a_tr_raw = a[train_idx]
        X_tr_df, y_tr, a_tr = apply_da_pct(
            X_tr_raw.reset_index(drop=True),
            y_tr_raw, a_tr_raw,
            DA_PCT, TRAIN_SIZE, SEED + fi,
        )

        da_plus   = int(((a_tr == 0) & (y_tr == 1)).sum())
        n_va_fpos = int(((a_va == 0) & (y_va == 1)).sum())
        n_te_fpos = int(((a_te == 0) & (y_te == 1)).sum())

        X_tr_p, X_va_p, X_te_p = encode_splits(X_tr_df, X_va, X_te)

        val_r, test_r = train_and_eval(
            X_tr_p, y_tr, X_va_p, y_va, a_va, X_te_p, y_te, a_te
        )

        v_eo, t_eo   = val_r["eo"], test_r["eo"]
        align_delta  = abs(v_eo - t_eo)
        wgl_ok_v     = val_r["wgl_aligned"]
        wgl_ok_t     = test_r["wgl_aligned"]

        ok_eo_v  = v_eo > EO_THRESHOLD
        ok_eo_t  = t_eo > EO_THRESHOLD
        ok_align = align_delta <= ALIGN_TOL
        ok_wgl   = wgl_ok_v and wgl_ok_t

        status_parts = []
        if not ok_eo_v:
            status_parts.append("LOW-EO-val")
        if not ok_eo_t:
            status_parts.append("LOW-EO-test")
        if not ok_align:
            status_parts.append(f"MISALIGNED(Δ={align_delta:.3f})")
        if not ok_wgl:
            status_parts.append("WGL-MISMATCH")
        status = "OK" if not status_parts else " | ".join(status_parts)

        print(f"  Fold {fi}: "
              f"val_EO={v_eo:.4f}  test_EO={t_eo:.4f}  Δ={align_delta:.4f}  "
              f"WGL_v={val_r['wgl']:.3f}(g{val_r['wgl_group']})  "
              f"WGL_t={test_r['wgl']:.3f}(g{test_r['wgl_group']})  "
              f"DA+={da_plus}  val_F+={n_va_fpos}  test_F+={n_te_fpos}  "
              f"[{status}]")

        val_eos.append(v_eo)
        test_eos.append(t_eo)
        wgl_aligned_all.append(ok_wgl)
        fold_ok.append(ok_eo_v and ok_eo_t and ok_align and ok_wgl)

    all_ok = all(fold_ok)
    print(f"\n  --> min_val_EO={min(val_eos):.4f}  min_test_EO={min(test_eos):.4f}  "
          f"max_Δ={max(abs(v-t) for v,t in zip(val_eos, test_eos)):.4f}  "
          f"WGL_aligned={sum(wgl_aligned_all)}/{len(wgl_aligned_all)}  "
          f"VIABLE={'YES ***' if all_ok else 'no'}")
    return val_eos, test_eos, fold_ok


# ---------------------------------------------------------------------------
# Fold builders
# ---------------------------------------------------------------------------

def make_skf_folds(k, y, a, rng_seed=None):
    """Stratified k-fold by y×a. rng_seed=None → use SEED."""
    strat = y * 2 + a
    skf   = StratifiedKFold(n_splits=k, shuffle=True,
                             random_state=rng_seed if rng_seed is not None else SEED)
    return [te_idx for _, te_idx in skf.split(np.zeros(len(y)), strat)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading census_income...")
    X, y, a = load_census()
    n = len(y)
    n_fpos = int(((a == 0) & (y == 1)).sum())
    print(f"Loaded: {n:,} rows  female_pos={n_fpos}  pos_rate_F={n_fpos/max((a==0).sum(),1):.3f}")
    print(f"Thresholds: α-EO > {EO_THRESHOLD}, val-test Δ ≤ {ALIGN_TOL}")

    results = {}

    # ---- k=3 ----
    for seed in [SEED, 0, 1, 7, 13, 99]:
        label = f"k=3  seed={seed}"
        folds = make_skf_folds(3, y, a, rng_seed=seed)
        val_eos, test_eos, fok = run_preflight(label, folds, X, y, a)
        results[label] = (val_eos, test_eos, fok)

    # ---- k=5 ----
    for seed in [SEED, 0, 1, 7]:
        label = f"k=5  seed={seed}"
        folds = make_skf_folds(5, y, a, rng_seed=seed)
        val_eos, test_eos, fok = run_preflight(label, folds, X, y, a)
        results[label] = (val_eos, test_eos, fok)

    # ---- Summary ----
    print("\n" + "=" * 90)
    print("SUMMARY (sorted by min val α-EO)")
    print("=" * 90)
    hdr = f"{'Strategy':<28}  {'val α-EO per fold':>36}  {'min_val':>7}  {'max_Δ':>7}  {'VIABLE':>7}"
    print(hdr)
    print("-" * 90)
    sorted_r = sorted(results.items(),
                      key=lambda kv: min(kv[1][0]), reverse=True)
    for label, (val_eos, test_eos, fok) in sorted_r:
        eo_str   = "  ".join(f"{e:.3f}" for e in val_eos)
        max_delta = max(abs(v - t) for v, t in zip(val_eos, test_eos))
        viable   = all(fok)
        print(f"  {label:<28} [{eo_str}]  {min(val_eos):.3f}  {max_delta:.3f}  "
              f"{'YES ***' if viable else ''}")
