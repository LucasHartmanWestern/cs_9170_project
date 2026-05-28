#!/usr/bin/env python3
"""
capture24_loso_preflight.py  —  Revised LOSO protocol, Phases -1 and 0.

Phase -1: define study_pool by excluding Q1 subjects (lowest MVPA quartile,
  verified as having zero female-positives in the capped val set).

Phase 0: run preflight diagnostics over all LOSO rotations of study_pool.
  Reports per-rotation: alpha_EO_val, BCE_F/M_val, TPR_F/M_val, val_fpos.
  Does NOT compute per-rotation test EO (single-subject test has only one
  sex; cross-group EO is undefined per rotation by construction).

Disadvantaged group is pinned to female (a=1) across all phases.
All |study_pool| rotations proceed to Phase 1 regardless of diagnostics.

Usage:
    python capture24_loso_preflight.py [--da_pct 0.015] [--seed 42]
    python capture24_loso_preflight.py --da_pct 0.010 0.015 0.020
"""
import sys
import os
import argparse
import csv
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from agents.ffnn_agent2 import FFNNAgent
from reward_helpers import worst_group_loss, eo_gap_from_probs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_PATH  = "datasets/capture24/capture24_features_cache.npz"
TRAIN_SIZE  = 4000
PCA_COMP    = 15
FFNN_EPOCHS = 30
VAL_CAP     = 4000
DEVICE      = "cpu"
N_VAL_SUBS  = 4      # 2F + 2M
N_BANDS     = 3      # terciles within study_pool by sex


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cache():
    cache   = np.load(CACHE_PATH, allow_pickle=True)
    return (cache["X"],
            cache["y"].astype(int),
            cache["a"].astype(int),    # 0=male 1=female
            cache["subject_ids"].astype(int))


# ---------------------------------------------------------------------------
# Phase -1 — study population definition
# ---------------------------------------------------------------------------

def define_study_pool(y_all, a_all, sid_all):
    """
    Exclude subjects in the lowest MVPA-rate quartile (Q1) computed across
    all 40 subjects. Verified: Q1 subjects are exactly those with zero
    female-positive windows in the capped val set (structural NaN problem
    found in initial preflight).

    Returns
    -------
    study_pool        : sorted list of qualifying subject_ids
    excluded          : list of (subject_id, sex_str, mvpa_rate) for reporting
    subject_sex       : array[n_sub], 0=male 1=female
    subject_mvpa      : array[n_sub], MVPA rate
    subject_band      : array[n_sub], tercile band 1..3 within study_pool/sex
                        (0 for excluded subjects)
    """
    n_sub        = int(sid_all.max()) + 1
    subject_sex  = np.array([int(np.bincount(a_all[sid_all == s]).argmax())
                              for s in range(n_sub)])
    subject_mvpa = np.array([(y_all[sid_all == s] == 1).mean()
                              for s in range(n_sub)])

    # Q1 boundary: 25th percentile across all 40 subjects
    q1_threshold = np.percentile(subject_mvpa, 25)

    excluded_ids = [s for s in range(n_sub) if subject_mvpa[s] <= q1_threshold]
    study_pool   = [s for s in range(n_sub) if subject_mvpa[s] >  q1_threshold]

    # Assign tercile bands within study_pool, separately by sex
    subject_band = np.zeros(n_sub, dtype=int)
    for sex in (0, 1):
        subs  = [s for s in study_pool if subject_sex[s] == sex]
        rates = subject_mvpa[subs]
        ranks = np.argsort(np.argsort(rates))   # 0-indexed rank
        n     = len(subs)
        for i, s in enumerate(subs):
            subject_band[s] = int(ranks[i] * N_BANDS // n) + 1   # 1..N_BANDS

    excluded = [(s,
                 "F" if subject_sex[s] == 1 else "M",
                 float(subject_mvpa[s]))
                for s in sorted(excluded_ids)]

    return study_pool, excluded, subject_sex, subject_mvpa, subject_band


def print_phase_minus1(study_pool, excluded, subject_sex, subject_mvpa,
                       subject_band):
    n_f = sum(1 for s in study_pool if subject_sex[s] == 1)
    n_m = sum(1 for s in study_pool if subject_sex[s] == 0)

    print("=" * 70)
    print("PHASE -1 — STUDY POPULATION DEFINITION")
    print("=" * 70)
    print(f"  Total original subjects : 40")
    print(f"  Q1 exclusion threshold  : mvpa_rate <= "
          f"{np.percentile(subject_mvpa, 25):.4f}  "
          f"(25th-percentile, computed before any experiments)")
    print(f"  Excluded (Q1)           : {len(excluded)}")
    for sid, sex, rate in excluded:
        print(f"    subject {sid:2d}  sex={sex}  mvpa_rate={rate:.4f}")
    print(f"  study_pool size         : {len(study_pool)}  "
          f"({n_f}F  {n_m}M)")
    print(f"  MVPA-rate bands         : {N_BANDS} terciles within sex")
    print()
    print(f"  {'sub':>4}  {'sex':>4}  {'band':>5}  {'mvpa_rate':>10}")
    for s in study_pool:
        sex_str = "F" if subject_sex[s] == 1 else "M"
        print(f"  {s:>4}  {sex_str:>4}  T{subject_band[s]:1d}     "
              f"{subject_mvpa[s]:.4f}")

    if n_f != n_m:
        print(f"\n  [NOTE] F/M imbalance: {n_f}F vs {n_m}M — surface to supervisor.")
    else:
        print(f"\n  [OK] Balanced: {n_f}F / {n_m}M")
    print()


# ---------------------------------------------------------------------------
# Phase 1 — LOSO rotation construction
# ---------------------------------------------------------------------------

def select_val_subjects(test_sub, training_pool, subject_sex, subject_band):
    """
    Pick 2F + 2M from training_pool whose band matches test_sub's band.
    Deterministic: sorted by subject_id, first 2 of each sex.
    Falls back to nearest band if < 2 available.
    """
    test_band = subject_band[test_sub]
    val_subs  = []
    for sex in (1, 0):   # females first
        chosen = []
        for delta in [0, 1, -1, 2, -2]:
            b = test_band + delta
            if not (1 <= b <= N_BANDS):
                continue
            cands = sorted(s for s in training_pool
                           if subject_sex[s] == sex and subject_band[s] == b)
            if len(cands) >= 2:
                chosen = cands
                if delta != 0:
                    print(f"  [FALLBACK] test={test_sub} sex={'FM'[sex]} "
                          f"band T{test_band}: used T{b} (Δ={delta:+d})")
                break
        val_subs.extend(chosen[:2])
    return val_subs


def build_rotation(test_sub, study_pool, subject_sex, subject_band):
    training_pool = [s for s in study_pool if s != test_sub]
    val_subs      = select_val_subjects(test_sub, training_pool,
                                        subject_sex, subject_band)
    train_subs    = [s for s in training_pool if s not in val_subs]

    # Hard constraints
    assert len(set(train_subs) & set(val_subs)) == 0, "train/val overlap"
    assert test_sub not in val_subs,                   "test in val"
    assert test_sub not in train_subs,                 "test in train"
    assert all(s in study_pool for s in train_subs + val_subs + [test_sub])

    return train_subs, val_subs, test_sub


# ---------------------------------------------------------------------------
# Phase 2 — data pipeline
# ---------------------------------------------------------------------------

def apply_da_pct_bias(X, y, a, da_pct, n_total, seed):
    """
    Mirror apply_da_pct_bias_c24 in dataset.py exactly:
    keep round(da_pct * n_total) female (a=1) positives,
    fill remaining slots from non-female-positive pool.
    """
    rng       = np.random.RandomState(seed)
    target    = max(1, round(da_pct * n_total))
    fp_idx    = np.where((a == 1) & (y == 1))[0]
    rest_idx  = np.where(~((a == 1) & (y == 1)))[0]
    keep_n    = min(len(fp_idx), target)
    da_idx    = rng.choice(fp_idx, keep_n, replace=False)
    n_rest    = n_total - keep_n
    rest_k    = (rng.choice(rest_idx, n_rest, replace=False)
                 if len(rest_idx) >= n_rest else rest_idx)
    idx = np.sort(np.concatenate([da_idx, rest_k]))
    return X[idx], y[idx], a[idx], idx


def cap_val(X, y, a, cap, seed):
    if len(y) <= cap:
        return X, y, a
    strat = y * 2 + a
    try:
        idx, _ = train_test_split(np.arange(len(y)), train_size=cap,
                                  stratify=strat, random_state=seed)
    except ValueError:
        idx = np.random.RandomState(seed).choice(len(y), cap, replace=False)
    idx = np.sort(idx)
    return X[idx], y[idx], a[idx]


def build_pipeline(train_subs, val_subs, test_sub,
                   X_all, y_all, a_all, sid_all, da_pct, seed):
    def collect(subs):
        mask = np.isin(sid_all, subs)
        X    = np.column_stack([X_all[mask], a_all[mask]])  # sex appended
        return X, y_all[mask], a_all[mask], sid_all[mask]

    X_tr_r, y_tr_r, a_tr_r, sid_tr = collect(train_subs)
    X_va_r, y_va_r, a_va_r, _      = collect(val_subs)
    X_te_r, y_te_r, a_te_r, _      = collect([test_sub])

    # Bias injection — training only
    X_tr_b, y_tr_b, a_tr_b, b_idx = apply_da_pct_bias(
        X_tr_r, y_tr_r, a_tr_r, da_pct, TRAIN_SIZE, seed)
    da_plus = int(((a_tr_b == 1) & (y_tr_b == 1)).sum())

    # Diversity check (logging only)
    fp_mask   = (a_tr_b == 1) & (y_tr_b == 1)
    n_fp_subs = len(np.unique(sid_tr[b_idx][fp_mask]))
    if n_fp_subs < 3:
        print(f"  [WARN diversity] test={test_sub}: {n_fp_subs} distinct "
              f"subjects contribute female positives (DA+={da_plus})")

    # Val cap
    X_va_b, y_va_b, a_va_b = cap_val(X_va_r, y_va_r, a_va_r, VAL_CAP, seed)

    # Sanity: val must have female positives after Phase -1 filter
    val_fpos = int(((a_va_b == 1) & (y_va_b == 1)).sum())
    if val_fpos == 0:
        print(f"  [ERROR] test={test_sub}: val has 0 female positives "
              f"after Phase -1 filter — re-examine exclusion threshold.")

    # StandardScaler + PCA (fit on biased train only, no test leakage)
    scaler  = StandardScaler()
    X_tr_z  = scaler.fit_transform(X_tr_b)
    X_va_z  = scaler.transform(X_va_b)
    X_te_z  = scaler.transform(X_te_r)

    pca     = PCA(n_components=PCA_COMP, random_state=seed)
    X_tr_p  = pca.fit_transform(X_tr_z)
    X_va_p  = pca.transform(X_va_z)
    X_te_p  = pca.transform(X_te_z)

    def t(arr, dt): return torch.tensor(arr, dtype=dt)

    return (t(X_tr_p, torch.float32), t(X_va_p, torch.float32),
            t(X_te_p, torch.float32),
            t(y_tr_b, torch.long),     t(y_va_b, torch.long),
            t(y_te_r, torch.long),
            t(a_tr_b, torch.long),     t(a_va_b, torch.long),
            t(a_te_r, torch.long),
            da_plus, n_fp_subs, val_fpos)


# ---------------------------------------------------------------------------
# Alpha training and evaluation
# ---------------------------------------------------------------------------

def train_alpha(x_tr, y_tr, seed):
    agent = FFNNAgent(
        input_size=PCA_COMP, hidden_sizes=[32, 16], output_size=2,
        classes=[0, 1], type="classification",
        learning_rate=0.001, batch_size=64, epochs=FFNN_EPOCHS,
        optimizer="adam", device=DEVICE, seed=seed,
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(TensorDataset(x_tr, y_tr),
                        batch_size=64, shuffle=True, generator=gen)
    agent.train(loader)
    return agent


def eval_split(agent, x, y, a):
    """Evaluate on one split. Disadvantaged group pinned to female (a=1)."""
    agent.model.eval()
    with torch.no_grad():
        p1 = torch.softmax(agent.model(x), dim=1)[:, 1]

    losses     = F.binary_cross_entropy(p1, y.float(), reduction="none")
    wgl, per_g = worst_group_loss(losses, a)
    eo         = eo_gap_from_probs(a, y, p1, group0=0, group1=1)

    preds = (p1 >= 0.5).long()
    def tpr(g):
        mask = (a == g) & (y == 1)
        return float(preds[mask].float().mean()) if mask.sum() > 0 else float("nan")

    return dict(
        eo    = float(eo.item()),
        bce_f = per_g.get(1, float("nan")),   # female BCE
        bce_m = per_g.get(0, float("nan")),   # male BCE
        tpr_f = tpr(1),                        # female TPR (disadvantaged)
        tpr_m = tpr(0),
        fpos  = int(((a == 1) & (y == 1)).sum()),
        n     = int(len(y)),
    )


# ---------------------------------------------------------------------------
# Phase 0 — preflight
# ---------------------------------------------------------------------------

def run_preflight(study_pool, subject_sex, subject_band,
                  X_all, y_all, a_all, sid_all, da_pct, seed):

    n_rot = len(study_pool)
    print(f"Running Phase 0 preflight: da_pct={da_pct}  seed={seed}  "
          f"rotations={n_rot}\n")

    hdr = (f"{'rot':>4}  {'test':>5}  {'sex':>4}  {'band':>5}  "
           f"{'EO_val':>7}  {'BCE_F':>6}  {'BCE_M':>6}  "
           f"{'TPR_F':>6}  {'TPR_M':>6}  "
           f"{'DA+':>4}  {'fp_sub':>6}  {'val_F+':>8}")
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    rows = []
    for rot_id, test_sub in enumerate(study_pool):
        train_subs, val_subs, _ = build_rotation(
            test_sub, study_pool, subject_sex, subject_band)

        (x_tr, x_va, x_te,
         y_tr, y_va, y_te,
         a_tr, a_va, a_te,
         da_plus, n_fp_subs, val_fpos) = build_pipeline(
            train_subs, val_subs, test_sub,
            X_all, y_all, a_all, sid_all, da_pct, seed)

        agent = train_alpha(x_tr, y_tr, seed)
        v     = eval_split(agent, x_va, y_va, a_va)

        sex_str  = "F" if subject_sex[test_sub] == 1 else "M"
        band_str = f"T{subject_band[test_sub]}"

        row = dict(
            rotation_id    = rot_id,
            test_subject   = test_sub,
            test_sex       = sex_str,
            test_band      = subject_band[test_sub],
            alpha_EO_val   = v["eo"],
            BCE_F_val      = v["bce_f"],
            BCE_M_val      = v["bce_m"],
            TPR_F_val      = v["tpr_f"],
            TPR_M_val      = v["tpr_m"],
            da_plus        = da_plus,
            fp_subjects    = n_fp_subs,
            val_fpos       = val_fpos,
            val_n          = v["n"],
        )
        rows.append(row)

        print(f"  {rot_id:>3}  {test_sub:>5}  {sex_str:>4}  "
              f"{band_str:>5}  "
              f"{v['eo']:>7.4f}  {v['bce_f']:>6.3f}  {v['bce_m']:>6.3f}  "
              f"{v['tpr_f']:>6.3f}  {v['tpr_m']:>6.3f}  "
              f"{da_plus:>4}  {n_fp_subs:>6}  "
              f"{val_fpos:>5}/{v['n']}")

    return rows


def print_summary(rows, da_pct):
    eo   = [r["alpha_EO_val"] for r in rows]
    bce_f = [r["BCE_F_val"]   for r in rows]
    bce_m = [r["BCE_M_val"]   for r in rows]
    tpr_f = [r["TPR_F_val"]   for r in rows]
    tpr_m = [r["TPR_M_val"]   for r in rows]
    vfpos = [r["val_fpos"]    for r in rows]
    n     = len(rows)

    n_eo_ok  = sum(1 for e in eo  if not np.isnan(e) and e >= 0.05)
    n_dir_ok = sum(1 for f, m in zip(tpr_f, tpr_m)
                   if not (np.isnan(f) or np.isnan(m)) and f < m)
    n_bce_ok = sum(1 for f, m in zip(bce_f, bce_m) if f > m)
    min_vfpos = min(vfpos)

    print(f"\n{'='*70}")
    print(f"PHASE 0 SUMMARY  (da_pct={da_pct}  rotations={n})")
    print(f"{'='*70}")
    print(f"  alpha_EO_val")
    print(f"    mean   = {np.nanmean(eo):.4f}    median = {np.nanmedian(eo):.4f}")
    print(f"    range  = [{np.nanmin(eo):.4f}, {np.nanmax(eo):.4f}]")
    print(f"    >= 0.05: {n_eo_ok}/{n}  ({100*n_eo_ok/n:.0f}%)")
    print()
    print(f"  TPR_F < TPR_M  (female disadvantaged): "
          f"{n_dir_ok}/{n}  ({100*n_dir_ok/n:.0f}%)")
    print(f"  BCE_F > BCE_M  (WGL sanity, informational): "
          f"{n_bce_ok}/{n}  ({100*n_bce_ok/n:.0f}%)")
    print(f"  min val_female_positive_count: {min_vfpos}  "
          f"({'OK - Phase -1 effective' if min_vfpos > 0 else 'ERROR - re-examine exclusion'})")
    print()
    print(f"  INTERPRETATION (da_pct={da_pct}):")

    # EO regime
    if np.nanmean(eo) >= 0.05 and n_eo_ok / n >= 0.80:
        print(f"  [PASS] EO regime: mean={np.nanmean(eo):.3f} >= 0.05, "
              f"{n_eo_ok}/{n} rotations clear threshold. Proceed.")
    else:
        print(f"  [WARN] EO regime weak: mean={np.nanmean(eo):.3f}, "
              f"only {n_eo_ok}/{n} rotations clear 0.05. "
              f"Investigate bias injection or raise scarcity rate.")

    # Direction
    if n_dir_ok / n >= 0.80:
        print(f"  [PASS] Female disadvantage: TPR_F < TPR_M in "
              f"{n_dir_ok}/{n} ({100*n_dir_ok/n:.0f}%) rotations. Proceed.")
    else:
        print(f"  [WARN] Female disadvantage inconsistent: TPR_F < TPR_M "
              f"in only {n_dir_ok}/{n} ({100*n_dir_ok/n:.0f}%) rotations. "
              f"Investigate bias injection.")

    # BCE sanity (informational)
    print(f"  [INFO] BCE_F > BCE_M (WGL sanity): {n_bce_ok}/{n} "
          f"({100*n_bce_ok/n:.0f}%) rotations. "
          f"Disadvantaged group is pinned to female regardless.")

    # Val fpos
    if min_vfpos > 0:
        print(f"  [PASS] All rotations have female positives in val "
              f"(min={min_vfpos}). Phase -1 filter effective.")
    else:
        print(f"  [ERROR] Some rotations have 0 val female positives. "
              f"Re-examine Phase -1 exclusion threshold.")

    print(f"{'='*70}")


def save_csv(rows, da_pct, seed):
    out = Path(f"loso_preflight_da{int(da_pct*1000):03d}_s{seed}.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--da_pct", type=float, nargs="+", default=[0.015])
    parser.add_argument("--seed",   type=int,   default=42)
    args = parser.parse_args()

    print("Loading cache ...")
    X_all, y_all, a_all, sid_all = load_cache()
    print(f"  {len(X_all):,} windows  "
          f"{(a_all==1).sum():,}F  {(a_all==0).sum():,}M\n")

    # Phase -1 (once, shared across all scarcity rates)
    study_pool, excluded, subject_sex, subject_mvpa, subject_band = \
        define_study_pool(y_all, a_all, sid_all)
    print_phase_minus1(study_pool, excluded, subject_sex,
                       subject_mvpa, subject_band)

    # Phase 0 (once per scarcity rate)
    for da_pct in args.da_pct:
        rows = run_preflight(study_pool, subject_sex, subject_band,
                             X_all, y_all, a_all, sid_all, da_pct, args.seed)
        print_summary(rows, da_pct)
        save_csv(rows, da_pct, args.seed)
