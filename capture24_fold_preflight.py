#!/usr/bin/env python3
"""
capture24_fold_preflight.py

Checks whether a k=5 fold assignment (fold_rng_seed) produces folds with
a viable α-EO signal and WGL correctly targeting female (a=1).

Uses Dataset.get_data_splits(), FFNNAgent, worst_group_loss, and
eo_gap_from_probs directly — the same infrastructure as the training loop —
so the α-EO and WGL values here are in the same environment as FORGE.

Usage:
    python capture24_fold_preflight.py --seeds 40 190
    python capture24_fold_preflight.py            # sweeps seeds 0-199
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from dataset import Dataset
from agents.ffnn_agent2 import FFNNAgent
from reward_helpers import worst_group_loss, eo_gap_from_probs

# ---------------------------------------------------------------------------
# Config — must match the EXP-046 fold specs exactly
# ---------------------------------------------------------------------------
DA_PCT         = 0.015
TRAIN_SIZE     = 4000
PCA_COMP       = 15
FFNN_EPOCHS    = 30
DATASET_SEED   = 42        # fixed seed for preflight; actual runs vary
K              = 5
KFOLD_VAL_FRAC = 0.4
WIN_SECONDS    = 1.0
STEP_SECONDS   = 0.5
DEVICE         = "cpu"

EO_THRESHOLD   = 0.05
ALIGN_TOL      = 0.15   # max |val_EO − test_EO| to pass alignment check


# ---------------------------------------------------------------------------
# Train alpha (same pattern as Training.train_predictor_model)
# ---------------------------------------------------------------------------

def train_alpha(x_tr, y_tr, seed):
    agent = FFNNAgent(
        input_size=PCA_COMP,
        hidden_sizes=[32, 16],
        output_size=2,
        classes=[0, 1],
        type="classification",
        learning_rate=0.001,
        batch_size=64,
        epochs=FFNN_EPOCHS,
        optimizer="adam",
        device=DEVICE,
        seed=seed,
    )
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_tr, y_tr),
        batch_size=64,
        shuffle=True,
        generator=gen,
    )
    agent.train(loader)
    return agent


# ---------------------------------------------------------------------------
# Evaluate a trained agent on any split — call for val and test separately
# ---------------------------------------------------------------------------

def eval_split(agent, x, y, a):
    """Returns EO, WGL, BCE per group, TPR per group, positive counts."""
    agent.model.eval()
    with torch.no_grad():
        logits = agent.model(x)
        p1     = torch.softmax(logits, dim=1)[:, 1]

    losses = F.binary_cross_entropy(p1, y.float(), reduction="none")
    wgl, per_g = worst_group_loss(losses, a)
    eo         = eo_gap_from_probs(a, y, p1, group0=0, group1=1)

    bce_m = per_g.get(0, float("nan"))
    bce_f = per_g.get(1, float("nan"))

    preds = (p1 >= 0.5).long()
    def tpr(g):
        mask = (a == g) & (y == 1)
        return float(preds[mask].float().mean()) if mask.sum() > 0 else 0.0

    return dict(
        eo    = float(eo.item()),
        wgl   = float(wgl.item()),
        bce_f = bce_f,  bce_m = bce_m,
        tpr_f = tpr(1), tpr_m = tpr(0),
        fpos  = int(((a == 1) & (y == 1)).sum()),
        f_tot = int((a == 1).sum()),
    )


# ---------------------------------------------------------------------------
# Run preflight for one fold_rng_seed
# ---------------------------------------------------------------------------

def run_preflight(fold_rng_seed):
    label = f"k={K}  rng_seed={fold_rng_seed}"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    fold_eos, fold_wgls = [], []
    fold_wgl_ok, fold_dir_ok, fold_align_ok = [], [], []

    for fi in range(K):
        ds = Dataset(
            dataset_name="capture24",
            seed=DATASET_SEED,
            device=DEVICE,
            multiclass=False,
            minority_id=1,
            majority_id=0,
            third_id=None,
            use_pca=True,
        )
        splits = ds.get_data_splits(
            train_size=TRAIN_SIZE,
            da_pct=DA_PCT,
            pca_components=PCA_COMP,
            win_seconds=WIN_SECONDS,
            step_seconds=STEP_SECONDS,
            fold_idx=fi,
            n_folds=K,
            kfold_val_frac=KFOLD_VAL_FRAC,
            fold_rng_seed=fold_rng_seed,
        )
        x_tr, x_val, x_test, y_tr, y_val, y_test = splits
        a_val  = ds.a_val
        a_test = ds.a_test

        da_plus = int(((ds.a_train == 1) & (y_tr == 1)).sum())

        agent = train_alpha(x_tr, y_tr, seed=DATASET_SEED)

        v = eval_split(agent, x_val,  y_val,  a_val)   # capped val — what reward sees
        t = eval_split(agent, x_test, y_test, a_test)   # full test — what paper reports

        wgl_dir_ok  = v["bce_f"] > v["bce_m"]          # WGL targets female
        eo_dir_ok   = v["tpr_f"] < v["tpr_m"]          # female is disadvantaged in val
        align_delta = abs(v["eo"] - t["eo"])
        align_ok    = align_delta <= ALIGN_TOL

        ok_eo = v["eo"] > EO_THRESHOLD

        issues = []
        if not ok_eo:
            issues.append("LOW-EO")
        if not wgl_dir_ok:
            issues.append("WGL-FLIP")
        if not eo_dir_ok:
            issues.append("DIR-FLIP")
        if not align_ok:
            issues.append(f"ALIGN({align_delta:.2f})")
        status = "OK" if not issues else " | ".join(issues)

        print(f"  Fold {fi}: "
              f"val_EO={v['eo']:.3f}  tst_EO={t['eo']:.3f}  Δ={align_delta:.3f}  "
              f"WGL={v['wgl']:.3f}  "
              f"[TPR F={v['tpr_f']:.3f} M={v['tpr_m']:.3f}]  "
              f"[BCE F={v['bce_f']:.2f} M={v['bce_m']:.2f}]  "
              f"DA+={da_plus}  val_F+={v['fpos']}/{v['f_tot']}  "
              f"[{status}]")

        fold_eos.append(v["eo"])
        fold_wgls.append(v["wgl"])
        fold_wgl_ok.append(wgl_dir_ok)
        fold_dir_ok.append(eo_dir_ok)
        fold_align_ok.append(align_ok)

    n_wgl   = sum(fold_wgl_ok)
    n_dir   = sum(fold_dir_ok)
    n_align = sum(fold_align_ok)
    viable  = (all(e > EO_THRESHOLD for e in fold_eos)
               and n_wgl == K and n_dir == K and n_align == K)
    print(f"\n  --> min_val_EO={min(fold_eos):.4f}  max WGL={max(fold_wgls):.3f}"
          f"  wgl={n_wgl}/{K}  dir={n_dir}/{K}  align={n_align}/{K}"
          f"  VIABLE={'YES ***' if viable else 'no'}")

    return fold_eos, fold_wgls, fold_wgl_ok, fold_dir_ok, fold_align_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Fold rng seeds to test (default: 0–199)")
    args = parser.parse_args()

    seed_range = args.seeds if args.seeds is not None else range(200)

    results = {}
    for s in seed_range:
        eos, wgls, wgl_ok, dir_ok, align_ok = run_preflight(s)
        results[s] = (eos, wgls, wgl_ok, dir_ok, align_ok)

    print("\n" + "=" * 105)
    print("SUMMARY (sorted by all-checks-passed desc, then min val_EO desc)")
    print("=" * 105)
    print(f"  {'seed':>6}  {'per-fold val_EO':>44}  {'min':>6}  {'maxWGL':>7}"
          f"  {'wgl':>5}  {'dir':>5}  {'aln':>5}  {'VIABLE':>7}")
    print("-" * 105)
    sorted_r = sorted(
        results.items(),
        key=lambda kv: (sum(kv[1][2]) + sum(kv[1][3]) + sum(kv[1][4]),
                        min(kv[1][0])),
        reverse=True,
    )
    for seed, (eos, wgls, wgl_ok, dir_ok, align_ok) in sorted_r:
        eo_str = "  ".join(f"{e:.3f}" for e in eos)
        n_wgl, n_dir, n_aln = sum(wgl_ok), sum(dir_ok), sum(align_ok)
        viable = (all(e > EO_THRESHOLD for e in eos)
                  and n_wgl == K and n_dir == K and n_aln == K)
        print(f"  {seed:>6}  [{eo_str}]  {min(eos):.3f}  {max(wgls):.3f}"
              f"  {n_wgl}/{K}  {n_dir}/{K}  {n_aln}/{K}"
              f"  {'YES ***' if viable else ''}")
