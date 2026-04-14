"""
dataset_viability.py — Unified dataset structural viability checker.

Runs three checks needed before committing to full RL experiments:
  1. DA+ scan: sweep bias_pct values (and seeds) to find a regime with DA+ ≈ 43-45
  2. Alpha-EO baseline: train FFNN alpha at target bias_pct(s), report fairness + utility
  3. Feature separability: centroid distance between group-conditional positives in PCA space

Pass/fail criteria (from CLAUDE.md):
  - val_disadv_pos  ≥ 30
  - test_disadv_pos ≥ 200
  - alpha_EO        ≥ 0.10
  - Feature separability: qualitative — group-positive centroids should be distinct

Usage:
    # da_pct mode (preferred — targets DA+ directly as % of train set):
    python dataset_viability.py \\
        --dataset compas \\
        --minority-id 0 --majority-id 1 \\
        --dp-col race \\
        --da-pcts 0.010 0.014 0.018 \\
        --seeds 0 2 3 42

    # Legacy bias_pct mode (subsamples all positives):
    python dataset_viability.py \\
        --dataset meps \\
        --minority-id 0 --majority-id 1 \\
        --dp-col ethnicity \\
        --bias-pcts 0.06 0.08 0.10 0.12

    # No-bias baseline (unbiased scan, no --da-pcts or --bias-pcts):
    python dataset_viability.py \\
        --dataset ptb_xl \\
        --minority-id 1 --majority-id 0 \\
        --dp-col sex
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))

from dataset import Dataset
from agents.ffnn_agent2 import FFNNAgent
from reward_helpers import fairness_classification_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Dataset structural viability checker")
    p.add_argument("--dataset",      required=True, help="Dataset name (e.g. compas, meps, ptb_xl)")
    p.add_argument("--minority-id",  type=int, required=True, help="Disadvantaged group ID (a=minority_id)")
    p.add_argument("--majority-id",  type=int, required=True, help="Advantaged group ID")
    p.add_argument("--dp-col",       default=None, help="Protected column name (dp_protected_col)")
    p.add_argument("--da-pcts",      nargs="+", type=float, default=None,
                   help="da_pct values to sweep (group-specific; DA+ = round(da_pct * train_size))")
    p.add_argument("--bias-pcts",    nargs="+", type=float, default=None,
                   help="Legacy bias_pct values to sweep (subsamples all positives)")
    p.add_argument("--seeds",        nargs="+", type=int, default=[42, 0, 1],
                   help="Seeds to run (default: 42 0 1)")
    p.add_argument("--real-data-size", type=int, default=3000, help="Training set size cap")
    p.add_argument("--pca-components", type=int, default=10)
    p.add_argument("--win-seconds",  type=float, default=1.0, help="capture24 windowing")
    p.add_argument("--step-seconds", type=float, default=0.5, help="capture24 step")
    p.add_argument("--device",       default="cpu")
    p.add_argument("--skip-separability", action="store_true",
                   help="Skip feature separability analysis (faster)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dataset(args, seed, bias_pct=None, da_pct=None):
    ds = Dataset(
        dataset_name=args.dataset,
        multiclass=False,
        minority_id=args.minority_id,
        majority_id=args.majority_id,
        third_id=None,
        pca_components=args.pca_components,
        seed=seed,
        device=args.device,
        use_pca=True,
    )
    kwargs = dict(
        train_size=args.real_data_size,
        bias_pct=bias_pct,
        da_pct=da_pct,
        pca_components=args.pca_components,
        drop_protected=False,
        protected_cols=ds.protected_attributes,
        win_seconds=args.win_seconds,
        step_seconds=args.step_seconds,
    )
    if args.dp_col is not None:
        kwargs["dp_protected_col"] = args.dp_col

    splits = ds.get_data_splits(**kwargs)
    x_train, x_val, x_test, y_train, y_val, y_test = splits
    return ds, x_train, x_val, x_test, y_train, y_val, y_test


def train_alpha(x_train, y_train, seed, args):
    agent = FFNNAgent(
        input_size=x_train.shape[1],
        hidden_sizes=[32, 16],
        output_size=1,
        learning_rate=0.001,
        batch_size=64,
        epochs=20,
        device=args.device,
        seed=seed,
    )
    loader = DataLoader(TensorDataset(x_train, y_train.float()), batch_size=64, shuffle=True)
    agent.train(loader)
    return agent


def get_probs(agent, x, device):
    agent.model.eval()
    with torch.no_grad():
        logits = agent.model(x.to(device)).squeeze(-1)
        return torch.sigmoid(logits)


def group_counts(a_tensor, y_tensor, minority_id):
    a = a_tensor.cpu().numpy() if torch.is_tensor(a_tensor) else np.array(a_tensor)
    y = y_tensor.cpu().numpy() if torch.is_tensor(y_tensor) else np.array(y_tensor)
    da_plus  = int(np.sum((a == minority_id) & (y == 1)))
    n_disadv = int(np.sum(a == minority_id))
    return da_plus, n_disadv


def eo_gap(a_tensor, y_tensor, probs, threshold=0.5):
    metrics = fairness_classification_metrics(a_tensor, y_tensor, probs, threshold=threshold)
    return float(metrics.get("eo_tpr_diff", float("nan")))


# ---------------------------------------------------------------------------
# Step 1: DA+ scan
# ---------------------------------------------------------------------------

def step1_da_scan(args):
    print("=" * 70)
    print("STEP 1: DA+ scan across scarcity values and seeds")
    print(f"  dataset={args.dataset}  minority_id={args.minority_id}  "
          f"majority_id={args.majority_id}  dp_col={args.dp_col}")
    print("=" * 70)
    print(f"{'param':>12}  {'value':>8}  {'seed':>5}  {'DA+':>5}  {'n_disadv':>9}  "
          f"{'val_disadv_pos':>14}  {'test_disadv_pos':>15}")
    print("-" * 80)

    # Build list of (param_name, param_value, kwargs) to sweep
    sweep = []
    if args.da_pcts:
        for v in args.da_pcts:
            sweep.append(("da_pct", v, {"da_pct": v}))
    if args.bias_pcts:
        for v in args.bias_pcts:
            sweep.append(("bias_pct", v, {"bias_pct": v}))
    if not sweep:
        sweep.append(("da_pct", None, {}))  # unbiased baseline

    scan_results = []
    for param_name, param_val, kw in sweep:
        for seed in args.seeds:
            try:
                ds, x_tr, x_val, x_te, y_tr, y_val, y_te = make_dataset(args, seed, **kw)
                da_plus, n_disadv = group_counts(ds.a_train, y_tr, args.minority_id)

                a_val_np = ds.a_val.cpu().numpy() if torch.is_tensor(ds.a_val) else np.array(ds.a_val)
                y_val_np = y_val.cpu().numpy()
                val_dp = int(np.sum((a_val_np == args.minority_id) & (y_val_np == 1)))

                a_te_np = ds.a_test.cpu().numpy() if torch.is_tensor(ds.a_test) else np.array(ds.a_test)
                y_te_np = y_te.cpu().numpy()
                test_dp = int(np.sum((a_te_np == args.minority_id) & (y_te_np == 1)))

                val_str = f"{param_val:.5f}" if param_val is not None else "   None"
                print(f"{param_name:>12}  {val_str:>8}  {seed:>5}  {da_plus:>5}  {n_disadv:>9}  "
                      f"{val_dp:>14}  {test_dp:>15}")

                scan_results.append({
                    param_name: param_val, "seed": seed,
                    "da_plus": da_plus, "n_disadv": n_disadv,
                    "val_disadv_pos": val_dp, "test_disadv_pos": test_dp,
                })
            except Exception as e:
                print(f"{param_name:>12}  {param_val!r:>8}  {seed:>5}  ERROR: {e}")

    print()
    return scan_results


# ---------------------------------------------------------------------------
# Step 2: Alpha-EO baseline
# ---------------------------------------------------------------------------

def step2_alpha_eo(args):
    print("=" * 70)
    print("STEP 2: Alpha-EO baseline")
    print(f"  da_pcts={args.da_pcts}  bias_pcts={args.bias_pcts}  seeds={args.seeds}")
    print("=" * 70)
    print(f"{'param':>12}  {'value':>8}  {'seed':>5}  {'DA+':>5}  {'val_EO':>7}  "
          f"{'test_EO':>8}  {'AUC':>7}  {'F1w':>7}")
    print("-" * 80)

    from sklearn.metrics import roc_auc_score, f1_score

    # Build same sweep list as step1
    sweep = []
    if args.da_pcts:
        for v in args.da_pcts:
            sweep.append(("da_pct", v, {"da_pct": v}))
    if args.bias_pcts:
        for v in args.bias_pcts:
            sweep.append(("bias_pct", v, {"bias_pct": v}))
    if not sweep:
        sweep.append(("da_pct", None, {}))  # unbiased baseline

    alpha_results = []
    for param_name, param_val, kw in sweep:
        for seed in args.seeds:
            try:
                ds, x_tr, x_val, x_te, y_tr, y_val, y_te = make_dataset(args, seed, **kw)
                da_plus, _ = group_counts(ds.a_train, y_tr, args.minority_id)

                alpha = train_alpha(x_tr, y_tr, seed, args)
                p_val  = get_probs(alpha, x_val,  args.device)
                p_test = get_probs(alpha, x_te, args.device)

                val_eo  = eo_gap(ds.a_val,  y_val, p_val)
                test_eo = eo_gap(ds.a_test, y_te,  p_test)

                y_te_np = y_te.cpu().numpy()
                p_np    = p_test.cpu().numpy()
                preds   = (p_np >= 0.5).astype(int)
                auc = roc_auc_score(y_te_np, p_np) if len(np.unique(y_te_np)) > 1 else float("nan")
                f1w = f1_score(y_te_np, preds, average="weighted", zero_division=0)

                val_str = f"{param_val:.5f}" if param_val is not None else "   None"
                print(f"{param_name:>12}  {val_str:>8}  {seed:>5}  {da_plus:>5}  {val_eo:>7.4f}  "
                      f"{test_eo:>8.4f}  {auc:>7.4f}  {f1w:>7.4f}")

                alpha_results.append({
                    param_name: param_val, "seed": seed,
                    "da_plus": da_plus, "val_eo": val_eo, "test_eo": test_eo,
                    "auc": auc, "f1w": f1w,
                })
            except Exception as e:
                val_str = f"{param_val:.5f}" if param_val is not None else "   None"
                print(f"{param_name:>12}  {val_str:>8}  {seed:>5}  ERROR: {e}")

    print()
    return alpha_results


# ---------------------------------------------------------------------------
# Step 3: Feature separability
# ---------------------------------------------------------------------------

def step3_separability(args):
    """
    In PCA space: compare the centroids of disadvantaged-positive vs
    advantaged-positive subgroups. A large ratio of between-group distance
    to within-group spread indicates the agent can place group-specific signal.
    """
    print("=" * 70)
    print("STEP 3: Feature separability (PCA space, seed=42, no bias)")
    print("  Checks if disadvantaged-positive and advantaged-positive")
    print("  subgroups are spatially distinct — prerequisite for RL signal.")
    print("=" * 70)

    seed = args.seeds[0]
    try:
        ds, x_tr, x_val, x_te, y_tr, y_val, y_te = make_dataset(args, seed)
    except Exception as e:
        print(f"  ERROR loading unbiased dataset: {e}\n")
        return

    x_np = x_te.cpu().numpy()
    y_np = y_te.cpu().numpy()
    a_np = ds.a_test.cpu().numpy() if torch.is_tensor(ds.a_test) else np.array(ds.a_test)

    disadv_pos = x_np[(y_np == 1) & (a_np == args.minority_id)]
    adv_pos    = x_np[(y_np == 1) & (a_np == args.majority_id)]
    disadv_neg = x_np[(y_np == 0) & (a_np == args.minority_id)]
    adv_neg    = x_np[(y_np == 0) & (a_np == args.majority_id)]

    print(f"  Test subgroup sizes:")
    print(f"    disadv_pos={len(disadv_pos)}  adv_pos={len(adv_pos)}  "
          f"disadv_neg={len(disadv_neg)}  adv_neg={len(adv_neg)}")

    if len(disadv_pos) < 2 or len(adv_pos) < 2:
        print("  Too few positive examples for separability analysis.\n")
        return

    c_disadv = disadv_pos.mean(axis=0)
    c_adv    = adv_pos.mean(axis=0)
    between  = float(np.linalg.norm(c_disadv - c_adv))
    within_d = float(disadv_pos.std(axis=0).mean())
    within_a = float(adv_pos.std(axis=0).mean())
    sep_ratio = between / max(within_d, within_a, 1e-8)

    print(f"\n  Centroid distance (disadv_pos vs adv_pos): {between:.4f}")
    print(f"  Within-group spread: disadv={within_d:.4f}  adv={within_a:.4f}")
    print(f"  Separability ratio (between/within):       {sep_ratio:.4f}")

    # Cosine similarity of discriminant vectors (pos_centroid - neg_centroid per group)
    if len(disadv_neg) > 0 and len(adv_neg) > 0:
        c_disadv_neg = disadv_neg.mean(axis=0)
        c_adv_neg    = adv_neg.mean(axis=0)
        disc_d = c_disadv - c_disadv_neg
        disc_a = c_adv    - c_adv_neg
        cos = float(np.dot(disc_d, disc_a) /
                    (np.linalg.norm(disc_d) * np.linalg.norm(disc_a) + 1e-8))
        print(f"  Cosine similarity of discriminant vectors: {cos:.4f}")
        print(f"    (1.0 = identical; <0.7 = group-specific structure)")

    print(f"\n  Interpretation:")
    if sep_ratio > 0.5:
        print(f"  sep_ratio={sep_ratio:.2f} — groups are spatially distinct. RL signal likely viable.")
    else:
        print(f"  sep_ratio={sep_ratio:.2f} — groups overlap heavily in PCA space. "
              f"RL may not be able to carry group-specific signal.")
    print()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(scan_results, alpha_results, args):
    print("=" * 70)
    print("SUMMARY — Pass/Fail against paper criteria")
    print("=" * 70)

    criteria = {
        "val_disadv_pos ≥ 30":  False,
        "test_disadv_pos ≥ 200": False,
        "alpha_EO ≥ 0.10":      False,
    }

    if scan_results:
        max_val  = max(r["val_disadv_pos"]  for r in scan_results)
        max_test = max(r["test_disadv_pos"] for r in scan_results)
        criteria["val_disadv_pos ≥ 30"]   = max_val  >= 30
        criteria["test_disadv_pos ≥ 200"] = max_test >= 200
        print(f"  Best val_disadv_pos:  {max_val}  {'PASS' if max_val >= 30 else 'FAIL'}")
        print(f"  Best test_disadv_pos: {max_test}  {'PASS' if max_test >= 200 else 'FAIL'}")

    if alpha_results:
        max_val_eo  = max(r["val_eo"]  for r in alpha_results if not np.isnan(r["val_eo"]))
        max_test_eo = max(r["test_eo"] for r in alpha_results if not np.isnan(r["test_eo"]))
        criteria["alpha_EO ≥ 0.10"] = max_val_eo >= 0.10
        print(f"  Best val alpha-EO:    {max_val_eo:.4f}  {'PASS' if max_val_eo >= 0.10 else 'FAIL'}")
        print(f"  Best test alpha-EO:   {max_test_eo:.4f}")

    print()
    all_pass = all(criteria.values())
    print(f"  Overall: {'ALL CRITERIA PASS — viable for RL experiments.' if all_pass else 'ONE OR MORE CRITERIA FAIL — investigate before committing.'}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    scan_results  = step1_da_scan(args)
    alpha_results = step2_alpha_eo(args)

    if not args.skip_separability:
        step3_separability(args)

    print_summary(scan_results, alpha_results, args)


if __name__ == "__main__":
    main()
