"""
dataset_viability.py — Unified dataset structural viability checker.

Runs four checks needed before committing to full RL experiments:
  1. DA+ scan: sweep bias_pct values (and seeds) to find a regime with DA+ ≈ 43-45
  2. Alpha-EO baseline: train FFNN alpha at target bias_pct(s), report fairness + utility
  3. Feature separability: centroid distance between group-conditional positives in PCA space;
     also reports cosine similarity of discriminant vectors (pos-neg direction per group) —
     high cosine (>0.7) means both groups' positive class is in the same PCA direction,
     so targeted generation cannot close the EO gap specifically.
  4. WGL-EO alignment probe: two fast pre-flight checks for WGL↔EO tracking:
     (a) Alpha WGL group dominance — is DA the consistent worst group by validation BCE?
     (b) Targeted augmentation test — generate synthetic DA+ near centroid, train beta,
         check if EO drops. If it doesn't, FORGE cannot help regardless of reward quality.

Pass/fail criteria:
  - val_disadv_pos  ≥ 30
  - test_disadv_pos ≥ 200
  - alpha_EO        clearly non-zero (≥ 0.05 in viability; actual is ~3-5x higher)
  - sep_ratio       > 1.0
  - cosine_sim      < 0.70  (WARN if 0.50-0.70, FAIL if ≥ 0.70)
  - wgl_dominance   DA wgl > AA wgl on val (WARN if ratio < 1.5)
  - targeted_aug    beta_EO < alpha_EO on val after synthetic injection

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
    p.add_argument("--brfss-outcome", default="cvdinfr4",
                   help="BRFSS outcome to predict (cvdinfr4, cvdcrhd4, cvdstrk3, addepev3)")
    p.add_argument("--covertype-positive", type=int, default=5,
                   help="Covertype: cover type id (1-7) to treat as y=1 (default=5 Aspen)")
    p.add_argument("--acs-states", nargs="+", default=None,
                   help="ACS Employment: list of state codes to load (e.g. CA TX NY FL)")
    p.add_argument("--ffnn-epochs", type=int, default=20,
                   help="Epochs to train alpha FFNN for EO baseline (default: 20)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dataset(args, seed, bias_pct=None, da_pct=None):
    # covertype uses binary group IDs (0=minority, 1=majority) in a_train;
    # the original area IDs are passed separately as minority_area/majority_area.
    minority_id = 0 if args.dataset == "covertype" else args.minority_id
    majority_id = 1 if args.dataset == "covertype" else args.majority_id

    ds = Dataset(
        dataset_name=args.dataset,
        multiclass=False,
        minority_id=minority_id,
        majority_id=majority_id,
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
    if args.dataset == "brfss":
        kwargs["brfss_outcome"] = args.brfss_outcome
    if args.dataset == "covertype":
        kwargs["positive_cover_type"] = args.covertype_positive
        kwargs["minority_area"] = args.minority_id  # original wilderness area ID
        kwargs["majority_area"] = args.majority_id
    if args.dataset == "acs_employment":
        if args.acs_states is not None:
            kwargs["acs_states"] = args.acs_states
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
        epochs=args.ffnn_epochs,
        device=args.device,
        seed=seed,
    )
    # Plain shuffle — matches training.py's train_predictor_model exactly.
    # No class weighting: the alpha model is intentionally trained on the biased
    # distribution so it under-detects minority positives, producing the EO gap
    # that FORGE is designed to close.
    loader = DataLoader(TensorDataset(x_train, y_train.float()),
                        batch_size=64, shuffle=True)
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
                da_plus, n_disadv = group_counts(ds.a_train, y_tr, ds.MINORITY_ID)

                a_val_np = ds.a_val.cpu().numpy() if torch.is_tensor(ds.a_val) else np.array(ds.a_val)
                y_val_np = y_val.cpu().numpy()
                val_dp = int(np.sum((a_val_np == ds.MINORITY_ID) & (y_val_np == 1)))

                a_te_np = ds.a_test.cpu().numpy() if torch.is_tensor(ds.a_test) else np.array(ds.a_test)
                y_te_np = y_te.cpu().numpy()
                test_dp = int(np.sum((a_te_np == ds.MINORITY_ID) & (y_te_np == 1)))

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
                da_plus, _ = group_counts(ds.a_train, y_tr, ds.MINORITY_ID)

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

    Also computes cosine similarity of discriminant vectors (pos-neg direction
    per group): high cosine means both groups' positive class manifolds are in
    the same PCA direction, so targeted generation cannot specifically close the
    EO gap (WGL-EO disconnect risk).

    Returns dict with sep_ratio and cosine_sim (or None on error).
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
        return None

    x_np = x_te.cpu().numpy()
    y_np = y_te.cpu().numpy()
    a_np = ds.a_test.cpu().numpy() if torch.is_tensor(ds.a_test) else np.array(ds.a_test)

    disadv_pos = x_np[(y_np == 1) & (a_np == ds.MINORITY_ID)]
    adv_pos    = x_np[(y_np == 1) & (a_np == ds.MAJORITY_ID)]
    disadv_neg = x_np[(y_np == 0) & (a_np == ds.MINORITY_ID)]
    adv_neg    = x_np[(y_np == 0) & (a_np == ds.MAJORITY_ID)]

    print(f"  Test subgroup sizes:")
    print(f"    disadv_pos={len(disadv_pos)}  adv_pos={len(adv_pos)}  "
          f"disadv_neg={len(disadv_neg)}  adv_neg={len(adv_neg)}")

    if len(disadv_pos) < 2 or len(adv_pos) < 2:
        print("  Too few positive examples for separability analysis.\n")
        return None

    c_disadv = disadv_pos.mean(axis=0)
    c_adv    = adv_pos.mean(axis=0)
    between  = float(np.linalg.norm(c_disadv - c_adv))
    within_d = float(disadv_pos.std(axis=0).mean())
    within_a = float(adv_pos.std(axis=0).mean())
    sep_ratio = between / max(within_d, within_a, 1e-8)

    print(f"\n  Centroid distance (disadv_pos vs adv_pos): {between:.4f}")
    print(f"  Within-group spread: disadv={within_d:.4f}  adv={within_a:.4f}")
    print(f"  Separability ratio (between/within):       {sep_ratio:.4f}")
    sep_label = "PASS" if sep_ratio > 1.0 else "FAIL"
    print(f"  sep_ratio threshold (> 1.0):               {sep_label}")

    cosine_sim = None
    if len(disadv_neg) > 0 and len(adv_neg) > 0:
        c_disadv_neg = disadv_neg.mean(axis=0)
        c_adv_neg    = adv_neg.mean(axis=0)
        disc_d = c_disadv - c_disadv_neg
        disc_a = c_adv    - c_adv_neg
        cosine_sim = float(np.dot(disc_d, disc_a) /
                           (np.linalg.norm(disc_d) * np.linalg.norm(disc_a) + 1e-8))
        if cosine_sim < 0.90:
            cos_label = "OK   (group-specific PCA structure present)"
        elif cosine_sim < 0.95:
            cos_label = "WARN (moderate directional overlap — monitor WGL-EO tracking at ep ~500)"
        else:
            cos_label = "WARN (high directional overlap — WGL-EO disconnect risk; verify with probe)"
        print(f"\n  Cosine similarity of discriminant vectors: {cosine_sim:.4f}  {cos_label}")
        print(f"    Interpretation: measures whether the pos-vs-neg direction is the")
        print(f"    same for both groups in PCA space. Near 1.0 = targeted generation")
        print(f"    may move both groups equally. Use Step 4 targeted aug probe to verify.")

    print()
    return {"sep_ratio": sep_ratio, "cosine_sim": cosine_sim}


# ---------------------------------------------------------------------------
# Step 4: WGL-EO alignment probe
# ---------------------------------------------------------------------------

def step4_wgl_eo_probe(args):
    """
    Two pre-flight checks for WGL-EO alignment, using the target da_pct.

    A. Alpha WGL group dominance
       Train alpha on biased data. On the validation set, compute mean BCE loss
       separately for DA and AA groups. If wgl_DA > wgl_AA, the DA group is the
       consistent worst group — the WGL reward will reliably target the right group.
       Ratio close to 1.0 or below 1.0 signals the reward may not target DA group.

    B. Targeted augmentation test
       Generate synthetic DA+ points near the DA+ training centroid (Gaussian noise
       with within-group std), augment the real training set, train a beta model,
       evaluate EO on validation. If beta-EO < alpha-EO: targeted generation can
       reduce the fairness gap — FORGE has a viable path. If not: even perfect
       spatial targeting by the RL agent would not close EO, and the run will waste
       GPU hours with WGL improving while EO stagnates.

    Returns dict with probe results, or None if insufficient data.
    """
    # Pick the last (target) da_pct; skip if no scarcity values given
    if args.da_pcts:
        target_da_pct = args.da_pcts[-1]
        kw = {"da_pct": target_da_pct}
        param_str = f"da_pct={target_da_pct}"
    elif args.bias_pcts:
        target_da_pct = None
        kw = {"bias_pct": args.bias_pcts[-1]}
        param_str = f"bias_pct={args.bias_pcts[-1]}"
    else:
        print("STEP 4: WGL-EO alignment probe — skipped (no scarcity pct specified)\n")
        return None

    print("=" * 70)
    print(f"STEP 4: WGL-EO alignment probe  ({param_str}, seed={args.seeds[0]})")
    print("  A) Alpha WGL group dominance — is DA the consistent worst group by val BCE?")
    print("  B) Targeted augmentation test — does generating near DA+ centroid reduce EO?")
    print("=" * 70)

    import torch.nn as nn

    seed = args.seeds[0]
    try:
        ds, x_tr, x_val, x_te, y_tr, y_val, y_te = make_dataset(args, seed, **kw)
    except Exception as e:
        print(f"  ERROR loading biased dataset: {e}\n")
        return None

    # --- Train alpha ---
    alpha = train_alpha(x_tr, y_tr, seed, args)
    p_val_alpha = get_probs(alpha, x_val, args.device)
    alpha_eo = eo_gap(ds.a_val, y_val, p_val_alpha)

    # --- A: WGL group dominance ---
    a_val_np = ds.a_val.cpu().numpy() if torch.is_tensor(ds.a_val) else np.array(ds.a_val)
    y_val_np  = y_val.cpu().numpy()

    bce_fn = nn.BCELoss(reduction="none")
    alpha.model.eval()
    with torch.no_grad():
        logits = alpha.model(x_val.to(args.device)).squeeze(-1)
        probs  = torch.sigmoid(logits)
        bce_all = bce_fn(probs, y_val.float().to(args.device)).cpu().numpy()

    da_mask = (a_val_np == ds.MINORITY_ID)
    aa_mask = (a_val_np == ds.MAJORITY_ID)
    wgl_da = float(bce_all[da_mask].mean()) if da_mask.sum() > 0 else float("nan")
    wgl_aa = float(bce_all[aa_mask].mean()) if aa_mask.sum() > 0 else float("nan")
    dominance_ratio = wgl_da / (wgl_aa + 1e-8)

    print(f"\n  A) WGL group dominance (alpha, validation BCE):")
    print(f"     wgl_DA={wgl_da:.4f}  wgl_AA={wgl_aa:.4f}  ratio={dominance_ratio:.3f}")
    if dominance_ratio >= 1.5:
        dom_label = "strong (DA group is clear worst group by val BCE)"
    elif dominance_ratio >= 1.0:
        dom_label = "marginal (DA slightly worse — reward may be noisy)"
    else:
        dom_label = "inverted (AA has higher BCE on val — WGL targets AA; watch for disconnect)"
    print(f"     {dom_label}")
    print(f"     Note: computed on unbiased val set; ratio < 1.0 is common for working datasets.")

    # --- B: Targeted augmentation test ---
    x_tr_np = x_tr.cpu().numpy()
    y_tr_np = y_tr.cpu().numpy()
    a_tr_np = ds.a_train.cpu().numpy() if torch.is_tensor(ds.a_train) else np.array(ds.a_train)

    da_pos_mask = (a_tr_np == ds.MINORITY_ID) & (y_tr_np == 1)
    n_da_pos = da_pos_mask.sum()

    print(f"\n  B) Targeted augmentation test:")
    print(f"     DA+ in training: {n_da_pos}")

    if n_da_pos < 2:
        print("     Too few DA+ training examples for probe. Skipping.\n")
        return {"wgl_da": wgl_da, "wgl_aa": wgl_aa, "dominance_ratio": dominance_ratio,
                "alpha_eo": alpha_eo, "aug_beta_eo": float("nan")}

    da_pos_pts = x_tr_np[da_pos_mask]
    centroid   = da_pos_pts.mean(axis=0)
    spread     = da_pos_pts.std(axis=0).mean()

    # Generate n_synth synthetic DA+ points: centroid + Gaussian noise
    n_synth = min(2000, max(200, 10 * n_da_pos))
    rng = np.random.RandomState(seed)
    noise = rng.randn(n_synth, centroid.shape[0]) * max(spread, 0.1)
    syn_x = centroid[None, :] + noise          # (n_synth, d)
    syn_y = np.ones(n_synth)

    x_aug = np.concatenate([x_tr_np, syn_x], axis=0)
    y_aug = np.concatenate([y_tr_np, syn_y],  axis=0)

    x_aug_t = torch.tensor(x_aug, dtype=torch.float32)
    y_aug_t = torch.tensor(y_aug, dtype=torch.float32)

    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(x_aug_t, y_aug_t), batch_size=64, shuffle=True)

    from agents.ffnn_agent2 import FFNNAgent
    beta = FFNNAgent(
        input_size=x_aug_t.shape[1],
        hidden_sizes=[32, 16],
        output_size=1,
        learning_rate=0.001,
        batch_size=64,
        epochs=args.ffnn_epochs,
        device=args.device,
        seed=seed + 1,
    )
    beta.train(loader)
    p_val_beta = get_probs(beta, x_val, args.device)
    beta_eo = eo_gap(ds.a_val, y_val, p_val_beta)

    delta = alpha_eo - beta_eo
    print(f"     Synthetic points injected: {n_synth}  (near DA+ centroid, spread={spread:.3f})")
    print(f"     alpha-EO={alpha_eo:.4f}  beta-EO={beta_eo:.4f}  delta={delta:+.4f}")
    if delta > 0.02:
        aug_label = "PASS  (targeted augmentation reduces EO — FORGE has a viable path)"
    elif delta > 0.0:
        aug_label = "WARN  (marginal improvement — FORGE may work but signal is weak)"
    else:
        aug_label = "FAIL  (EO did not improve — even perfect targeting cannot close gap)"
    print(f"     {aug_label}")
    print()

    return {
        "wgl_da": wgl_da, "wgl_aa": wgl_aa, "dominance_ratio": dominance_ratio,
        "alpha_eo": alpha_eo, "aug_beta_eo": beta_eo, "aug_delta": delta,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(scan_results, alpha_results, sep_results, probe_results, args):
    print("=" * 70)
    print("SUMMARY — Pass/Fail against paper criteria")
    print("=" * 70)

    hard_pass = {}   # must all be True to proceed
    soft_warn = {}   # warnings that warrant caution

    # --- Scan criteria ---
    if scan_results:
        max_val  = max(r["val_disadv_pos"]  for r in scan_results)
        max_test = max(r["test_disadv_pos"] for r in scan_results)
        hard_pass["val_disadv_pos ≥ 30"]   = max_val  >= 30
        hard_pass["test_disadv_pos ≥ 200"] = max_test >= 200
        print(f"  val_disadv_pos  (best): {max_val:>6}   {'PASS' if max_val >= 30 else 'FAIL'}")
        print(f"  test_disadv_pos (best): {max_test:>6}   {'PASS' if max_test >= 200 else 'FAIL'}")

    # --- Alpha-EO criteria ---
    if alpha_results:
        valid_val = [r["val_eo"]  for r in alpha_results if not np.isnan(r["val_eo"])]
        valid_te  = [r["test_eo"] for r in alpha_results if not np.isnan(r["test_eo"])]
        if valid_val:
            max_val_eo  = max(valid_val)
            max_test_eo = max(valid_te) if valid_te else float("nan")
            hard_pass["alpha_EO ≥ 0.05"]   = max_val_eo  >= 0.05
            print(f"  best val  alpha-EO:     {max_val_eo:.4f}   {'PASS' if max_val_eo >= 0.05 else 'FAIL'}")
            print(f"  best test alpha-EO:     {max_test_eo:.4f}")

    # --- Separability criteria ---
    if sep_results:
        sr = sep_results.get("sep_ratio", float("nan"))
        cs = sep_results.get("cosine_sim", None)
        hard_pass["sep_ratio > 1.0"] = sr > 1.0
        print(f"  sep_ratio:              {sr:.4f}   {'PASS' if sr > 1.0 else 'FAIL'}")
        if cs is not None:
            # Cosine is informational only — no reliable hard threshold distinguishes
            # working datasets (census ~0.97) from failing ones (ACS Employment ~0.98).
            # Flag as WARN above 0.95 to prompt closer inspection of the aug probe.
            if cs >= 0.95:
                cs_status = "WARN"
                soft_warn["cosine_sim < 0.95"] = False
            else:
                cs_status = "OK  "
            print(f"  cosine_sim (disc vec):  {cs:.4f}   {cs_status}  (informational; see Step 4 probe)")

    # --- Probe criteria ---
    if probe_results:
        dom   = probe_results.get("dominance_ratio", float("nan"))
        delta = probe_results.get("aug_delta", float("nan"))

        # WGL dominance: informational only — ratio < 1.0 is common for datasets that work.
        if not np.isnan(dom):
            if dom < 1.0:
                dom_status = "WARN (inverted; see note above)"
                soft_warn["wgl_dominance_inverted"] = False
            else:
                dom_status = "OK  "
            print(f"  WGL dominance (DA/AA):  {dom:.3f}   {dom_status}")

        # Targeted augmentation: hard criterion.
        # delta <= 0: even perfect targeting can't close EO gap — FORGE cannot help.
        # delta > 0 but ≤ 0.02: marginal; proceed with caution.
        # delta > 0.02: necessary (not sufficient) condition for FORGE to work.
        # NOTE: passing here does not guarantee success — ACS Employment passes at
        # +0.10 yet fails in practice due to noisy reward in large-validation-set regime.
        if not np.isnan(delta):
            if delta > 0.02:
                aug_status = "PASS"
            elif delta > 0.0:
                aug_status = "WARN"
                soft_warn["targeted_aug delta > 0.02"] = False
            else:
                aug_status = "FAIL"
                hard_pass["targeted_aug delta > 0.0"] = False
            print(f"  targeted_aug delta EO:  {delta:+.4f}  {aug_status}  (hard threshold: > 0.0)")

    print()
    hard_ok = all(hard_pass.values()) if hard_pass else False
    n_warn   = len(soft_warn)
    if hard_ok and n_warn == 0:
        verdict = "ALL CRITERIA PASS — viable for RL experiments."
    elif hard_ok and n_warn > 0:
        verdict = f"HARD CRITERIA PASS with {n_warn} warning(s) — proceed with caution; monitor WGL-EO tracking at ep ~500."
    else:
        fails = [k for k, v in hard_pass.items() if not v]
        verdict = f"FAIL on: {', '.join(fails)} — do not proceed."
    print(f"  Overall: {verdict}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    scan_results  = step1_da_scan(args)
    alpha_results = step2_alpha_eo(args)

    sep_results   = None
    probe_results = None
    if not args.skip_separability:
        sep_results   = step3_separability(args)
        probe_results = step4_wgl_eo_probe(args)

    print_summary(scan_results, alpha_results, sep_results, probe_results, args)


if __name__ == "__main__":
    main()
