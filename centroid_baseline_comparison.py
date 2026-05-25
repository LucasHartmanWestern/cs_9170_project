"""
centroid_baseline_comparison.py

Quick comparison of three augmentation strategies on census_income,
using the same dataset splits and FFNN config as the FORGE k=10 best run.

Methods
-------
FORGE          — RL-guided walk; best-checkpoint test metrics loaded from CSV
Centroid-Fix   — all T synthetic samples placed exactly at the DA+ centroid
Centroid-Gauss — T samples ~ N(centroid, diag(sigma^2)),
                 sigma = per-dim empirical std of real DA+ training samples;
                 repeated N_REPEAT times, mean ± std reported

The Centroid-Fix case is intentionally degenerate (2000 identical points) to
establish a floor. Centroid-Gauss is the real comparison: can FORGE beat
informed random noise around the target cluster?

Usage
-----
    source ~/envs/rl/bin/activate
    python centroid_baseline_comparison.py [--device cpu] [--n-repeat 20]
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from dataset import Dataset
from agents.ffnn_agent2 import FFNNAgent
import reward_helpers as rh

# ── Config — must match the k=10, pca=10, ep=30, traj=2000 census best run ──

DATASET_NAME   = "census_income"
MINORITY_ID    = 0          # female (a=0) is disadvantaged
MAJORITY_ID    = 1
DP_COL         = "sex"
PCA_COMPONENTS = 10
TRAIN_SIZE     = 3000
DA_PCT         = 0.01433
TRAJ_LENGTH    = 2000       # synthetic samples per episode
WIN_SECONDS    = 5.0        # ignored by census but required by Dataset API
STEP_SECONDS   = 2.5

FFNN_HIDDEN    = [32, 16]
FFNN_LR        = 1e-3
FFNN_BATCH     = 64
FFNN_EPOCHS    = 30
FFNN_OPTIMIZER = "adam"

SEEDS = [0, 1, 42]

# ── FORGE results (k=10, pca=10, ep=30, ratio=0.4) from grid_seed_rows.csv ──

FORGE_RESULTS = {
    0:  {"eo": 0.0212, "auc": 0.8788, "f1w": 0.8219},
    1:  {"eo": 0.0113, "auc": 0.8830, "f1w": 0.8185},
    42: {"eo": 0.0210, "auc": 0.8652, "f1w": 0.8101},
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_dataset(seed: int, device: str):
    ds = Dataset(
        dataset_name=DATASET_NAME,
        multiclass=False,
        minority_id=MINORITY_ID,
        majority_id=MAJORITY_ID,
        third_id=None,
        pca_components=PCA_COMPONENTS,
        use_pca=True,
        seed=seed,
        device=device,
    )
    x_tr, x_val, x_test, y_tr, y_val, y_test = ds.get_data_splits(
        train_size=TRAIN_SIZE,
        da_pct=DA_PCT,
        bias_pct=None,
        pca_components=PCA_COMPONENTS,
        drop_protected=False,
        protected_cols=ds.protected_attributes,
        dp_protected_col=DP_COL,
        win_seconds=WIN_SECONDS,
        step_seconds=STEP_SECONDS,
    )
    return ds, x_tr, x_val, x_test, y_tr, y_val, y_test


def da_plus_centroid_and_std(x_tr, y_tr, a_tr, minority_id: int, device: str):
    """Return (centroid, per_dim_std) of real DA+ samples in PCA space."""
    mask = (a_tr == minority_id) & (y_tr == 1)
    n_dp = int(mask.sum())
    if n_dp == 0:
        raise RuntimeError("No DA+ samples found — check minority_id / da_pct config")
    pts = x_tr[mask].cpu().numpy()
    centroid = pts.mean(axis=0)
    sigma    = pts.std(axis=0)
    sigma    = np.where(sigma < 1e-6, 1e-6, sigma)   # avoid zero std dims
    print(f"    DA+ count: {n_dp},  centroid norm: {np.linalg.norm(centroid):.3f},  mean-sigma: {sigma.mean():.3f}")
    return torch.tensor(centroid, dtype=torch.float32, device=device), \
           torch.tensor(sigma,    dtype=torch.float32, device=device)


def make_ffnn(seed: int, device: str) -> FFNNAgent:
    return FFNNAgent(
        input_size=PCA_COMPONENTS,
        hidden_sizes=FFNN_HIDDEN,
        output_size=2,
        learning_rate=FFNN_LR,
        batch_size=FFNN_BATCH,
        epochs=FFNN_EPOCHS,
        type="classification",
        classes=[0, 1],
        device=device,
        seed=seed,
        optimizer=FFNN_OPTIMIZER,
    )


def train_and_eval(model: FFNNAgent, x_tr, y_tr, x_test, y_test, a_test, device: str) -> dict:
    """Train model on (x_tr, y_tr), evaluate fairness + utility on test set."""
    loader = DataLoader(
        TensorDataset(x_tr, y_tr),
        batch_size=FFNN_BATCH,
        shuffle=True,
        generator=torch.Generator().manual_seed(model.seed),
    )
    model.train(loader)

    x_test = x_test.to(device)
    y_test = y_test.to(device)
    a_test = torch.as_tensor(a_test, device=device)

    p1   = rh.p1_from_agent(model, x_test)
    fair = rh.fairness_classification_metrics(a_test, y_test, p1)

    y_np = y_test.cpu().numpy()
    p_np = p1.cpu().numpy()
    y_hat = (p1 >= 0.5).long().cpu().numpy()

    auc = float(roc_auc_score(y_np, p_np))
    f1w = float(f1_score(y_np, y_hat, average="weighted", zero_division=0))

    return {"eo": fair["eo_tpr_diff"], "auc": auc, "f1w": f1w}


def augment(x_tr, y_tr, synthetic_x: torch.Tensor, device: str):
    """Append synthetic samples (all label=1) to real training data."""
    n_syn = synthetic_x.shape[0]
    y_syn = torch.ones(n_syn, dtype=torch.long, device=device)
    x_aug = torch.cat([x_tr, synthetic_x.to(device)], dim=0)
    y_aug = torch.cat([y_tr, y_syn], dim=0)
    return x_aug, y_aug


# ── Centroid-Fixed ────────────────────────────────────────────────────────────

def run_centroid_fixed(seed: int, device: str) -> dict:
    ds, x_tr, x_val, x_test, y_tr, y_val, y_test = build_dataset(seed, device)
    a_tr   = ds.a_train.cpu().numpy()
    a_test = ds.a_test.cpu().numpy()

    centroid, _ = da_plus_centroid_and_std(
        x_tr, y_tr.cpu().numpy(), a_tr, MINORITY_ID, device
    )

    # T identical copies of the centroid
    syn_x = centroid.unsqueeze(0).expand(TRAJ_LENGTH, -1).clone()
    x_aug, y_aug = augment(x_tr, y_tr, syn_x, device)

    model = make_ffnn(seed, device)
    return train_and_eval(model, x_aug, y_aug, x_test, y_test, a_test, device)


# ── Centroid-Gauss ────────────────────────────────────────────────────────────

def run_centroid_gauss(seed: int, device: str, n_repeat: int, rng: np.random.Generator) -> dict:
    ds, x_tr, x_val, x_test, y_tr, y_val, y_test = build_dataset(seed, device)
    a_tr   = ds.a_train.cpu().numpy()
    a_test = ds.a_test.cpu().numpy()

    centroid, sigma = da_plus_centroid_and_std(
        x_tr, y_tr.cpu().numpy(), a_tr, MINORITY_ID, device
    )

    eos, aucs, f1ws = [], [], []
    for i in range(n_repeat):
        noise = torch.tensor(
            rng.normal(0, 1, size=(TRAJ_LENGTH, PCA_COMPONENTS)).astype(np.float32),
            device=device,
        )
        syn_x = centroid.unsqueeze(0) + noise * sigma.unsqueeze(0)
        x_aug, y_aug = augment(x_tr, y_tr, syn_x, device)

        model = make_ffnn(seed + i * 1000, device)   # different init per repeat
        result = train_and_eval(model, x_aug, y_aug, x_test, y_test, a_test, device)
        eos.append(result["eo"])
        aucs.append(result["auc"])
        f1ws.append(result["f1w"])
        print(f"      repeat {i+1:02d}/{n_repeat}  EO={result['eo']:.4f}  AUC={result['auc']:.4f}")

    return {
        "eo_mean":  float(np.mean(eos)),
        "eo_std":   float(np.std(eos)),
        "eo_best":  float(np.min(eos)),
        "auc_mean": float(np.mean(aucs)),
        "f1w_mean": float(np.mean(f1ws)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device",   default="cpu")
    p.add_argument("--n-repeat", type=int, default=20,
                   help="Number of Gaussian noise draws per seed")
    args = p.parse_args()

    rng = np.random.default_rng(0)

    fix_rows   = {}
    gauss_rows = {}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        print("\n  [Centroid-Fixed]")
        fix_rows[seed] = run_centroid_fixed(seed, args.device)
        r = fix_rows[seed]
        print(f"  → EO={r['eo']:.4f}  AUC={r['auc']:.4f}  F1w={r['f1w']:.4f}")

        print(f"\n  [Centroid-Gauss  (n_repeat={args.n_repeat})]")
        gauss_rows[seed] = run_centroid_gauss(seed, args.device, args.n_repeat, rng)
        r = gauss_rows[seed]
        print(f"  → EO mean={r['eo_mean']:.4f}±{r['eo_std']:.4f}  best={r['eo_best']:.4f}  AUC={r['auc_mean']:.4f}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY (census_income, k=10 config)")
    print(f"{'='*60}")
    print(f"\n{'Method':<22} {'EO (mean±std)':<18} {'EO best':<10} {'AUC':<8} {'F1w'}")
    print("-" * 68)

    # FORGE row
    forge_eos  = [FORGE_RESULTS[s]["eo"]  for s in SEEDS]
    forge_aucs = [FORGE_RESULTS[s]["auc"] for s in SEEDS]
    forge_f1ws = [FORGE_RESULTS[s]["f1w"] for s in SEEDS]
    forge_eo_mean = np.mean(forge_eos)
    forge_eo_std  = np.std(forge_eos)
    forge_auc_mean = np.mean(forge_aucs)
    forge_f1w_mean = np.mean(forge_f1ws)
    print(f"{'FORGE (k=10, RL)':<22} {forge_eo_mean:.4f}±{forge_eo_std:.4f}    {'—':<10} {forge_auc_mean:.4f}   {forge_f1w_mean:.4f}")

    # Centroid-Fixed rows
    fix_eos  = [fix_rows[s]["eo"]  for s in SEEDS]
    fix_aucs = [fix_rows[s]["auc"] for s in SEEDS]
    fix_f1ws = [fix_rows[s]["f1w"] for s in SEEDS]
    fix_eo_mean = np.mean(fix_eos)
    fix_eo_std  = np.std(fix_eos)
    print(f"{'Centroid-Fixed':<22} {fix_eo_mean:.4f}±{fix_eo_std:.4f}    {'—':<10} {np.mean(fix_aucs):.4f}   {np.mean(fix_f1ws):.4f}")

    # Centroid-Gauss rows
    gauss_eo_means = [gauss_rows[s]["eo_mean"] for s in SEEDS]
    gauss_eo_stds  = [gauss_rows[s]["eo_std"]  for s in SEEDS]
    gauss_eo_bests = [gauss_rows[s]["eo_best"] for s in SEEDS]
    gauss_aucs     = [gauss_rows[s]["auc_mean"] for s in SEEDS]
    gauss_f1ws     = [gauss_rows[s]["f1w_mean"] for s in SEEDS]
    gauss_eo_grand_mean = np.mean(gauss_eo_means)
    gauss_eo_grand_std  = np.mean(gauss_eo_stds)
    gauss_eo_best_mean  = np.mean(gauss_eo_bests)
    print(f"{'Centroid-Gauss (mean)':<22} {gauss_eo_grand_mean:.4f}±{gauss_eo_grand_std:.4f}    {gauss_eo_best_mean:.4f}     {np.mean(gauss_aucs):.4f}   {np.mean(gauss_f1ws):.4f}")

    print()
    print("Per-seed breakdown:")
    print(f"  {'Seed':<6} {'FORGE-EO':<12} {'Fix-EO':<12} {'Gauss-EO(mean)':<18} {'Gauss-EO(best)'}")
    print("  " + "-" * 60)
    for s in SEEDS:
        g = gauss_rows[s]
        print(f"  {s:<6} {FORGE_RESULTS[s]['eo']:<12.4f} {fix_rows[s]['eo']:<12.4f} {g['eo_mean']:<18.4f} {g['eo_best']:.4f}")


if __name__ == "__main__":
    main()
