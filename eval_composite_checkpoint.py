"""
Retroactive evaluation using composite checkpoint selection (global_obj * utility_factor).

For each seed in a v15a run:
  1. Find the episode with max(global_obj * utility_factor) from metrics.csv
  2. Load the nearest saved phase1 snapshot
  3. Combine with the original best phase2 synthetic
  4. Train a fresh beta on real + syn1 + syn0
  5. Evaluate with the test suite (alpha + beta metrics)

Usage:
    python eval_composite_checkpoint.py
"""

import json
import random
import numpy as np
import torch
import pandas as pd
from pathlib import Path

from dataset import Dataset
from agents.ffnn_agent2 import FFNNAgent
from episode_tracker import EpisodeTracker
import reward_helpers as rh

# ── Config ────────────────────────────────────────────────────
ROOT = Path("training_runs")
RUN_FOLDER = sorted(ROOT.glob("SPECv15a_census_lambda05_3seeds_*"))[-1]
SPEC_PATH  = Path("experiment_specs/v15a_census_lambda05_3seeds.json")
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Run:    {RUN_FOLDER.name}")
print(f"Device: {DEVICE}")

with open(SPEC_PATH) as f:
    spec = json.load(f)

FFNN_CONFIG = dict(
    hidden_sizes=spec["ffnn"]["hidden_sizes"],
    learning_rate=spec["ffnn"]["learning_rate"],
    batch_size=spec["ffnn"]["batch_size"],
    epochs=spec["ffnn"]["epochs"],
    output_size=2,
    classes=[0, 1],
    type="classification",
)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_ffnn(input_size, seed):
    return FFNNAgent(
        input_size=input_size,
        device=DEVICE,
        seed=seed,
        **FFNN_CONFIG,
    )

def train_model(model, x, y):
    from torch.utils.data import TensorDataset, DataLoader
    x = x.to(DEVICE); y = y.to(DEVICE)
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=model.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(model.seed))
    model.train(loader)
    return model

def evaluate(model, x_test, y_test, a_test, f1_thresh=0.5):
    model.model.to(DEVICE)
    x_test = x_test.to(DEVICE); y_test = y_test.to(DEVICE)
    p1 = rh.p1_from_agent(model, x_test)
    fair = rh.fairness_classification_metrics(a_test, y_test, p1, threshold=f1_thresh)
    auc  = rh.roc_auc_from_probs(y_test, p1)
    acc  = rh.acc_from_probs(y_test, p1, threshold=f1_thresh)
    # f1_weighted via sklearn
    from sklearn.metrics import f1_score
    y_np = y_test.cpu().numpy().astype(int)
    yhat = (p1.detach().cpu().numpy() >= f1_thresh).astype(int)
    f1_w = float(f1_score(y_np, yhat, average="weighted", zero_division=0))
    # brier
    brier = float(rh.brier_per_sample(y_test.float(), p1).mean().item())
    return {**fair, "roc_auc": auc, "acc": acc, "f1_weighted": f1_w, "brier": brier}

# ── Per-seed evaluation ───────────────────────────────────────
results = []

for seed_dir in sorted(RUN_FOLDER.glob("seed_*")):
    seed = int(seed_dir.name.split("_")[1])
    seed_everything(seed)
    print(f"\n{'='*60}")
    print(f"Seed {seed}")

    # 1. Find composite-best phase1 episode
    df = pd.read_csv(seed_dir / "metrics.csv")
    p1_df = df[df["meta.phase"].str.contains("phase1", na=False)].copy()
    p1_df["composite"] = p1_df["global.global_obj"] * p1_df["global.utility_factor"]
    best_row = p1_df.loc[p1_df["composite"].idxmax()]
    best_ep  = int(best_row["episode"])
    print(f"  Composite-best ep={best_ep}  global={best_row['global.global_obj']:.4f}  "
          f"util_factor={best_row['global.utility_factor']:.4f}  "
          f"composite={best_row['composite']:.4f}  "
          f"EO(train)={best_row['fairness.eo_tpr_diff']:.4f}")

    # 2. Find nearest saved snapshot
    snaps = sorted(seed_dir.glob("synthetic_snapshots/synthetic_ep*_phase1_class1.npz"))
    snap_eps = [int(s.name.split("ep")[1].split("_")[0]) for s in snaps]
    closest_ep = min(snap_eps, key=lambda e: abs(e - best_ep))
    snap_path = seed_dir / "synthetic_snapshots" / f"synthetic_ep{closest_ep:04d}_phase1_class1.npz"
    print(f"  Using snapshot ep={closest_ep}: {snap_path.name}")

    # 3. Load phase1 synthetic (composite-best)
    syn1_data = np.load(snap_path)
    x_syn1 = torch.tensor(syn1_data["x"], dtype=torch.float32)
    y_syn1 = torch.tensor(syn1_data["y"], dtype=torch.long)

    # 4. Load phase2 synthetic (original best)
    syn0_path = seed_dir / "best_synthetic_phase2_class0.npz"
    syn0_data = np.load(syn0_path)
    x_syn0 = torch.tensor(syn0_data["x"], dtype=torch.float32)
    y_syn0 = torch.tensor(syn0_data["y"], dtype=torch.long)

    # 5. Reconstruct dataset + data splits
    dataset = Dataset(
        dataset_name=spec["dataset_name"],
        multiclass=spec.get("multiclass", False),
        minority_id=spec["minority_id"],
        majority_id=spec["majority_id"],
        third_id=spec.get("third_id"),
        pca_components=spec["pca_components"],
        seed=seed,
        device=DEVICE,
        use_pca=spec.get("use_pca", True),
        whiten_pca=spec.get("whiten_pca", False),
    )
    x_train, x_val, x_test, y_train, y_val, y_test = dataset.get_data_splits(
        train_size=spec["real_data_size"],
        bias_pct=spec.get("bias_pct"),
        pca_components=spec["pca_components"],
        drop_protected=False,
        protected_cols=dataset.protected_attributes,
        bias_val=spec.get("bias_val", True),
    )
    a_test = dataset.a_test
    print(f"  Data: train={len(x_train)}, test={len(x_test)}")

    input_size = x_train.shape[1]

    # 6. Load and evaluate alpha
    alpha = make_ffnn(input_size, seed)
    alpha.model.load_state_dict(torch.load(seed_dir / "alpha_state_dict.pt", map_location=DEVICE))
    alpha_metrics = evaluate(alpha, x_test, y_test, a_test)

    # 7. Train fresh beta on real + composite syn1 + best syn0
    dev = torch.device(DEVICE)
    x_combined = torch.cat([x_train.to(dev), x_syn1.to(dev), x_syn0.to(dev)])
    y_combined = torch.cat([y_train.to(dev), y_syn1.to(dev), y_syn0.to(dev)])
    beta = make_ffnn(input_size, seed)
    beta = train_model(beta, x_combined, y_combined)
    beta_metrics = evaluate(beta, x_test, y_test, a_test)

    print(f"  Alpha: EO={alpha_metrics['eo_tpr_diff']:.4f}  F1w={alpha_metrics['f1_weighted']:.4f}  AUC={alpha_metrics['roc_auc']:.4f}")
    print(f"  Beta:  EO={beta_metrics['eo_tpr_diff']:.4f}  F1w={beta_metrics['f1_weighted']:.4f}  AUC={beta_metrics['roc_auc']:.4f}  "
          f"dEO={((beta_metrics['eo_tpr_diff']-alpha_metrics['eo_tpr_diff'])/alpha_metrics['eo_tpr_diff']*100):+.1f}%  "
          f"dF1w={(beta_metrics['f1_weighted']-alpha_metrics['f1_weighted'])*100:+.2f}pp  "
          f"dAUC={(beta_metrics['roc_auc']-alpha_metrics['roc_auc'])*100:+.2f}pp")

    results.append({
        "seed": seed,
        "alpha_eo": alpha_metrics["eo_tpr_diff"],
        "beta_eo":  beta_metrics["eo_tpr_diff"],
        "alpha_f1w": alpha_metrics["f1_weighted"],
        "beta_f1w":  beta_metrics["f1_weighted"],
        "alpha_auc": alpha_metrics["roc_auc"],
        "beta_auc":  beta_metrics["roc_auc"],
    })

# ── Summary ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print("COMPOSITE CHECKPOINT RESULTS (v15a re-eval)")
df_r = pd.DataFrame(results)
print(df_r.to_string(index=False))
print()
print(f"Mean beta EO:  {df_r['beta_eo'].mean():.4f} ± {df_r['beta_eo'].std():.4f}")
print(f"Mean beta F1w: {df_r['beta_f1w'].mean():.4f} ± {df_r['beta_f1w'].std():.4f}")
print(f"Mean beta AUC: {df_r['beta_auc'].mean():.4f} ± {df_r['beta_auc'].std():.4f}")
print()
print("Compare to v15a original final results:")
print("  Mean EO:  0.0557 ± 0.0687")
print("  Mean F1w: 0.8135 ± 0.0090")
print("  Mean AUC: 0.8586 ± 0.0056")
