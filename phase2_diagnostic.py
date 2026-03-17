"""
Phase 2 diagnostic: does phase 2 synthetic data (y=0 majority) actually help?

Compares four scenarios on the census v11 run:
  A) real + syn_y1 only          (phase 1 best synthetic)
  B) real + syn_y0 only          (phase 2 best synthetic)
  C) real + syn_y1 + syn_y0      (combined — current approach)
  D) phase 1 best_beta checkpoint (no retraining, already trained during phase 1)

Usage:
  python phase2_diagnostic.py [--run_dir <path>] [--device cpu]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# -- project imports --
sys.path.insert(0, str(Path(__file__).parent))
from dataset import Dataset
from agents.ffnn_agent2 import FFNNAgent
from test_suite import TestSuite

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def train_beta(agent, x_train, y_train, epochs, batch_size, lr, device):
    """Fresh retrain of a beta FFNNAgent on the given tensors."""
    agent.model.train()
    opt = torch.optim.Adam(agent.model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    ds = TensorDataset(x_train, y_train.long())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = agent.model(xb)
            loss_fn(logits, yb).backward()
            opt.step()
    agent.model.eval()


def evaluate(ts: TestSuite, agent, x_test, y_test, a_test, label: str):
    """Use TestSuite helpers to compute metrics, print a row."""
    agent.model.eval()
    with torch.no_grad():
        logits = agent.model(x_test)
        p1 = torch.softmax(logits, dim=-1)[:, 1]

    fair  = ts._fairness_from_probs(y_test, p1, a_test)
    f1pos, f1neg, f1w, f1mac = ts._all_f1_from_probs(y_test, p1)
    auc   = ts._roc_auc_from_probs(y_test, p1)

    eo   = fair["eo_tpr_diff"]
    dp   = fair["dp_diff"]
    eod  = fair["eod_max_diff"]

    print(f"  {label:<40s}  EO={eo:.4f}  DP={dp:.4f}  EOd={eod:.4f}  "
          f"F1w={f1w:.4f}  F1min={f1pos:.4f}  AUC={auc:.4f}")
    return dict(scenario=label, eo=eo, dp=dp, eod=eod, f1w=f1w, f1min=f1pos, auc=auc)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=(
        "training_runs/"
        "SPECv11_census_dvrl_twophase_1000_EP1000_PCA10_REWfairness_minID1_majID0_"
        "TRJ2000_REAL3000_GG202603161455_5919adff/seed_42"
    ))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    device  = torch.device(args.device)

    # ── Load metadata ──────────────────────────────────────────────────────────
    ffnn_meta = json.loads((run_dir / "ffnn_meta.json").read_text())
    input_size   = ffnn_meta["input_size"]
    hidden_sizes = ffnn_meta["hidden_sizes"]
    lr           = ffnn_meta["learning_rate"]
    batch_size   = ffnn_meta["batch_size"]
    epochs       = ffnn_meta["epochs"]
    n_pca        = ffnn_meta["n_pca_components"]

    print(f"[diag] run_dir  = {run_dir}")
    print(f"[diag] ffnn     : input={input_size}, hidden={hidden_sizes}, "
          f"lr={lr}, epochs={epochs}, pca={n_pca}")
    print(f"[diag] device   = {device}\n")

    # ── Rebuild dataset (same as training) ────────────────────────────────────
    ds = Dataset(dataset_name="census_income", multiclass=False,
                 minority_id=1, majority_id=0, third_id=None,
                 use_pca=True, pca_components=n_pca, seed=42, device=str(device))
    x_train, x_val, x_test, y_train, y_val, y_test = ds.get_data_splits(
        train_size=3000,
        bias_pct=None,
        val_frac=0.1,
        test_frac=0.2,
        pca_components=n_pca,
    )
    # protected attributes stored on ds after split
    a_train = ds.a_train
    a_val   = ds.a_val
    a_test  = ds.a_test

    print(f"[diag] train={x_train.shape[0]}  val={x_val.shape[0]}  test={x_test.shape[0]}")
    print(f"[diag] test y-dist: {y_test.unique(return_counts=True)}")
    print()

    # ── Load synthetic files ───────────────────────────────────────────────────
    ph1 = np.load(run_dir / "best_synthetic_phase1_class1.npz")
    ph2 = np.load(run_dir / "best_synthetic_phase2_class0.npz")

    x_syn1 = torch.tensor(ph1["x"], dtype=torch.float32, device=device)
    y_syn1 = torch.tensor(ph1["y"], dtype=torch.float32, device=device)
    x_syn0 = torch.tensor(ph2["x"], dtype=torch.float32, device=device)
    y_syn0 = torch.tensor(ph2["y"], dtype=torch.float32, device=device)

    print(f"[diag] syn_y1 : x={x_syn1.shape}  y-unique={y_syn1.unique().tolist()}")
    print(f"[diag] syn_y0 : x={x_syn0.shape}  y-unique={y_syn0.unique().tolist()}")
    print()

    # ── Agent factory ─────────────────────────────────────────────────────────
    def make_agent():
        return FFNNAgent(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=2,
            learning_rate=lr,
            device=str(device),
            seed=42,
        )

    # ── TestSuite (just for metric helpers, no file I/O needed) ───────────────
    ts = TestSuite(seed_dir=str(run_dir), experiment_dir=str(run_dir.parent),
                   seed=42, run_id="diag")

    results = []

    # ── Scenario D: phase1 checkpoint (no retrain) ────────────────────────────
    print("── Scenario D: phase1 best_beta checkpoint (no retrain) ─────────────")
    beta_d = make_agent()
    ckpt_p1 = run_dir / "best_beta_state_dict_phase1_class1.pt"
    beta_d.model.load_state_dict(torch.load(ckpt_p1, map_location=device))
    beta_d.model.to(device).eval()
    results.append(evaluate(ts, beta_d, x_test.to(device), y_test.to(device), a_test.to(device),
                             "D) phase1 checkpoint (no retrain)"))

    # ── Scenario A: real + syn_y1 ─────────────────────────────────────────────
    print("\n── Scenario A: real + syn_y1 (phase1 syn only) ──────────────────────")
    beta_a = make_agent()
    xa = torch.cat([x_train, x_syn1], dim=0)
    ya = torch.cat([y_train, y_syn1], dim=0)
    train_beta(beta_a, xa, ya, epochs, batch_size, lr, device)
    results.append(evaluate(ts, beta_a, x_test.to(device), y_test.to(device), a_test.to(device),
                             "A) real + syn_y1 (phase1 only)"))

    # ── Scenario B: real + syn_y0 ─────────────────────────────────────────────
    print("\n── Scenario B: real + syn_y0 (phase2 syn only) ──────────────────────")
    beta_b = make_agent()
    xb = torch.cat([x_train, x_syn0], dim=0)
    yb = torch.cat([y_train, y_syn0], dim=0)
    train_beta(beta_b, xb, yb, epochs, batch_size, lr, device)
    results.append(evaluate(ts, beta_b, x_test.to(device), y_test.to(device), a_test.to(device),
                             "B) real + syn_y0 (phase2 only)"))

    # ── Scenario C: real + syn_y1 + syn_y0 ───────────────────────────────────
    print("\n── Scenario C: real + syn_y1 + syn_y0 (combined, current) ──────────")
    beta_c = make_agent()
    xc = torch.cat([x_train, x_syn1, x_syn0], dim=0)
    yc = torch.cat([y_train, y_syn1, y_syn0], dim=0)
    train_beta(beta_c, xc, yc, epochs, batch_size, lr, device)
    results.append(evaluate(ts, beta_c, x_test.to(device), y_test.to(device), a_test.to(device),
                             "C) real + syn_y1 + syn_y0 (combined)"))

    # ── Alpha baseline (ERM on real only) ────────────────────────────────────
    print("\n── Alpha (ERM on real only, from checkpoint) ────────────────────────")
    alpha = make_agent()
    alpha.model.load_state_dict(torch.load(run_dir / "alpha_state_dict.pt", map_location=device))
    alpha.model.to(device).eval()
    results.append(evaluate(ts, alpha, x_test.to(device), y_test.to(device), a_test.to(device),
                             "Alpha (ERM, real only)"))

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Scenario':<42s}  {'EO':>6}  {'DP':>6}  {'EOd':>6}  {'F1w':>6}  {'F1min':>6}  {'AUC':>6}")
    print("-" * 90)
    for r in results:
        print(f"  {r['scenario']:<40s}  {r['eo']:>6.4f}  {r['dp']:>6.4f}  "
              f"{r['eod']:>6.4f}  {r['f1w']:>6.4f}  {r['f1min']:>6.4f}  {r['auc']:>6.4f}")
    print("=" * 90)


if __name__ == "__main__":
    main()
