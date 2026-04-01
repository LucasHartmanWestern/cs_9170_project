"""
MEPS diagnostic: outpatient visits (OPTOTV22), Hispanic vs non-Hispanic.

Checks DA+ at different bias_pct values, then runs the alpha baseline to
confirm EO gap > 0.10 before committing to full RL experiments.

Usage:
    python meps_diagnostic.py

Expected file: datasets/meps/h243.csv  (pre-converted from SAS transport)
Fallback:      datasets/meps/h243.ssp  (read directly via pandas.read_sas)
Download HC-243 from: https://meps.ahrq.gov/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-243
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from dataset import Dataset
from agents.ffnn_agent2 import FFNNAgent
from reward_helpers import p1_from_agent, eo_gap_from_probs

SEED   = 42
DEVICE = "cpu"

def make_dataset(bias_pct, use_pca=False):
    ds = Dataset(
        dataset_name="meps",
        multiclass=False,
        minority_id=0,
        majority_id=1,
        third_id=None,
        seed=SEED,
        device=DEVICE,
        use_pca=use_pca,
    )
    splits = ds.get_data_splits(
        train_size=3000,
        bias_pct=bias_pct,
        pca_components=10,
        dp_protected_col="ethnicity",
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

# ---- Step 1: DA+ scan ----
print("=" * 60)
print("STEP 1: DA+ scan across bias_pct values")
print("  (disadvantaged = Hispanic, a=0; outcome = OPTOTV22)")
print("=" * 60)

for bias_pct in [None, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]:
    try:
        ds, splits = make_dataset(bias_pct, use_pca=False)
    except Exception as e:
        print(f"  bias_pct={bias_pct}: ERROR — {e}")
        continue
    a_tr = ds.a_train.numpy()
    y_tr = splits[3].numpy()
    da_plus  = int(np.sum((y_tr == 1) & (a_tr == 0)))
    n_disadv = int(np.sum(a_tr == 0))
    print(f"  bias_pct={str(bias_pct):>5}: DA+={da_plus:3d}  (of {n_disadv} Hispanic train examples)")

# ---- Step 2: alpha-EO at target bias_pct ----
# Adjust TARGET_BIAS_PCT after seeing Step 1 output to land near DA+≈43.
TARGET_BIAS_PCT = 0.08

print()
print("=" * 60)
print(f"STEP 2: Alpha-EO baseline at bias_pct={TARGET_BIAS_PCT}")
print("=" * 60)

ds, (X_tr, X_val, X_te, y_tr, y_val, y_te) = make_dataset(TARGET_BIAS_PCT, use_pca=True)

da_plus  = int(np.sum((y_tr.numpy() == 1) & (ds.a_train.numpy() == 0)))
n_disadv = int(np.sum(ds.a_train.numpy() == 0))
print(f"DA+ (train): {da_plus}  ({n_disadv} Hispanic train examples total)")

alpha = train_ffnn(X_tr, y_tr, input_dim=X_tr.shape[1])

p1_val  = p1_from_agent(alpha, X_val)
p1_test = p1_from_agent(alpha, X_te)

eo_val  = eo_gap_from_probs(ds.a_val,   y_val, p1_val).item()
eo_test = eo_gap_from_probs(ds.a_test,  y_te,  p1_test).item()

print(f"Alpha EO gap (val):  {eo_val:.3f}")
print(f"Alpha EO gap (test): {eo_test:.3f}")
print()
print("Interpretation:")
print("  EO > 0.10 → viable, proceed to GroupDRO baseline test")
print("  EO < 0.05 → classifier already fair, not useful for paper")
