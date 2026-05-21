#!/bin/bash
# Run dataset viability checks on all 3 paper datasets + ACS PCA analysis
# FFNN: 20 epochs; ACS: disability framing, 10 states
source ~/envs/rl/bin/activate
cd /home/epigou/cs_9170_project

echo "##########################################################################"
echo "# CENSUS INCOME  (da_pct=0.01433, real=3000, pca=10, minority=female=0) #"
echo "##########################################################################"
python dataset_viability.py \
    --dataset census_income \
    --minority-id 0 --majority-id 1 \
    --da-pcts 0.01433 \
    --seeds 42 0 1 \
    --real-data-size 3000 \
    --pca-components 10 \
    --ffnn-epochs 20 \
    --device cpu

echo ""
echo "##########################################################################"
echo "# CAPTURE-24  (da_pct=0.015, real=4000, pca=15, minority=female=1)      #"
echo "##########################################################################"
python dataset_viability.py \
    --dataset capture24 \
    --minority-id 1 --majority-id 0 \
    --da-pcts 0.015 \
    --seeds 42 0 1 \
    --real-data-size 4000 \
    --pca-components 15 \
    --win-seconds 1.0 --step-seconds 0.5 \
    --ffnn-epochs 20 \
    --device cpu

echo ""
echo "##########################################################################"
echo "# ACS EMPLOYMENT — disability framing (DIS=1->a=0, 10 states)           #"
echo "# da_pct=0.01433, real=3000, pca=10                                      #"
echo "##########################################################################"
python dataset_viability.py \
    --dataset acs_employment \
    --minority-id 0 --majority-id 1 \
    --dp-col disability \
    --da-pcts 0.01433 \
    --seeds 42 0 1 \
    --real-data-size 3000 \
    --pca-components 10 \
    --acs-states CA TX NY FL PA OH IL GA NC MI \
    --ffnn-epochs 20 \
    --device cpu

echo ""
echo "##########################################################################"
echo "# ACS PCA VARIANCE ANALYSIS (disability framing, 10 states, up to k=30) #"
echo "##########################################################################"
python - <<'PYEOF'
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path('/home/epigou/cs_9170_project')))

from dataset import Dataset

print("Loading ACS Employment (seed=42, disability framing, no bias, pca_components=30)...")

ds = Dataset(
    dataset_name='acs_employment',
    multiclass=False,
    minority_id=0,
    majority_id=1,
    third_id=None,
    pca_components=30,
    seed=42,
    device='cpu',
    use_pca=True,
)

splits = ds.get_data_splits(
    train_size=3000,
    da_pct=None,
    pca_components=30,
    drop_protected=True,
    protected_cols=ds.protected_attributes,
    dp_protected_col='disability',
    acs_states=['CA', 'TX', 'NY', 'FL', 'PA', 'OH', 'IL', 'GA', 'NC', 'MI'],
)

if hasattr(ds, 'pca_transform') and ds.pca_transform is not None:
    pca = ds.pca_transform
    var_ratio = pca.explained_variance_ratio_
    cumvar = np.cumsum(var_ratio)
    print(f"\nTotal features before PCA: {pca.n_features_in_}")
    print(f"\n{'Components':>10}  {'Var @k':>10}  {'Cumulative':>11}")
    print("-" * 35)
    for k in [5, 8, 10, 12, 15, 20, 25, 30]:
        if k <= len(cumvar):
            print(f"{k:>10}  {var_ratio[k-1]:>10.4f}  {cumvar[k-1]:>11.4f}")
    print(f"\nFull scree (first 30 components):")
    for i, v in enumerate(var_ratio[:30]):
        bar = '#' * int(v * 200)
        print(f"  PC{i+1:02d}: {v:.4f}  cum={cumvar[i]:.4f}  {bar}")
else:
    from sklearn.decomposition import PCA
    x_train = splits[0]
    x_np = x_train.cpu().numpy() if hasattr(x_train, 'cpu') else np.array(x_train)
    print(f"\nFallback PCA on training set (n={len(x_np)}, d={x_np.shape[1]})")
    pca = PCA(n_components=min(30, x_np.shape[1]))
    pca.fit(x_np)
    var_ratio = pca.explained_variance_ratio_
    cumvar = np.cumsum(var_ratio)
    print(f"\n{'Components':>10}  {'Var @k':>10}  {'Cumulative':>11}")
    print("-" * 35)
    for k in [5, 8, 10, 12, 15, 20, 25, 30]:
        if k <= len(cumvar):
            print(f"{k:>10}  {var_ratio[k-1]:>10.4f}  {cumvar[k-1]:>11.4f}")
    print(f"\nFull scree (all components):")
    for i, v in enumerate(var_ratio[:30]):
        bar = '#' * int(v * 200)
        print(f"  PC{i+1:02d}: {v:.4f}  cum={cumvar[i]:.4f}  {bar}")

print("\nRecommended PCA for ACS Employment (disability framing):")
for threshold in [0.80, 0.85, 0.90, 0.95]:
    k = int(np.searchsorted(cumvar, threshold)) + 1
    print(f"  {int(threshold*100)}% variance: k={k}")
PYEOF
