#!/bin/bash
# EXP-046 smoke test fold 2/5 + baselines — Huron cuda:1
# FORGE + 4 lightweight baselines launched in parallel
# Settings: fold_idx=2, n_folds=5, fold_rng_seed=6, seed=42, pca=15, real=4000
# Launch: bash experiment_specs/capture24_kfold_grid/run_smoke_fold2_huron.sh
set -eo pipefail

cd ~/cs_9170_project
source ~/envs/rl/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

LOGDIR=experiment_specs/capture24_kfold_grid/logs
BLOG=experiment_specs/capture24_kfold_grid/baselines/logs
mkdir -p $LOGDIR $BLOG

echo "[$(date)] Launching FORGE fold2 on cuda:1"
nohup python -u main.py \
    --spec experiment_specs/capture24_kfold_grid/test_fold2.yaml \
    --device cuda:1 \
    > $LOGDIR/test_fold2_huron_gpu1.out \
    2> $LOGDIR/test_fold2_huron_gpu1.err &
FORGE_PID=$!
echo "FORGE fold2 PID=$FORGE_PID"

echo "[$(date)] Launching 4 baselines fold2 on cuda:1 (sequential)"
for spec in \
    experiment_specs/capture24_kfold_grid/baselines/group_dro_fold2.yaml \
    experiment_specs/capture24_kfold_grid/baselines/fairness_loss_balancing_fold2.yaml \
    experiment_specs/capture24_kfold_grid/baselines/smote_fold2.yaml \
    experiment_specs/capture24_kfold_grid/baselines/gaussian_ot_repair_fold2.yaml; do
  name=$(basename $spec .yaml)
  echo "[$(date)] Starting $name"
  python -u run_baseline.py --spec $spec --device cuda:1 \
    > $BLOG/${name}_huron.out \
    2> $BLOG/${name}_huron.err
  echo "[$(date)] Done $name"
done &
BASELINE_PID=$!
echo "Baseline chain fold2 PID=$BASELINE_PID"

echo "Waiting for FORGE fold2 (PID=$FORGE_PID)..."
wait $FORGE_PID
echo "[$(date)] FORGE fold2 complete."
wait $BASELINE_PID
echo "[$(date)] All fold2 runs complete."
