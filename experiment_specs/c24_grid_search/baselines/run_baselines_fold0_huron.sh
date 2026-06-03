#!/bin/bash
# EXP-046 baselines fold 0/5 — Huron cuda:0
# 4 lightweight baselines (GroupDRO, FLB, SMOTE, OT Repair)
# Settings: fold_idx=0, n_folds=5, fold_rng_seed=6, seed=42, pca=15, real=4000
# Launch: bash experiment_specs/c24_grid_search/baselines/run_baselines_fold0_huron.sh
set -eo pipefail

cd ~/cs_9170_project
source ~/envs/rl/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

LOGDIR=experiment_specs/c24_grid_search/baselines/logs
mkdir -p $LOGDIR

SPECS=(
  experiment_specs/c24_grid_search/baselines/group_dro_fold0.yaml
  experiment_specs/c24_grid_search/baselines/fairness_loss_balancing_fold0.yaml
  experiment_specs/c24_grid_search/baselines/smote_fold0.yaml
  experiment_specs/c24_grid_search/baselines/gaussian_ot_repair_fold0.yaml
)

for spec in "${SPECS[@]}"; do
  name=$(basename $spec .yaml)
  echo "[$(date)] Starting $name ..."
  python -u run_baseline.py --spec $spec --device cuda:0 \
    > $LOGDIR/${name}_huron.out 2> $LOGDIR/${name}_huron.err
  echo "[$(date)] Done $name"
done

echo "All fold0 baselines complete."
