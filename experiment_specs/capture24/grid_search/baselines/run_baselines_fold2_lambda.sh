#!/bin/bash
# EXP-046 baselines fold 2/3 — Lambda cuda:1
# 4 lightweight baselines (GroupDRO, FLB, SMOTE, OT Repair), pca=15, seed=20
# Launch: bash experiment_specs/c24_grid_search/baselines/run_baselines_fold2_lambda.sh
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
  experiment_specs/c24_grid_search/baselines/group_dro_fold2.yaml
  experiment_specs/c24_grid_search/baselines/fairness_loss_balancing_fold2.yaml
  experiment_specs/c24_grid_search/baselines/smote_fold2.yaml
  experiment_specs/c24_grid_search/baselines/gaussian_ot_repair_fold2.yaml
)

for spec in "${SPECS[@]}"; do
  name=$(basename $spec .yaml)
  echo "[$(date)] Starting $name ..."
  python -u run_baseline.py --spec $spec --device cuda:1 \
    > $LOGDIR/${name}.out 2> $LOGDIR/${name}.err
  echo "[$(date)] Done $name"
done

echo "All fold2 baselines complete."
