#!/bin/bash
# EXP-046 baselines fold 1/3 — Aulavik cuda:0
# 4 lightweight baselines (GroupDRO, FLB, SMOTE, OT Repair), pca=15, seed=20
# Launch: bash experiment_specs/capture24_kfold_grid/baselines/run_baselines_fold1_aulavik.sh
set -eo pipefail

cd ~/cs_9170_project
source ~/envs/rl/bin/activate

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

LOGDIR=experiment_specs/capture24_kfold_grid/baselines/logs
mkdir -p $LOGDIR

SPECS=(
  experiment_specs/capture24_kfold_grid/baselines/group_dro_fold1.yaml
  experiment_specs/capture24_kfold_grid/baselines/fairness_loss_balancing_fold1.yaml
  experiment_specs/capture24_kfold_grid/baselines/smote_fold1.yaml
  experiment_specs/capture24_kfold_grid/baselines/gaussian_ot_repair_fold1.yaml
)

for spec in "${SPECS[@]}"; do
  name=$(basename $spec .yaml)
  echo "[$(date)] Starting $name ..."
  python -u run_baseline.py --spec $spec --device cuda:0 \
    > $LOGDIR/${name}.out 2> $LOGDIR/${name}.err
  echo "[$(date)] Done $name"
done

echo "All fold1 baselines complete."
