#!/bin/bash
# EXP-046 fold 1/3 — Aulavik cuda:0
# Launch: bash experiment_specs/capture24_kfold_grid/run_fold1_aulavik.sh
set -eo pipefail

cd ~/cs_9170_project

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

source ~/envs/rl/bin/activate

mkdir -p experiment_specs/capture24_kfold_grid/logs

nohup python -u main.py \
    --spec experiment_specs/capture24_kfold_grid/fold1.yaml \
    --device cuda:0 \
    --parallel \
    > experiment_specs/capture24_kfold_grid/logs/fold1_aulavik.out \
    2> experiment_specs/capture24_kfold_grid/logs/fold1_aulavik.err &

echo "EXP-046 fold1 launched on Aulavik cuda:0, PID=$!"
echo "Log: experiment_specs/capture24_kfold_grid/logs/fold1_aulavik.out"
