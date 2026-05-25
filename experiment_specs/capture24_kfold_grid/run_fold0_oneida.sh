#!/bin/bash
# EXP-046 fold 0/3 — Oneida cuda:0
# Launch: bash experiment_specs/capture24_kfold_grid/run_fold0_oneida.sh
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
    --spec experiment_specs/capture24_kfold_grid/fold0.yaml \
    --device cuda:0 \
    --parallel \
    > experiment_specs/capture24_kfold_grid/logs/fold0_oneida.out \
    2> experiment_specs/capture24_kfold_grid/logs/fold0_oneida.err &

echo "EXP-046 fold0 launched on Oneida cuda:0, PID=$!"
echo "Log: experiment_specs/capture24_kfold_grid/logs/fold0_oneida.out"
