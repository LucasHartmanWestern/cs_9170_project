#!/bin/bash
# EXP-046 smoke test — fold 0/5, pca=15 ep=10 ratio=0.2 — Oneida cuda:0
# Succeeds if beta-EO < alpha-EO. If yes, proceed to full grid.
# Launch: bash experiment_specs/capture24_kfold_grid/run_test_fold0_oneida.sh
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
    --spec experiment_specs/capture24_kfold_grid/test_fold0.yaml \
    --device cuda:0 \
    > experiment_specs/capture24_kfold_grid/logs/test_fold0_oneida.out \
    2> experiment_specs/capture24_kfold_grid/logs/test_fold0_oneida.err &

echo "EXP-046 test fold0 launched on Oneida cuda:0, PID=$!"
echo "Log: experiment_specs/capture24_kfold_grid/logs/test_fold0_oneida.out"
