#!/bin/bash
# EXP-046 fold 3/5 (fold_rng_seed=190) — Oneida cuda:1
# Launch: bash experiment_specs/capture24_kfold_grid/run_fold3_oneida.sh
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
    --spec experiment_specs/capture24_kfold_grid/fold3.yaml \
    --device cuda:1 \
    --parallel \
    > experiment_specs/capture24_kfold_grid/logs/fold3_oneida.out \
    2> experiment_specs/capture24_kfold_grid/logs/fold3_oneida.err &

echo "EXP-046 fold3 launched on Oneida cuda:1, PID=$!"
echo "Log: experiment_specs/capture24_kfold_grid/logs/fold3_oneida.out"
