#!/bin/bash
# EXP-046 fold 2/3 — Lambda cuda:1  (cuda:0 occupied by mdanish Jupyter kernels)
# Launch: bash experiment_specs/capture24_kfold_grid/run_fold2_lambda.sh
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
    --spec experiment_specs/capture24_kfold_grid/fold2.yaml \
    --device cuda:1 \
    --parallel \
    > experiment_specs/capture24_kfold_grid/logs/fold2_lambda.out \
    2> experiment_specs/capture24_kfold_grid/logs/fold2_lambda.err &

echo "EXP-046 fold2 launched on Lambda cuda:1, PID=$!"
echo "Log: experiment_specs/capture24_kfold_grid/logs/fold2_lambda.out"
