#!/bin/bash
#SBATCH --job-name=c24_k5_e10_r4_f2
#SBATCH --account=def-mcapretz
#SBATCH --time=10:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/capture24_kfold_grid/drac/logs/forge_k5_ep10_ratio04_fold2.out
#SBATCH --error=experiment_specs/capture24_kfold_grid/drac/logs/forge_k5_ep10_ratio04_fold2.err

set -euo pipefail
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/capture24_kfold_grid/drac/logs

python -u main.py --spec experiment_specs/capture24_kfold_grid/drac/forge_k5_ep10_ratio04_fold2.yaml --device cuda:0
