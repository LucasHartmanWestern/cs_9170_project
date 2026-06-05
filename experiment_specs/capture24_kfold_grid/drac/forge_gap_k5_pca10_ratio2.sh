#!/bin/bash
#SBATCH --job-name=gc24_k5p10r2
#SBATCH --account=def-mcapretz
#SBATCH --time=24:00:00
#SBATCH --mem=9G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/capture24_kfold_grid/drac/logs/forge_gap_k5_pca10_ratio2.out
#SBATCH --error=experiment_specs/capture24_kfold_grid/drac/logs/forge_gap_k5_pca10_ratio2.err

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

python -u main.py --spec experiment_specs/capture24_kfold_grid/drac/forge_gap_k5_pca10_ratio2.yaml --device cuda:0
