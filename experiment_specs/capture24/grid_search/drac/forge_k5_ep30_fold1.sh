#!/bin/bash
#SBATCH --job-name=c24_k5_e30_f1
#SBATCH --account=def-mcapretz
#SBATCH --time=15:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/c24_grid_search/drac/logs/forge_k5_ep30_fold1.out
#SBATCH --error=experiment_specs/c24_grid_search/drac/logs/forge_k5_ep30_fold1.err

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
mkdir -p experiment_specs/c24_grid_search/drac/logs

python -u main.py --spec experiment_specs/c24_grid_search/drac/forge_k5_ep30_fold1.yaml --device cuda:0
