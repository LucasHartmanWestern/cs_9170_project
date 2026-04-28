#!/bin/bash
#SBATCH --job-name=k10_r06_e10
#SBATCH --account=rrg-kgroling
#SBATCH --time=68:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/census_grid_v2/logs/k10_r06_e10.out
#SBATCH --error=experiment_specs/census_grid_v2/logs/k10_r06_e10.err

set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/census_grid_v2/logs

python main.py --spec experiment_specs/census_grid_v2/census_k10_r06_e10.yaml --parallel --device cuda:0
