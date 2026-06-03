#!/bin/bash
#SBATCH --job-name=census_k10_test2
#SBATCH --account=rrg-kgroling
#SBATCH --time=2:00:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=27
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/census_grid/logs/census_k10_test2.out
#SBATCH --error=experiment_specs/census_grid/logs/census_k10_test2.err

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
mkdir -p experiment_specs/census_grid/logs

python main.py --spec experiment_specs/census_grid/census_k10_test2.yaml --parallel --device cuda:0
