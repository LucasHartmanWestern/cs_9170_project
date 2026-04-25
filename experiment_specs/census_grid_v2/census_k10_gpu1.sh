#!/bin/bash
#SBATCH --job-name=census_k10_gpu1
#SBATCH --account=rrg-kgroling
#SBATCH --time=168:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/census_grid_v2/logs/census_k10_gpu1.out
#SBATCH --error=experiment_specs/census_grid_v2/logs/census_k10_gpu1.err

set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module purge
module load python/3.10
source $SLURM_TMPDIR/env/bin/activate

python main.py --spec experiment_specs/census_grid_v2/census_k10_gpu1.yaml --parallel --device cuda:0
