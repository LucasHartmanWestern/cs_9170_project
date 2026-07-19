#!/bin/bash
#SBATCH --job-name=grid_global_sigm
#SBATCH --account=def-mcapretz
#SBATCH --time=24:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/census_stage1/logs/grid_global_sigmoid_k0_pca_components5_epochs20_traj_length4500.out
#SBATCH --error=experiment_specs/census_stage1/logs/grid_global_sigmoid_k0_pca_components5_epochs20_traj_length4500.err

set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

module purge
module load python/3.12.4
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/census_stage1/logs

python -u main.py --spec experiment_specs/census_stage1/grid_global_sigmoid_k0_pca_components5_epochs20_traj_length4500.yaml --device cpu
