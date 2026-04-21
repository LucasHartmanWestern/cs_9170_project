#!/bin/bash
#SBATCH --job-name=grid_global_sigm
#SBATCH --account=rrg-kgroling
#SBATCH --time=17:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/census_grid/logs/grid_global_sigmoid_k3_pca_components15_epochs40_traj_length4000_real_data_size1000.out
#SBATCH --error=experiment_specs/census_grid/logs/grid_global_sigmoid_k3_pca_components15_epochs40_traj_length4000_real_data_size1000.err

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
mkdir -p experiment_specs/census_grid/logs

python -u main.py --spec experiment_specs/census_grid/grid_global_sigmoid_k3_pca_components15_epochs40_traj_length4000_real_data_size1000.json --device cuda:0
