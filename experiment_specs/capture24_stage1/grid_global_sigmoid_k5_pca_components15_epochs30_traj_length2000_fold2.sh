#!/bin/bash
#SBATCH --job-name=grid_global_sigm
#SBATCH --account=def-mcapretz
#SBATCH --time=10:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/capture24_stage1/logs/grid_global_sigmoid_k5_pca_components15_epochs30_traj_length2000_fold2.out
#SBATCH --error=experiment_specs/capture24_stage1/logs/grid_global_sigmoid_k5_pca_components15_epochs30_traj_length2000_fold2.err

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
mkdir -p experiment_specs/capture24_stage1/logs

python -u main.py --spec experiment_specs/capture24_stage1/grid_global_sigmoid_k5_pca_components15_epochs30_traj_length2000_fold2.yaml --device cpu
