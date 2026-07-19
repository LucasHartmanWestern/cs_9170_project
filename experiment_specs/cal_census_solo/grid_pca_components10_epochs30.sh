#!/bin/bash
#SBATCH --job-name=grid_pca_compone
#SBATCH --account=rrg-kgroling
#SBATCH --time=03:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/cal_census_solo/logs/grid_pca_components10_epochs30.out
#SBATCH --error=experiment_specs/cal_census_solo/logs/grid_pca_components10_epochs30.err

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
mkdir -p experiment_specs/cal_census_solo/logs

python -u main.py --spec experiment_specs/cal_census_solo/grid_pca_components10_epochs30.yaml --device cuda:0
