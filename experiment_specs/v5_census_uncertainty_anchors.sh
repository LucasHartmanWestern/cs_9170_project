#!/bin/bash
#SBATCH --job-name=v5_census_unc_anchors
#SBATCH --account=def-mcapretz
#SBATCH --time=14:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/logs/v5_census_uncertainty_anchors.out
#SBATCH --error=experiment_specs/logs/v5_census_uncertainty_anchors.err

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

mkdir -p experiment_specs/logs

python -u main.py --spec experiment_specs/v5_census_uncertainty_anchors.json --device cuda:0
