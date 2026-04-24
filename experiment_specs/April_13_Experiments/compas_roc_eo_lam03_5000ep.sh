#!/bin/bash
#SBATCH --job-name=cmp_eo_l03
#SBATCH --account=def-mcapretz
#SBATCH --time=12:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/April_13_Experiments/logs/compas_roc_eo_lam03_5000ep.out
#SBATCH --error=experiment_specs/April_13_Experiments/logs/compas_roc_eo_lam03_5000ep.err

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
mkdir -p experiment_specs/April_13_Experiments/logs

python -u main.py --spec experiment_specs/April_13_Experiments/compas_roc_eo_lam03_5000ep.yaml\1 --device cuda:0
