#!/bin/bash
#SBATCH --job-name=diab_ctgan
#SBATCH --account=def-mcapretz
#SBATCH --time=6:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/diabetes130/baselines/logs/diabetes130_ctgan.out
#SBATCH --error=experiment_specs/diabetes130/baselines/logs/diabetes130_ctgan.err

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
mkdir -p experiment_specs/diabetes130/baselines/logs

python -u run_baseline.py --spec experiment_specs/diabetes130/baselines/diabetes130_ctgan.yaml\1 --device cuda:0
