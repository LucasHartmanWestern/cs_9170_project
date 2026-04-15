#!/bin/bash
#SBATCH --job-name=cmp_gdro
#SBATCH --account=def-mcapretz
#SBATCH --time=6:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/April_13_Experiments/baselines/logs/compas_gdro.out
#SBATCH --error=experiment_specs/April_13_Experiments/baselines/logs/compas_gdro.err

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
mkdir -p experiment_specs/April_13_Experiments/baselines/logs

python -u run_baseline.py --spec experiment_specs/April_13_Experiments/baselines/compas_gdro.json --device cuda:0
