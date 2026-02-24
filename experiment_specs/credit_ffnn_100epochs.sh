#!/bin/bash
#SBATCH --job-name=credit_ffnn_100epochs
#SBATCH --account=def-mcapretz
#SBATCH --time=84:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/logs/credit_ffnn_100epochs.out
#SBATCH --error=experiment_specs/logs/credit_ffnn_100epochs.err

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

python -u main.py --spec experiment_specs/credit_ffnn_100epochs.json --device cuda:0
