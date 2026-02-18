#!/bin/bash
#SBATCH --job-name=credit_2ph
#SBATCH --account=def-mcapretz
#SBATCH --time=25:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1

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

python -u main.py --spec experiment_specs/credit_nopca_nobias_2phase.json --device cuda:0
