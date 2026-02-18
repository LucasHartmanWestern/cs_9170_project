#!/bin/bash
#SBATCH --job-name=pamap2_1ph
#SBATCH --account=def-mcapretz
#SBATCH --time=13:00:00
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

python -u main.py --spec experiment_specs/pamap2_nopca_nobias_1phase.json --device cuda:0
