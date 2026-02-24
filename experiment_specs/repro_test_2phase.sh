#!/bin/bash
#SBATCH --job-name=repro_test_2phase
#SBATCH --account=def-mcapretz
#SBATCH --time=0:30:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/logs/repro_test_2phase.out
#SBATCH --error=experiment_specs/logs/repro_test_2phase.err

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

python -u main.py --spec experiment_specs/repro_test_2phase.json --device cuda:0
