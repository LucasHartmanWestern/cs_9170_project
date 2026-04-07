#!/bin/bash
#SBATCH --job-name=c24_nobias_ours
#SBATCH --account=def-mcapretz
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/paper_results_v2/logs/c24_nobias_ours.out
#SBATCH --error=experiment_specs/paper_results_v2/logs/c24_nobias_ours.err

set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/paper_results_v2/logs

python -u main.py --spec experiment_specs/paper_results_v2/c24_nobias_ours.json --device cuda:0
