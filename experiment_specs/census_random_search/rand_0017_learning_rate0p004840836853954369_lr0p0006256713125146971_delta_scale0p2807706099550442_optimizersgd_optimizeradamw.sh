#!/bin/bash
#SBATCH --job-name=rand_0017_learni
#SBATCH --account=def-mcapretz
#SBATCH --time=17:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/census_random_search/logs/rand_0017_learning_rate0p004840836853954369_lr0p0006256713125146971_delta_scale0p2807706099550442_optimizersgd_optimizeradamw.out
#SBATCH --error=experiment_specs/census_random_search/logs/rand_0017_learning_rate0p004840836853954369_lr0p0006256713125146971_delta_scale0p2807706099550442_optimizersgd_optimizeradamw.err

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
mkdir -p experiment_specs/census_random_search/logs

python -u main.py --spec experiment_specs/census_random_search/rand_0017_learning_rate0p004840836853954369_lr0p0006256713125146971_delta_scale0p2807706099550442_optimizersgd_optimizeradamw.yaml --device cuda:0
