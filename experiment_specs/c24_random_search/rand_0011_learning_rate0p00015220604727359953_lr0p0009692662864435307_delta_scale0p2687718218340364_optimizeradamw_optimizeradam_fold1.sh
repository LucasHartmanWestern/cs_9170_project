#!/bin/bash
#SBATCH --job-name=rand_0011_learni
#SBATCH --account=def-mcapretz
#SBATCH --time=10:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/c24_random_search/logs/rand_0011_learning_rate0p00015220604727359953_lr0p0009692662864435307_delta_scale0p2687718218340364_optimizeradamw_optimizeradam_fold1.out
#SBATCH --error=experiment_specs/c24_random_search/logs/rand_0011_learning_rate0p00015220604727359953_lr0p0009692662864435307_delta_scale0p2687718218340364_optimizeradamw_optimizeradam_fold1.err

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
mkdir -p experiment_specs/c24_random_search/logs

python -u main.py --spec experiment_specs/c24_random_search/rand_0011_learning_rate0p00015220604727359953_lr0p0009692662864435307_delta_scale0p2687718218340364_optimizeradamw_optimizeradam_fold1.yaml --device cuda:0
