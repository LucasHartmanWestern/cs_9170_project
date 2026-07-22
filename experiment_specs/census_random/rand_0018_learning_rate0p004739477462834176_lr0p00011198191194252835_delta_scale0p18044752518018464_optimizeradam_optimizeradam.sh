#!/bin/bash
#SBATCH --job-name=rand_0018_learni
#SBATCH --account=def-mcapretz
#SBATCH --time=20:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/census_random/logs/rand_0018_learning_rate0p004739477462834176_lr0p00011198191194252835_delta_scale0p18044752518018464_optimizeradam_optimizeradam.out
#SBATCH --error=experiment_specs/census_random/logs/rand_0018_learning_rate0p004739477462834176_lr0p00011198191194252835_delta_scale0p18044752518018464_optimizeradam_optimizeradam.err

set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

module purge
module load python/3.12.4
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/census_random/logs

python -u main.py --spec experiment_specs/census_random/rand_0018_learning_rate0p004739477462834176_lr0p00011198191194252835_delta_scale0p18044752518018464_optimizeradam_optimizeradam.yaml --device cpu
