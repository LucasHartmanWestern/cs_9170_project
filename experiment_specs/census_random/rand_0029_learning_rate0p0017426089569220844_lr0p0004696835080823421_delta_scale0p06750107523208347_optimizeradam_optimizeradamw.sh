#!/bin/bash
#SBATCH --job-name=rand_0029_learni
#SBATCH --account=def-mcapretz
#SBATCH --time=30:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/census_random/logs/rand_0029_learning_rate0p0017426089569220844_lr0p0004696835080823421_delta_scale0p06750107523208347_optimizeradam_optimizeradamw.out
#SBATCH --error=experiment_specs/census_random/logs/rand_0029_learning_rate0p0017426089569220844_lr0p0004696835080823421_delta_scale0p06750107523208347_optimizeradam_optimizeradamw.err

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

python -u main.py --spec experiment_specs/census_random/rand_0029_learning_rate0p0017426089569220844_lr0p0004696835080823421_delta_scale0p06750107523208347_optimizeradam_optimizeradamw.yaml --device cpu
