#!/bin/bash
#SBATCH --job-name=rand_0003_learni
#SBATCH --account=def-mcapretz
#SBATCH --time=30:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/census_random/logs/rand_0003_learning_rate0p003247943069996102_lr0p0001724797020759017_delta_scale0p11262658534061014_optimizersgd_optimizeradam.out
#SBATCH --error=experiment_specs/census_random/logs/rand_0003_learning_rate0p003247943069996102_lr0p0001724797020759017_delta_scale0p11262658534061014_optimizersgd_optimizeradam.err

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

python -u main.py --spec experiment_specs/census_random/rand_0003_learning_rate0p003247943069996102_lr0p0001724797020759017_delta_scale0p11262658534061014_optimizersgd_optimizeradam.yaml --device cpu
