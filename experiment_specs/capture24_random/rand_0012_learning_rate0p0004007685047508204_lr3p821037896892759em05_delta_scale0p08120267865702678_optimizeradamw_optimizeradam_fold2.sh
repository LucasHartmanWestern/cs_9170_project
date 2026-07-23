#!/bin/bash
#SBATCH --job-name=rand_0012_learni
#SBATCH --account=def-mcapretz
#SBATCH --time=08:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/capture24_random/logs/rand_0012_learning_rate0p0004007685047508204_lr3p821037896892759em05_delta_scale0p08120267865702678_optimizeradamw_optimizeradam_fold2.out
#SBATCH --error=experiment_specs/capture24_random/logs/rand_0012_learning_rate0p0004007685047508204_lr3p821037896892759em05_delta_scale0p08120267865702678_optimizeradamw_optimizeradam_fold2.err

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
mkdir -p experiment_specs/capture24_random/logs

python -u main.py --spec experiment_specs/capture24_random/rand_0012_learning_rate0p0004007685047508204_lr3p821037896892759em05_delta_scale0p08120267865702678_optimizeradamw_optimizeradam_fold2.yaml --device cpu
