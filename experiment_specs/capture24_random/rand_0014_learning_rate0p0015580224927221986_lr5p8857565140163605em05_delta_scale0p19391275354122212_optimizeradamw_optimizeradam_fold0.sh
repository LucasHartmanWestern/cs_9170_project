#!/bin/bash
#SBATCH --job-name=rand_0014_learni
#SBATCH --account=def-mcapretz
#SBATCH --time=08:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/capture24_random/logs/rand_0014_learning_rate0p0015580224927221986_lr5p8857565140163605em05_delta_scale0p19391275354122212_optimizeradamw_optimizeradam_fold0.out
#SBATCH --error=experiment_specs/capture24_random/logs/rand_0014_learning_rate0p0015580224927221986_lr5p8857565140163605em05_delta_scale0p19391275354122212_optimizeradamw_optimizeradam_fold0.err

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

python -u main.py --spec experiment_specs/capture24_random/rand_0014_learning_rate0p0015580224927221986_lr5p8857565140163605em05_delta_scale0p19391275354122212_optimizeradamw_optimizeradam_fold0.yaml --device cpu
