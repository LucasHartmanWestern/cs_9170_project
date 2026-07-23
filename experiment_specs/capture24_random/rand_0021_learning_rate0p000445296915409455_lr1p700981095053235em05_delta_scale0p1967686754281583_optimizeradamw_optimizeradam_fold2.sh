#!/bin/bash
#SBATCH --job-name=rand_0021_learni
#SBATCH --account=def-mcapretz
#SBATCH --time=08:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/capture24_random/logs/rand_0021_learning_rate0p000445296915409455_lr1p700981095053235em05_delta_scale0p1967686754281583_optimizeradamw_optimizeradam_fold2.out
#SBATCH --error=experiment_specs/capture24_random/logs/rand_0021_learning_rate0p000445296915409455_lr1p700981095053235em05_delta_scale0p1967686754281583_optimizeradamw_optimizeradam_fold2.err

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

python -u main.py --spec experiment_specs/capture24_random/rand_0021_learning_rate0p000445296915409455_lr1p700981095053235em05_delta_scale0p1967686754281583_optimizeradamw_optimizeradam_fold2.yaml --device cpu
