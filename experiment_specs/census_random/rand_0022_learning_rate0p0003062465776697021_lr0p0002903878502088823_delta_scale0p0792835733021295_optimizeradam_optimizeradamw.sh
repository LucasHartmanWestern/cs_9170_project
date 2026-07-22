#!/bin/bash
#SBATCH --job-name=rand_0022_learni
#SBATCH --account=def-mcapretz
#SBATCH --time=30:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/census_random/logs/rand_0022_learning_rate0p0003062465776697021_lr0p0002903878502088823_delta_scale0p0792835733021295_optimizeradam_optimizeradamw.out
#SBATCH --error=experiment_specs/census_random/logs/rand_0022_learning_rate0p0003062465776697021_lr0p0002903878502088823_delta_scale0p0792835733021295_optimizeradam_optimizeradamw.err

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

python -u main.py --spec experiment_specs/census_random/rand_0022_learning_rate0p0003062465776697021_lr0p0002903878502088823_delta_scale0p0792835733021295_optimizeradam_optimizeradamw.yaml --device cpu
