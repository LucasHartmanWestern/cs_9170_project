#!/bin/bash
#SBATCH --job-name=rand_0024_learni
#SBATCH --account=def-mcapretz
#SBATCH --time=08:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/capture24_random/logs/rand_0024_learning_rate0p005136239175673676_lr2p7387167512217737em05_delta_scale0p2542800757547208_optimizersgd_optimizeradam_fold2.out
#SBATCH --error=experiment_specs/capture24_random/logs/rand_0024_learning_rate0p005136239175673676_lr2p7387167512217737em05_delta_scale0p2542800757547208_optimizersgd_optimizeradam_fold2.err

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

python -u main.py --spec experiment_specs/capture24_random/rand_0024_learning_rate0p005136239175673676_lr2p7387167512217737em05_delta_scale0p2542800757547208_optimizersgd_optimizeradam_fold2.yaml --device cpu
