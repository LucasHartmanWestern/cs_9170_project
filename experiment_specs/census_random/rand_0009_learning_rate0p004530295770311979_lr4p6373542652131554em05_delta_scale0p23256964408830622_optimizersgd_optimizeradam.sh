#!/bin/bash
#SBATCH --job-name=rand_0009_learni
#SBATCH --account=def-mcapretz
#SBATCH --time=17:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/census_random/logs/rand_0009_learning_rate0p004530295770311979_lr4p6373542652131554em05_delta_scale0p23256964408830622_optimizersgd_optimizeradam.out
#SBATCH --error=experiment_specs/census_random/logs/rand_0009_learning_rate0p004530295770311979_lr4p6373542652131554em05_delta_scale0p23256964408830622_optimizersgd_optimizeradam.err

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
mkdir -p experiment_specs/census_random/logs

python -u main.py --spec experiment_specs/census_random/rand_0009_learning_rate0p004530295770311979_lr4p6373542652131554em05_delta_scale0p23256964408830622_optimizersgd_optimizeradam.json --device cuda:0
