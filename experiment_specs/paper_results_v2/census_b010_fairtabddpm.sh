#!/bin/bash
#SBATCH --job-name=census_b010_ftddpm
#SBATCH --account=def-mcapretz
#SBATCH --time=01:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/paper_results_v2/logs/census_b010_fairtabddpm.out
#SBATCH --error=experiment_specs/paper_results_v2/logs/census_b010_fairtabddpm.err

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
mkdir -p experiment_specs/paper_results_v2/logs

python -u run_baseline.py --spec experiment_specs/paper_results_v2/census_b010_fairtabddpm.json --device cuda:0
