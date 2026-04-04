#!/bin/bash
#SBATCH --job-name=v25a_cap24_s05
#SBATCH --account=def-mcapretz
#SBATCH --time=06:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/logs/v25a_capture24_cmaes_sigma05_5s.out
#SBATCH --error=experiment_specs/logs/v25a_capture24_cmaes_sigma05_5s.err

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
mkdir -p experiment_specs/logs

python -u main.py --spec experiment_specs/v25a_capture24_cmaes_sigma05_5s.json --device cuda:0 --output_dir paper_results_v4
