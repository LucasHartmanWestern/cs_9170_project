#!/bin/bash
#SBATCH --job-name=c24_sex_ep2000ph600
#SBATCH --account=def-mcapretz
#SBATCH --time=18:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/capture24_sex_feature/logs/c24_sex_ep2000ph600.out
#SBATCH --error=experiment_specs/capture24_sex_feature/logs/c24_sex_ep2000ph600.err

set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/capture24_sex_feature/logs

python -u main.py --spec experiment_specs/capture24_sex_feature/c24_sex_ep2000ph600.json --device cuda:0 --output_dir paper_results_v4/training_runs
