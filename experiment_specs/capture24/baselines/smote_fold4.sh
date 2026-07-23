#!/bin/bash
#SBATCH --job-name=blc_smote_fold4
#SBATCH --account=def-mcapretz
#SBATCH --time=08:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/capture24/baselines/logs/smote_fold4.out
#SBATCH --error=experiment_specs/capture24/baselines/logs/smote_fold4.err
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
mkdir -p experiment_specs/capture24/baselines/logs
python -u run_baseline.py --spec experiment_specs/capture24/baselines/smote_fold4.yaml --device cpu
