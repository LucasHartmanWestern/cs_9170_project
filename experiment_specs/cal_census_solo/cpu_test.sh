#!/bin/bash
#SBATCH --job-name=cal_cpu_test
#SBATCH --account=def-mcapretz
#SBATCH --time=01:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/cal_census_solo/logs/cpu_test.out
#SBATCH --error=experiment_specs/cal_census_solo/logs/cpu_test.err

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
mkdir -p experiment_specs/cal_census_solo/logs

# CPU-only: no --gres=gpu, --device cpu. Same census pca10/ep30/traj2000/250ep
# config as the GPU-solo baseline (4.23 s/ep) for a direct wall-clock comparison.
python -u main.py --spec experiment_specs/cal_census_solo/grid_pca_components10_epochs30.yaml --device cpu
