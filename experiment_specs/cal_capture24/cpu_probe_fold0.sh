#!/bin/bash
#SBATCH --job-name=cal_cap24_cpu
#SBATCH --account=def-mcapretz
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=experiment_specs/cal_capture24/logs/cpu_probe_fold0.out
#SBATCH --error=experiment_specs/cal_capture24/logs/cpu_probe_fold0.err

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
mkdir -p experiment_specs/cal_capture24/logs

# CPU-only capture24 timing probe: paper fold-0 config (pca15/ep10/traj1000/
# real4000) cut to 250 episodes to measure per-episode wall time on a CPU node,
# so we can size the capture24 k-fold re-run the same way we did census.
python -u main.py --spec experiment_specs/cal_capture24/cpu_probe_fold0.yaml --device cpu
