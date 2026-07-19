#!/bin/bash
#SBATCH --job-name=bundle_0
#SBATCH --account=rrg-kgroling
#SBATCH --time=03:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/cal_census/logs/bundle_0.out
#SBATCH --error=experiment_specs/cal_census/logs/bundle_0.err

set -euo pipefail

export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module purge
module load python/3.12.4 cuda cudnn
source ~/envs/rl/bin/activate
mkdir -p experiment_specs/cal_census/logs

python -u main.py --spec experiment_specs/cal_census/grid_pca_components10_epochs20.yaml --device cuda:0 > experiment_specs/cal_census/logs/grid_pca_components10_epochs20.out 2> experiment_specs/cal_census/logs/grid_pca_components10_epochs20.err &
python -u main.py --spec experiment_specs/cal_census/grid_pca_components10_epochs30.yaml --device cuda:0 > experiment_specs/cal_census/logs/grid_pca_components10_epochs30.out 2> experiment_specs/cal_census/logs/grid_pca_components10_epochs30.err &
python -u main.py --spec experiment_specs/cal_census/grid_pca_components15_epochs20.yaml --device cuda:0 > experiment_specs/cal_census/logs/grid_pca_components15_epochs20.out 2> experiment_specs/cal_census/logs/grid_pca_components15_epochs20.err &
python -u main.py --spec experiment_specs/cal_census/grid_pca_components15_epochs30.yaml --device cuda:0 > experiment_specs/cal_census/logs/grid_pca_components15_epochs30.out 2> experiment_specs/cal_census/logs/grid_pca_components15_epochs30.err &
wait
echo "Bundle bundle_0 complete."
