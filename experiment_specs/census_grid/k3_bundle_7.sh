#!/bin/bash
#SBATCH --job-name=k3_bundle_7
#SBATCH --account=rrg-kgroling
#SBATCH --time=48:00:00
#SBATCH --mem=9G
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/census_grid/logs/k3_bundle_7.out
#SBATCH --error=experiment_specs/census_grid/logs/k3_bundle_7.err

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
mkdir -p experiment_specs/census_grid/logs

python -u main.py --spec experiment_specs/census_grid/grid_global_sigmoid_k3_pca_components15_epochs10_traj_length3000_real_data_size2000.yaml --device cuda:0 > experiment_specs/census_grid/logs/grid_global_sigmoid_k3_pca_components15_epochs10_traj_length3000_real_data_size2000.out 2> experiment_specs/census_grid/logs/grid_global_sigmoid_k3_pca_components15_epochs10_traj_length3000_real_data_size2000.err &
python -u main.py --spec experiment_specs/census_grid/grid_global_sigmoid_k3_pca_components15_epochs20_traj_length3000_real_data_size2000.yaml --device cuda:0 > experiment_specs/census_grid/logs/grid_global_sigmoid_k3_pca_components15_epochs20_traj_length3000_real_data_size2000.out 2> experiment_specs/census_grid/logs/grid_global_sigmoid_k3_pca_components15_epochs20_traj_length3000_real_data_size2000.err &
python -u main.py --spec experiment_specs/census_grid/grid_global_sigmoid_k3_pca_components15_epochs40_traj_length3000_real_data_size2000.yaml --device cuda:0 > experiment_specs/census_grid/logs/grid_global_sigmoid_k3_pca_components15_epochs40_traj_length3000_real_data_size2000.out 2> experiment_specs/census_grid/logs/grid_global_sigmoid_k3_pca_components15_epochs40_traj_length3000_real_data_size2000.err &
wait
echo "Bundle k3_bundle_7 complete."
