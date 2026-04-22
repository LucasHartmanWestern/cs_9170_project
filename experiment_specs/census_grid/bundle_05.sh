#!/bin/bash
#SBATCH --job-name=bundle_05
#SBATCH --account=rrg-kgroling
#SBATCH --time=17:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=experiment_specs/census_grid/logs/bundle_05.out
#SBATCH --error=experiment_specs/census_grid/logs/bundle_05.err

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

python -u main.py --spec experiment_specs/census_grid/grid_global_sigmoid_k5_pca_components5_epochs40_traj_length1000_real_data_size4000.json --device cuda:0 > experiment_specs/census_grid/logs/grid_global_sigmoid_k5_pca_components5_epochs40_traj_length1000_real_data_size4000.out 2> experiment_specs/census_grid/logs/grid_global_sigmoid_k5_pca_components5_epochs40_traj_length1000_real_data_size4000.err &
python -u main.py --spec experiment_specs/census_grid/grid_global_sigmoid_k5_pca_components10_epochs10_traj_length1000_real_data_size4000.json --device cuda:0 > experiment_specs/census_grid/logs/grid_global_sigmoid_k5_pca_components10_epochs10_traj_length1000_real_data_size4000.out 2> experiment_specs/census_grid/logs/grid_global_sigmoid_k5_pca_components10_epochs10_traj_length1000_real_data_size4000.err &
python -u main.py --spec experiment_specs/census_grid/grid_global_sigmoid_k5_pca_components10_epochs20_traj_length1000_real_data_size4000.json --device cuda:0 > experiment_specs/census_grid/logs/grid_global_sigmoid_k5_pca_components10_epochs20_traj_length1000_real_data_size4000.out 2> experiment_specs/census_grid/logs/grid_global_sigmoid_k5_pca_components10_epochs20_traj_length1000_real_data_size4000.err &
python -u main.py --spec experiment_specs/census_grid/grid_global_sigmoid_k5_pca_components10_epochs40_traj_length1000_real_data_size4000.json --device cuda:0 > experiment_specs/census_grid/logs/grid_global_sigmoid_k5_pca_components10_epochs40_traj_length1000_real_data_size4000.out 2> experiment_specs/census_grid/logs/grid_global_sigmoid_k5_pca_components10_epochs40_traj_length1000_real_data_size4000.err &
wait
echo "Bundle bundle_05 complete."
