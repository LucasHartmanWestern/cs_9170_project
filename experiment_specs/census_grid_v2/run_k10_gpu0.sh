#!/bin/bash
# GPU 0: census k=10 grid specs (r02_e20, r02_e30, r04_e10, r04_e20)
set -euo pipefail
source ~/envs/rl/bin/activate
LOGDIR=/storage_1/epigou_storage/FORGE/training_runs_k10/logs
OUTDIR=/storage_1/epigou_storage/FORGE/training_runs_k10
SPECS=(
  experiment_specs/census_grid_v2/census_k10_r02_e20.yaml
  experiment_specs/census_grid_v2/census_k10_r02_e30.yaml
  experiment_specs/census_grid_v2/census_k10_r04_e10.yaml
  experiment_specs/census_grid_v2/census_k10_r04_e20.yaml
)
cd "$(dirname "$0")/../.."
for spec in "${SPECS[@]}"; do
  name=$(basename "$spec" .yaml)
  echo "[$(date)] Starting $name on cuda:0"
  python -u main.py --spec "$spec" --device cuda:0 --output_dir "$OUTDIR" \
    > "$LOGDIR/${name}.out" 2> "$LOGDIR/${name}.err"
  echo "[$(date)] Finished $name"
done
echo "GPU 0 batch complete."
