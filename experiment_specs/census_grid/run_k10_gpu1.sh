#!/bin/bash
# GPU 1: census k=10 grid specs (r04_e30, r06_e10, r06_e20, r06_e30)
set -euo pipefail
source ~/envs/rl/bin/activate
LOGDIR=/storage_1/epigou_storage/FORGE/training_runs_k10/logs
OUTDIR=/storage_1/epigou_storage/FORGE/training_runs_k10
SPECS=(
  experiment_specs/census_grid/census_k10_r04_e30.yaml
  experiment_specs/census_grid/census_k10_r06_e10.yaml
  experiment_specs/census_grid/census_k10_r06_e20.yaml
  experiment_specs/census_grid/census_k10_r06_e30.yaml
)
cd "$(dirname "$0")/../.."
for spec in "${SPECS[@]}"; do
  name=$(basename "$spec" .yaml)
  echo "[$(date)] Starting $name on cuda:1"
  python -u main.py --spec "$spec" --device cuda:1 --output_dir "$OUTDIR" \
    > "$LOGDIR/${name}.out" 2> "$LOGDIR/${name}.err"
  echo "[$(date)] Finished $name"
done
echo "GPU 1 batch complete."
