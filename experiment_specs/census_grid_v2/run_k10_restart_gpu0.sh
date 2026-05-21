#!/bin/bash
# Post-storm restart: GPU0 — r04_e20 PCA15 (seeds 0,1,42)
set -euo pipefail
source ~/envs/rl/bin/activate
LOGDIR=/storage_1/epigou_storage/FORGE/training_runs_k10/logs
OUTDIR=/storage_1/epigou_storage/FORGE/training_runs_k10
cd ~/cs_9170_project
echo "[$(date)] Starting r04_e20 PCA15 restart on cuda:0"
python -u main.py \
    --spec experiment_specs/census_grid_v2/census_k10_r04_e20_pca15_restart.yaml \
    --device cuda:0 \
    --output_dir "$OUTDIR" \
    > "$LOGDIR/census_k10_r04_e20_pca15_restart.out" 2>&1
echo "[$(date)] GPU0 restart complete."
