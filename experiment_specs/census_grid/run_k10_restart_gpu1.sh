#!/bin/bash
# Post-storm restart: GPU1 — r06_e20 PCA15 seed42, then r06_e30 (never submitted)
set -euo pipefail
source ~/envs/rl/bin/activate
LOGDIR=/storage_1/epigou_storage/FORGE/training_runs_k10/logs
OUTDIR=/storage_1/epigou_storage/FORGE/training_runs_k10
cd ~/cs_9170_project

echo "[$(date)] Starting r06_e20 PCA15 seed42 restart on cuda:1"
python -u main.py \
    --spec experiment_specs/census_grid/census_k10_r06_e20_pca15_seed42_restart.yaml \
    --device cuda:1 \
    --output_dir "$OUTDIR" \
    > "$LOGDIR/census_k10_r06_e20_pca15_seed42_restart.out" 2>&1

echo "[$(date)] Starting r06_e30 (first-ever run) on cuda:1"
python -u main.py \
    --spec experiment_specs/census_grid/census_k10_r06_e30.yaml \
    --device cuda:1 \
    --output_dir "$OUTDIR" \
    > "$LOGDIR/census_k10_r06_e30.out" 2>&1

echo "[$(date)] GPU1 restart complete."
