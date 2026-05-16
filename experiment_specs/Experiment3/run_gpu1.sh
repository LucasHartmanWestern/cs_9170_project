#!/bin/bash
# Oneida GPU 1 — Capture24 baselines (all 5 seeds) then FORGE best config seeds 2&3
set -e
source ~/envs/rl/bin/activate
cd ~/cs_9170_project

SPECS=experiment_specs/Experiment3
LOGS=$SPECS/logs
mkdir -p "$LOGS"

echo "[$(date)] === Capture24 Baselines ==="

for method in gdro flb smote ot_repair ctgan fairtabddpm; do
    echo "[$(date)] Starting capture24_$method"
    python -u run_baseline.py \
        --spec $SPECS/baselines/capture24_${method}.yaml \
        --device cuda:1 \
        2>&1 | tee -a "$LOGS/capture24_${method}.log"
    echo "[$(date)] Finished capture24_$method"
done

echo "[$(date)] === Capture24 FORGE best (seeds 2 & 3) ==="
python -u main.py \
    --spec $SPECS/capture24_forge_best_seeds23.yaml \
    --device cuda:1 \
    2>&1 | tee -a "$LOGS/capture24_forge_seeds23.log"

echo "[$(date)] GPU 1 complete."
