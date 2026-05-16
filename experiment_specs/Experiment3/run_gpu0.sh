#!/bin/bash
# Oneida GPU 0 — Census baselines (all 5 seeds) then FORGE best config seeds 2&3
set -e
source ~/envs/rl/bin/activate
cd ~/cs_9170_project

SPECS=experiment_specs/Experiment3
LOGS=$SPECS/logs
mkdir -p "$LOGS"

echo "[$(date)] === Census Baselines ==="

for method in gdro flb smote ot_repair ctgan fairtabddpm; do
    echo "[$(date)] Starting census_$method"
    python -u run_baseline.py \
        --spec $SPECS/baselines/census_${method}.yaml \
        --device cuda:0 \
        2>&1 | tee -a "$LOGS/census_${method}.log"
    echo "[$(date)] Finished census_$method"
done

echo "[$(date)] === Census FORGE best (seeds 2 & 3) ==="
python -u main.py \
    --spec $SPECS/census_forge_best_seeds23.yaml \
    --device cuda:0 \
    2>&1 | tee -a "$LOGS/census_forge_seeds23.log"

echo "[$(date)] GPU 0 complete."
