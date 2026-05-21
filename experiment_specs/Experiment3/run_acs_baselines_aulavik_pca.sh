#!/bin/bash
# ACS Employment baselines in PCA-10 space — Aulavik GPU1 (cuda:1).
# Matches FORGE feature space (use_pca=true, pca_components=10).
# Usage: bash experiment_specs/Experiment3/run_acs_baselines_aulavik_pca.sh
set -e
source ~/envs/rl/bin/activate
cd ~/cs_9170_project

SPECS=experiment_specs/Experiment3/baselines
LOGS=$SPECS/logs
mkdir -p "$LOGS"

run_baseline() {
    local spec=$1
    local device=$2
    local name=$(basename $spec .yaml)
    echo "[$(date)] Starting $name on $device"
    python -u run_baseline.py --spec $spec --device $device \
        2>&1 | tee "$LOGS/${name}_aulavik.log"
    echo "[$(date)] Done: $name"
}

# Light baselines in parallel on cuda:1
run_baseline $SPECS/acs_gdro_pca.yaml      cuda:1 &
run_baseline $SPECS/acs_flb_pca.yaml       cuda:1 &
run_baseline $SPECS/acs_smote_pca.yaml     cuda:1 &
run_baseline $SPECS/acs_ot_repair_pca.yaml cuda:1 &
wait
echo "[$(date)] Light PCA baselines complete. Starting generative baselines."

# Heavy generative baselines sequentially to avoid OOM
run_baseline $SPECS/acs_ctgan_pca.yaml       cuda:1
run_baseline $SPECS/acs_fairtabddpm_pca.yaml cuda:1

echo "[$(date)] All ACS PCA baselines complete (Aulavik GPU1)."
