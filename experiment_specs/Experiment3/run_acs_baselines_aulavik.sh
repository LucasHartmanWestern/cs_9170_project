#!/bin/bash
# Run all 6 ACS Employment baselines on Aulavik GPU1 (cuda:1).
# GPU0 is busy with capture24 k=0 gap-fills; baselines run immediately on GPU1.
# Usage: bash experiment_specs/Experiment3/run_acs_baselines_aulavik.sh
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
run_baseline $SPECS/acs_gdro.yaml      cuda:1 &
run_baseline $SPECS/acs_flb.yaml       cuda:1 &
run_baseline $SPECS/acs_smote.yaml     cuda:1 &
run_baseline $SPECS/acs_ot_repair.yaml cuda:1 &
wait
echo "[$(date)] Light baselines complete. Starting generative baselines."

# Heavy generative baselines sequentially to avoid OOM
run_baseline $SPECS/acs_ctgan.yaml       cuda:1
run_baseline $SPECS/acs_fairtabddpm.yaml cuda:1

echo "[$(date)] All ACS baselines complete (Aulavik GPU1)."
