#!/bin/bash
set -e
PROJECT="/home/epigou/cs_9170_project"
SPECS="$PROJECT/experiment_specs/April_13_Experiments"
LOG="$PROJECT/missing_seed1_gpu0.log"

cd "$PROJECT"
source ~/envs/rl/bin/activate

run() {
    echo "=== START: $1 ===" | tee -a "$LOG"
    date | tee -a "$LOG"
    python main.py --spec "$SPECS/$1" --device cuda:0 2>&1 | tee -a "$LOG"
    echo "=== DONE: $1 ===" | tee -a "$LOG"
    date | tee -a "$LOG"
    echo "" | tee -a "$LOG"
}

run "census_roc_eo_lam07_5000ep_seed1.json"
run "compas_roc_eo_lam03_5000ep_seed1.json"
run "compas_roc_eo_lam07_5000ep_seed1.json"

echo "ALL GPU0 JOBS COMPLETE" | tee -a "$LOG"
