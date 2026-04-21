#!/bin/bash
set -e
PROJECT="/home/epigou/cs_9170_project"
SPECS="$PROJECT/experiment_specs/April_13_Experiments"
LOG="$PROJECT/missing_seed1_gpu1.log"

cd "$PROJECT"
source ~/envs/rl/bin/activate

run() {
    echo "=== START: $1 ===" | tee -a "$LOG"
    date | tee -a "$LOG"
    python main.py --spec "$SPECS/$1" --device cuda:1 2>&1 | tee -a "$LOG"
    echo "=== DONE: $1 ===" | tee -a "$LOG"
    date | tee -a "$LOG"
    echo "" | tee -a "$LOG"
}

run "compas_wgl_k5_5000ep_seed1.json"
run "compas_roc_eo_lam05_5000ep_seed1.json"

echo "ALL GPU1 JOBS COMPLETE" | tee -a "$LOG"
