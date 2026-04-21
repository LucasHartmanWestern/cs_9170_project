#!/bin/bash
# GPU 0: census_wgl_k3, census_roc_eo_lam07, compas_roc_eo_lam03, compas_roc_eo_lam05
set -e

PROJECT="/home/epigou/cs_9170_project"
SPECS="$PROJECT/experiment_specs/April_13_Experiments"
RUNS="$PROJECT/training_runs"
DEVICE="cuda:0"
LOG="$PROJECT/seed1_retry_gpu0.log"

cd "$PROJECT"
source ~/envs/rl/bin/activate

run_and_consolidate() {
    local spec_file="$1"
    local target_dir="$2"

    echo "=== START: $spec_file -> $target_dir ===" | tee -a "$LOG"
    date | tee -a "$LOG"

    local tmp_out
    tmp_out=$(mktemp)

    python main.py --spec "$SPECS/$spec_file" --device "$DEVICE" 2>&1 | tee -a "$LOG" | tee "$tmp_out"

    local new_group
    new_group=$(grep "\[main\] exp_group=" "$tmp_out" | head -1 | sed 's/.*exp_group=//')
    rm -f "$tmp_out"

    if [ -z "$new_group" ]; then
        echo "ERROR: Could not find exp_group for $spec_file" | tee -a "$LOG"
        exit 1
    fi

    local new_dir="$RUNS/$new_group"
    echo "New run dir: $new_dir" | tee -a "$LOG"

    if [ ! -d "$new_dir/seed_1" ]; then
        echo "ERROR: seed_1 not found in $new_dir" | tee -a "$LOG"
        exit 1
    fi

    if [ -d "$RUNS/$target_dir/seed_1" ]; then
        echo "Removing existing seed_1 from target..." | tee -a "$LOG"
        rm -rf "$RUNS/$target_dir/seed_1"
    fi

    mv "$new_dir/seed_1" "$RUNS/$target_dir/"
    rm -rf "$new_dir"

    # Append seed_1 test metrics to experiment-level final_test_metrics.csv
    local seed1_test="$RUNS/$target_dir/seed_1/final_test_metrics.csv"
    local exp_test="$RUNS/$target_dir/final_test_metrics.csv"
    if [ -f "$seed1_test" ]; then
        tail -n +2 "$seed1_test" >> "$exp_test"
        echo "Appended seed_1 test metrics to experiment-level CSV." | tee -a "$LOG"
    else
        echo "WARNING: seed_1/final_test_metrics.csv not found." | tee -a "$LOG"
    fi

    echo "=== DONE: $target_dir/seed_1 consolidated ===" | tee -a "$LOG"
    date | tee -a "$LOG"
    echo "" | tee -a "$LOG"
}

run_and_consolidate \
    "census_wgl_k3_5000ep_seed1.json" \
    "SPECcensus_wgl_k3_5000ep_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202604132300_1ba3fd5c"

run_and_consolidate \
    "census_roc_eo_lam07_5000ep_seed1.json" \
    "SPECcensus_roc_eo_lam07_5000ep_EP5000_PCA10_REWroc_eo_minID0_majID1_TRJ2000_REAL3000_GG202604132330_6680a7d8"

run_and_consolidate \
    "compas_roc_eo_lam03_5000ep_seed1.json" \
    "SPECcompas_roc_eo_lam03_5000ep_EP5000_PCA10_REWroc_eo_minID0_majID1_TRJ2000_REAL3000_GG202604132237_8afd81e8"

run_and_consolidate \
    "compas_roc_eo_lam05_5000ep_seed1.json" \
    "SPECcompas_roc_eo_lam05_5000ep_EP5000_PCA10_REWroc_eo_minID0_majID1_TRJ2000_REAL3000_GG202604132237_a400fcf4"

echo "ALL GPU 0 RETRY JOBS COMPLETE" | tee -a "$LOG"
date | tee -a "$LOG"
