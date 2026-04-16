#!/bin/bash
# GPU 0: census wgl_k{0,3,5,10}, census roc_eo_lam{05,07}, compas roc_eo_lam03
set -e

PROJECT="/home/epigou/cs_9170_project"
SPECS="$PROJECT/experiment_specs/April_13_Experiments"
RUNS="$PROJECT/training_runs"
DEVICE="cuda:0"
LOG="$PROJECT/seed1_gpu0.log"

cd "$PROJECT"
source ~/envs/rl/bin/activate

run_and_consolidate() {
    local spec_file="$1"
    local target_dir="$2"

    echo "=== START: $spec_file -> $target_dir ===" | tee -a "$LOG"
    date | tee -a "$LOG"

    # Mark time before run
    touch /tmp/seed1_gpu0_marker_$$

    python main.py --spec "$SPECS/$spec_file" --device "$DEVICE" 2>&1 | tee -a "$LOG"

    # Find the new run directory (created after the marker, containing "seed1" in name)
    local new_dir
    new_dir=$(find "$RUNS" -maxdepth 1 -type d -newer /tmp/seed1_gpu0_marker_$$ -name "*seed1*" | head -1)

    if [ -z "$new_dir" ]; then
        echo "ERROR: Could not find new run directory for $spec_file" | tee -a "$LOG"
        rm -f /tmp/seed1_gpu0_marker_$$
        exit 1
    fi

    echo "New run dir: $new_dir" | tee -a "$LOG"

    # Remove incomplete seed_1 from target (if exists)
    if [ -d "$RUNS/$target_dir/seed_1" ]; then
        echo "Removing incomplete seed_1 from target..." | tee -a "$LOG"
        rm -rf "$RUNS/$target_dir/seed_1"
    fi

    # Move seed_1 into target
    mv "$new_dir/seed_1" "$RUNS/$target_dir/"

    # Remove the now-empty new run dir
    rm -rf "$new_dir"

    rm -f /tmp/seed1_gpu0_marker_$$
    echo "=== DONE: $target_dir/seed_1 consolidated ===" | tee -a "$LOG"
    date | tee -a "$LOG"
    echo "" | tee -a "$LOG"
}

run_and_consolidate \
    "census_wgl_k0_5000ep_seed1.json" \
    "SPECcensus_wgl_k0_5000ep_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202604132257_4e035c66"

run_and_consolidate \
    "census_wgl_k3_5000ep_seed1.json" \
    "SPECcensus_wgl_k3_5000ep_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202604132300_1ba3fd5c"

run_and_consolidate \
    "census_wgl_k5_5000ep_seed1.json" \
    "SPECcensus_wgl_k5_5000ep_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202604132300_d96ace32"

run_and_consolidate \
    "census_wgl_k10_5000ep_seed1.json" \
    "SPECcensus_wgl_k10_5000ep_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202604132300_97dd3630"

run_and_consolidate \
    "census_roc_eo_lam05_5000ep_seed1.json" \
    "SPECcensus_roc_eo_lam05_5000ep_EP5000_PCA10_REWroc_eo_minID0_majID1_TRJ2000_REAL3000_GG202604132327_6f8ed797"

run_and_consolidate \
    "census_roc_eo_lam07_5000ep_seed1.json" \
    "SPECcensus_roc_eo_lam07_5000ep_EP5000_PCA10_REWroc_eo_minID0_majID1_TRJ2000_REAL3000_GG202604132330_6680a7d8"

run_and_consolidate \
    "compas_roc_eo_lam03_5000ep_seed1.json" \
    "SPECcompas_roc_eo_lam03_5000ep_EP5000_PCA10_REWroc_eo_minID0_majID1_TRJ2000_REAL3000_GG202604132237_8afd81e8"

echo "ALL GPU 0 JOBS COMPLETE" | tee -a "$LOG"
date | tee -a "$LOG"
