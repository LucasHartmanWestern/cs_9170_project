#!/bin/bash
# GPU 1: compas wgl_k{0,3,5,10}, compas roc_eo_lam{05,07}
set -e

PROJECT="/home/epigou/cs_9170_project"
SPECS="$PROJECT/experiment_specs/April_13_Experiments"
RUNS="$PROJECT/training_runs"
DEVICE="cuda:1"
LOG="$PROJECT/seed1_gpu1.log"

cd "$PROJECT"
source ~/envs/rl/bin/activate

run_and_consolidate() {
    local spec_file="$1"
    local target_dir="$2"

    echo "=== START: $spec_file -> $target_dir ===" | tee -a "$LOG"
    date | tee -a "$LOG"

    # Mark time before run
    touch /tmp/seed1_gpu1_marker_$$

    python main.py --spec "$SPECS/$spec_file" --device "$DEVICE" 2>&1 | tee -a "$LOG"

    # Find the new run directory (created after the marker, containing "seed1" in name)
    local new_dir
    new_dir=$(find "$RUNS" -maxdepth 1 -type d -newer /tmp/seed1_gpu1_marker_$$ -name "*seed1*" | head -1)

    if [ -z "$new_dir" ]; then
        echo "ERROR: Could not find new run directory for $spec_file" | tee -a "$LOG"
        rm -f /tmp/seed1_gpu1_marker_$$
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

    rm -f /tmp/seed1_gpu1_marker_$$
    echo "=== DONE: $target_dir/seed_1 consolidated ===" | tee -a "$LOG"
    date | tee -a "$LOG"
    echo "" | tee -a "$LOG"
}

run_and_consolidate \
    "compas_wgl_k0_5000ep_seed1.json" \
    "SPECcompas_wgl_k0_5000ep_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202604132237_294539a3"

run_and_consolidate \
    "compas_wgl_k3_5000ep_seed1.json" \
    "SPECcompas_wgl_k3_5000ep_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202604132237_be5e33ae"

run_and_consolidate \
    "compas_wgl_k5_5000ep_seed1.json" \
    "SPECcompas_wgl_k5_5000ep_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202604132237_5e3f63f1"

run_and_consolidate \
    "compas_wgl_k10_5000ep_seed1.json" \
    "SPECcompas_wgl_k10_5000ep_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202604132237_80bc39af"

run_and_consolidate \
    "compas_roc_eo_lam05_5000ep_seed1.json" \
    "SPECcompas_roc_eo_lam05_5000ep_EP5000_PCA10_REWroc_eo_minID0_majID1_TRJ2000_REAL3000_GG202604132237_a400fcf4"

run_and_consolidate \
    "compas_roc_eo_lam07_5000ep_seed1.json" \
    "SPECcompas_roc_eo_lam07_5000ep_EP5000_PCA10_REWroc_eo_minID0_majID1_TRJ2000_REAL3000_GG202604132237_73dab287"

echo "ALL GPU 1 JOBS COMPLETE" | tee -a "$LOG"
date | tee -a "$LOG"
