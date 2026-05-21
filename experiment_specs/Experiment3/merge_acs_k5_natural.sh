#!/bin/bash
# Merge Oneida k5 natural-bias run after storm restart.
# seed_0 (complete) is in the original GPU0 dir (GG202605191004).
# seeds 1+42 will land in the new GPU0 restart dir (GG<new_timestamp>).
# This script copies seed_0 into the restart dir so check_run.py sees all 3 seeds.
# Run after acs_forge_gpu0_k5_restart completes.
set -euo pipefail
cd ~/cs_9170_project

OLD_DIR=$(ls -d training_runs/SPECacs_forge_gpu0_k5_EP5000_PCA10_REWwgl_minID0_majID1_TRJ2000_REAL3000_GG202605191004_* 2>/dev/null | head -1)
NEW_DIR=$(ls -dt training_runs/SPECacs_forge_gpu0_k5_restart_* 2>/dev/null | head -1)

if [ -z "$OLD_DIR" ] || [ -z "$NEW_DIR" ]; then
    echo "ERROR: could not find old ($OLD_DIR) or new ($NEW_DIR) run directory."
    exit 1
fi

echo "Old dir (seed_0 source): $OLD_DIR"
echo "New dir (restart target): $NEW_DIR"

if [ -d "$NEW_DIR/seed_0" ]; then
    echo "WARNING: seed_0 already exists in new dir — skipping copy."
else
    cp -r "$OLD_DIR/seed_0" "$NEW_DIR/seed_0"
    echo "Copied seed_0 into $NEW_DIR"
fi

echo "Merged. Run check_run.py on: $NEW_DIR"
echo "  python check_run.py $NEW_DIR --device cpu --no-gen-curve"
