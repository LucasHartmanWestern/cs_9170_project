#!/bin/bash
# EXP-051 grid-search-gaps: submit all missing capture-24 kfold jobs to DRAC
# Run from the project root directory on DRAC.
#
# New grouped specs (15 jobs): each runs ep∈{10,20,30} (or ep∈{10,20}) × folds 0-2
# in sequential batches of 3 parallel folds. max_parallel=3, 20h walltime.
for f in experiment_specs/capture24_kfold_grid/drac/forge_gap_*.sh; do
    echo "Submitting $f"
    sbatch "$f"
done

# k=5/pca=15/ratio=0.2/ep=20: only fold 2 is missing; submit individual spec
echo "Submitting forge_k5_ep20_fold2.sh"
sbatch experiment_specs/capture24_kfold_grid/drac/forge_k5_ep20_fold2.sh
