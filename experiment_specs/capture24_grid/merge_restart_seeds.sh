#!/usr/bin/env bash
# Merge seed_42 from restart runs into the original run directories.
# Run this after both restart jobs finish.
set -euo pipefail

STORAGE=/storage_1/epigou_storage/FORGE/training_runs

# Original run directories (the ones missing seed_42)
GPU0_ORIG="$STORAGE/SPECcapture24_k0_gpu0_EP5000_PCA10_REWwgl_minID1_majID0_TRJ3000_REAL2000_GG202605061348_1bbc30ca"
GPU1_TRJ2_ORIG="$STORAGE/SPECcapture24_k0_gpu1_EP5000_PCA5_REWwgl_minID1_majID0_TRJ2000_REAL3000_GG202605061348_4da60a6f"
GPU1_TRJ3_ORIG="$STORAGE/SPECcapture24_k0_gpu1_EP5000_PCA5_REWwgl_minID1_majID0_TRJ3000_REAL2000_GG202605061348_ca1594c1"

echo "=== Finding restart run directories ==="

# Restart run directories (created 2026-05-11, PIDs 79533 / 79604)
GPU0_NEW="$STORAGE/SPECcapture24_k0_gpu0_restart_EP5000_PCA10_REWwgl_minID1_majID0_TRJ3000_REAL2000_GG202605111117_43a8f51c"
GPU1_TRJ2_NEW="$STORAGE/SPECcapture24_k0_gpu1_restart_EP5000_PCA5_REWwgl_minID1_majID0_TRJ2000_REAL3000_GG202605111117_93ba5383"
# gpu1 TRJ3000 directory will appear once that permutation starts; find it dynamically:
GPU1_TRJ3_NEW=$(ls -d "$STORAGE"/SPECcapture24_k0_gpu1_restart_EP5000_PCA5_REWwgl_minID1_majID0_TRJ3000_REAL2000_* 2>/dev/null | head -1)

echo "gpu0 restart dir: ${GPU0_NEW}"
echo "gpu1 TRJ2000 restart dir: ${GPU1_TRJ2_NEW}"
echo "gpu1 TRJ3000 restart dir: ${GPU1_TRJ3_NEW:-NOT FOUND YET}"

# gpu0: move seed_42 into original
if [[ -d "$GPU0_NEW/seed_42" ]]; then
    echo "Merging gpu0 seed_42..."
    cp -r "$GPU0_NEW/seed_42" "$GPU0_ORIG/seed_42"
    python3 -c "
import json
p = '$GPU0_ORIG/seeds.json'
d = json.load(open(p))
seeds = set(d.get('seeds', []))
seeds.add('42')
d['seeds'] = sorted(seeds)
json.dump(d, open(p,'w'), indent=2)
print('  seeds.json updated:', d['seeds'])
"
    echo "  Done: $GPU0_ORIG/seed_42"
else
    echo "WARNING: $GPU0_NEW/seed_42 not found"
fi

for NEW_DIR in "$GPU1_TRJ2_NEW" "$GPU1_TRJ3_NEW"; do
    [[ -z "$NEW_DIR" ]] && continue
    if [[ "$NEW_DIR" == *TRJ2000* ]]; then
        ORIG="$GPU1_TRJ2_ORIG"
    else
        ORIG="$GPU1_TRJ3_ORIG"
    fi
    if [[ -d "$NEW_DIR/seed_42" ]]; then
        echo "Merging $(basename $NEW_DIR) seed_42 -> $(basename $ORIG)..."
        cp -r "$NEW_DIR/seed_42" "$ORIG/seed_42"
        python3 -c "
import json
p = '$ORIG/seeds.json'
d = json.load(open(p))
seeds = set(d.get('seeds', []))
seeds.add('42')
d['seeds'] = sorted(seeds)
json.dump(d, open(p,'w'), indent=2)
print('  seeds.json updated:', d['seeds'])
"
        echo "  Done: $ORIG/seed_42"
    else
        echo "WARNING: $NEW_DIR/seed_42 not found"
    fi
done

echo ""
echo "=== Merge complete. Run check_run.py on the original directories to verify. ==="
