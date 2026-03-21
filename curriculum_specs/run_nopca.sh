#!/bin/bash
# Run all no-PCA smoke tests. Census runs on cuda:0, credit on cuda:1.
# Usage: bash curriculum_specs/run_nopca.sh
set -euo pipefail
source ~/envs/rl/bin/activate

# Census specs on cuda:0
echo "=== nopca_control ===" && python3 -u main.py --spec curriculum_specs/nopca_control.json --device cuda:0 \
    > curriculum_specs/logs/nopca_control.log 2>&1 &
echo "=== nopca_curr_standard ===" && python3 -u main.py --spec curriculum_specs/nopca_curr_standard.json --device cuda:0 \
    > curriculum_specs/logs/nopca_curr_standard.log 2>&1 &
echo "=== nopca_curr_gentle ===" && python3 -u main.py --spec curriculum_specs/nopca_curr_gentle.json --device cuda:0 \
    > curriculum_specs/logs/nopca_curr_gentle.log 2>&1 &
echo "=== nopca_large_net_control ===" && python3 -u main.py --spec curriculum_specs/nopca_large_net_control.json --device cuda:0 \
    > curriculum_specs/logs/nopca_large_net_control.log 2>&1 &
echo "=== nopca_large_net_curr ===" && python3 -u main.py --spec curriculum_specs/nopca_large_net_curr.json --device cuda:0 \
    > curriculum_specs/logs/nopca_large_net_curr.log 2>&1 &

# Credit spec on cuda:1
echo "=== nopca_credit_control ===" && python3 -u main.py --spec curriculum_specs/nopca_credit_control.json --device cuda:1 \
    > curriculum_specs/logs/nopca_credit_control.log 2>&1 &

wait
echo "All done"