#!/bin/bash
# Run all curriculum smoke tests sequentially (local).
# Usage: bash curriculum_specs/run_all.sh
set -euo pipefail

echo "=== curr_control ==="
python3 -u main.py --spec curriculum_specs/curr_control.json --device cuda:0

echo "=== curr_gentle ==="
python3 -u main.py --spec curriculum_specs/curr_gentle.json --device cuda:0

echo "=== curr_standard ==="
python3 -u main.py --spec curriculum_specs/curr_standard.json --device cuda:0

echo "=== curr_aggressive ==="
python3 -u main.py --spec curriculum_specs/curr_aggressive.json --device cuda:0
