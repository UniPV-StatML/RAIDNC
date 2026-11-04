#!/bin/bash
# Train both RAIDNC v2 presets. Usage: bash train_all.sh [python]
set -e
cd "$(dirname "$0")"
PY="${1:-python3}"
mkdir -p ../results
for p in compact tuned; do
  "$PY" train_raidnc_v2.py --preset "$p" --out "../model/$p" > "../results/train_$p.log" 2>&1
done
