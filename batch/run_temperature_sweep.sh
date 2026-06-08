#!/usr/bin/env bash
# Temperature sweep: 5 temps x 30 runs (baseline conditions otherwise).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MODEL="${1:-gemma3-27b}"
RUNS="${2:-30}"
SEED="${3:-42}"
TEMPS=(0.0 0.3 0.5 0.7 1.0)

for T in "${TEMPS[@]}"; do
    TS=$(date +%Y%m%d_%H%M%S)
    LOG="logs/temp_${T}_${MODEL}_${TS}.log"
    echo "=== TEMPERATURE: $T ($MODEL, $RUNS runs) -> $LOG ==="
    python3 attanasi_experiment.py \
        --model "$MODEL" \
        --runs "$RUNS" \
        --seed "$SEED" \
        --temperature "$T" \
        --resume 2>&1 | tee "$LOG"
done
