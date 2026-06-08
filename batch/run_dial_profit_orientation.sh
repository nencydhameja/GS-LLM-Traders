#!/usr/bin/env bash
# Profit-orientation dial sweep: 5 levels x 30 runs.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MODEL="${1:-gemma3-27b}"
RUNS="${2:-30}"
SEED="${3:-42}"
LEVELS=(very_conservative conservative balanced maximizer extreme_maximizer)

for L in "${LEVELS[@]}"; do
    TS=$(date +%Y%m%d_%H%M%S)
    LOG="logs/dial_profit_${L}_${MODEL}_${TS}.log"
    echo "=== PROFIT_ORIENTATION: $L ($MODEL, $RUNS runs) -> $LOG ==="
    python3 attanasi_experiment.py \
        --model "$MODEL" \
        --runs "$RUNS" \
        --seed "$SEED" \
        --profit-orientation "$L" \
        --resume 2>&1 | tee "$LOG"
done
