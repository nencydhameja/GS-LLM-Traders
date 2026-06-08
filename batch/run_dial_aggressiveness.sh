#!/usr/bin/env bash
# Aggressiveness dial sweep: 5 levels x 30 runs.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MODEL="${1:-gemma3-27b}"
RUNS="${2:-30}"
SEED="${3:-42}"
LEVELS=(very_passive passive moderate aggressive very_aggressive)

for L in "${LEVELS[@]}"; do
    TS=$(date +%Y%m%d_%H%M%S)
    LOG="logs/dial_aggr_${L}_${MODEL}_${TS}.log"
    echo "=== AGGRESSIVENESS: $L ($MODEL, $RUNS runs) -> $LOG ==="
    python3 attanasi_experiment.py \
        --model "$MODEL" \
        --runs "$RUNS" \
        --seed "$SEED" \
        --aggressiveness "$L" \
        --resume 2>&1 | tee "$LOG"
done
