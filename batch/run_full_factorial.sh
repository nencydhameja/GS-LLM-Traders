#!/usr/bin/env bash
# Full factorial: dial x temperature combos via run_factorial.py.
# Default = 125 combos (5 risk x 5 aggr x 5 profit x 5 temp -- handled internally).
# Use --subset to vary only one dimension (5 combos instead of 125).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MODEL="${1:-gemma3-27b}"
RUNS_PER_COMBO="${2:-5}"
SEED="${3:-42}"
SUBSET="${4:-}"   # optional: risk_aversion | aggressiveness | profit_orientation | temperature

TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/factorial_${MODEL}_${TS}.log"

CMD=(python3 run_factorial.py
    --model "$MODEL"
    --runs-per-combo "$RUNS_PER_COMBO"
    --seed "$SEED"
    --resume)
[[ -n "$SUBSET" ]] && CMD+=(--subset "$SUBSET")

echo "=== FACTORIAL: $MODEL, $RUNS_PER_COMBO runs/combo, subset=${SUBSET:-FULL} -> $LOG ==="
"${CMD[@]}" 2>&1 | tee "$LOG"
