#!/usr/bin/env bash
# Exact human replication: no dial, no persona, no temp override, baseline memory.
# This is what Attanasi et al. (2021) humans saw — closest LLM analog.
# Args: [MODEL] [RUNS] [SEED]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MODEL="${1:-gemma3-27b}"
RUNS="${2:-30}"
SEED="${3:-42}"
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/human_baseline_${MODEL}_${TS}.log"

echo "=== HUMAN REPLICATION: $MODEL, $RUNS runs, seed=$SEED ==="
echo "Logging to: $LOG"

python3 attanasi_experiment.py \
    --model "$MODEL" \
    --runs "$RUNS" \
    --seed "$SEED" \
    --resume 2>&1 | tee "$LOG"
