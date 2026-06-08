#!/usr/bin/env bash
# Memory ablation: baseline vs refined memory, same model.
# Tests whether richer cross-period summaries close the LLM learning gap.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MODEL="${1:-gemma3-27b}"
RUNS="${2:-30}"
SEED="${3:-42}"

for MEM in baseline refined; do
    FLAGS=()
    [[ "$MEM" == "refined" ]] && FLAGS+=(--refined-memory)
    TS=$(date +%Y%m%d_%H%M%S)
    LOG="logs/memory_${MEM}_${MODEL}_${TS}.log"
    echo "=== MEMORY: $MEM ($MODEL, $RUNS runs) -> $LOG ==="
    python3 attanasi_experiment.py \
        --model "$MODEL" \
        --runs "$RUNS" \
        --seed "$SEED" \
        "${FLAGS[@]}" \
        --resume 2>&1 | tee "$LOG"
done
