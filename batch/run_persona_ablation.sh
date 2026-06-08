#!/usr/bin/env bash
# Persona sweep: undergrad_econ / grad_econ / undergrad_human, 30 runs each.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MODEL="${1:-gemma3-27b}"
RUNS="${2:-30}"
SEED="${3:-42}"
PERSONAS=(undergrad_econ grad_econ undergrad_human)

for P in "${PERSONAS[@]}"; do
    TS=$(date +%Y%m%d_%H%M%S)
    LOG="logs/persona_${P}_${MODEL}_${TS}.log"
    echo "=== PERSONA: $P ($MODEL, $RUNS runs) -> $LOG ==="
    python3 attanasi_experiment.py \
        --model "$MODEL" \
        --runs "$RUNS" \
        --seed "$SEED" \
        --persona "$P" \
        --resume 2>&1 | tee "$LOG"
done
