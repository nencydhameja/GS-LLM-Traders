#!/usr/bin/env bash
# Phase 0 across all available models, sequential.
# Adjust MODELS array to match what your server supports.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

RUNS="${1:-30}"
SEED="${2:-42}"
MODELS=(
    gemma3-27b
    llama3.3
    llama3.1-70B
    mixtral-8x22b
    mistral-instruct
    phi4
    qwen2.5-coder-32B
    gpt-oss-20b
    hermes3
    gemma3-4b
)

for M in "${MODELS[@]}"; do
    TS=$(date +%Y%m%d_%H%M%S)
    LOG="logs/human_baseline_${M}_${TS}.log"
    echo "=== MODEL: $M ($RUNS runs, seed=$SEED) -> $LOG ==="
    python3 attanasi_experiment.py \
        --model "$M" \
        --runs "$RUNS" \
        --seed "$SEED" \
        --resume 2>&1 | tee "$LOG" || echo "WARN: $M failed, continuing"
done
