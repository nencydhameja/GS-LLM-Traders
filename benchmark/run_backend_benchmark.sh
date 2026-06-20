#!/usr/bin/env bash
# Benchmark Ollama vs llama-cpp-python on the local GB10 GPU.
# Run on apape2. Emails results and syncs back to apape1 + GitHub when done.
#
# Usage: bash benchmark/run_backend_benchmark.sh

set -e

REPO="$HOME/zit/GS-LLM-Traders"
LOG_DIR="$REPO/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/backend_benchmark_${TIMESTAMP}.log"
RESULTS_JSON="$LOG_DIR/backend_benchmark_${TIMESTAMP}.json"
VENV="$HOME/llms/venv"
LLAMA_PORT=8081
LLAMA_PID=""
GGUF="$HOME/llms/models/gemma-3-27b-it-Q4_K_M.gguf"
BENCH="$REPO/benchmark/benchmark_backends.py"
N_CALLS=10

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "========================================"
echo " Backend Benchmark  $(date)"
echo " Host: $(hostname)"
echo "========================================"

# ----------------------------------------------------------------
# 1. OLLAMA
# ----------------------------------------------------------------
echo ""
echo "=== [1/4] Starting Ollama ==="
if ! pgrep -f "ollama serve" > /dev/null 2>&1; then
    nohup ~/.local/bin/ollama serve > /tmp/ollama.log 2>&1 &
    echo "Ollama started, waiting for it to be ready..."
    for i in $(seq 1 24); do
        curl -sf http://localhost:11434/api/tags > /dev/null && break
        sleep 5
    done
else
    echo "Ollama already running."
fi

echo ""
echo "=== [2/4] Benchmarking Ollama (gemma3:27b, $N_CALLS calls) ==="
python3 "$BENCH" \
    --url http://localhost:11434/v1/chat/completions \
    --model gemma3:27b \
    --label ollama \
    --n-calls "$N_CALLS" \
    --out-json "$RESULTS_JSON" \
    --ollama-native-url http://localhost:11434/api/generate

# ----------------------------------------------------------------
# 2. LLAMA-CPP-PYTHON
# ----------------------------------------------------------------
echo ""
echo "=== [3/4] Setting up llama-cpp-python ==="

if [ ! -f "$VENV/bin/python" ]; then
    echo "Creating venv at $VENV ..."
    python3 -m venv "$VENV"
fi

if ! "$VENV/bin/python" -c "import llama_cpp" 2>/dev/null; then
    echo "Installing llama-cpp-python with CUDA (this takes ~10-15 min) ..."
    PATH=/usr/local/cuda/bin:$PATH \
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=native" \
    "$VENV/bin/pip" install "llama-cpp-python[server]" --upgrade
fi

# Verify CUDA linked
LIBLLAMA=$(find "$VENV" -name "libllama.so" | head -1)
if ldd "$LIBLLAMA" 2>/dev/null | grep -q "libggml-cuda"; then
    echo "CUDA confirmed in libllama.so."
else
    echo "WARNING: libllama.so does not link libggml-cuda — may be CPU-only build."
fi

# Write server config for gemma3:27b
LLAMA_YAML="$REPO/benchmark/llama_cpp_gemma3.yaml"
cat > "$LLAMA_YAML" << YAML
host: "0.0.0.0"
port: $LLAMA_PORT
models:
  - model: "$GGUF"
    model_alias: "local-gguf"
    n_gpu_layers: -1
    flash_attn: true
    n_ctx: 4096
    n_batch: 512
YAML

echo "Starting llama-cpp-python server on port $LLAMA_PORT ..."
"$VENV/bin/python" -m llama_cpp.server \
    --config_file "$LLAMA_YAML" > /tmp/llama_cpp_bench.log 2>&1 &
LLAMA_PID=$!

echo "Waiting for llama-cpp-python server (up to 5 min) ..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:$LLAMA_PORT/v1/models" > /dev/null; then
        echo "Server ready after $((i*10))s."
        break
    fi
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "ERROR: llama-cpp-python server died. Last log lines:"
        tail -20 /tmp/llama_cpp_bench.log
        LLAMA_PID=""
        break
    fi
    sleep 10
done

if [ -n "$LLAMA_PID" ] && kill -0 "$LLAMA_PID" 2>/dev/null; then
    echo ""
    echo "=== [4/4] Benchmarking llama-cpp-python (gemma3:27b, $N_CALLS calls) ==="
    python3 "$BENCH" \
        --url "http://localhost:$LLAMA_PORT/v1/chat/completions" \
        --model local-gguf \
        --label llama_cpp \
        --n-calls "$N_CALLS" \
        --out-json "$RESULTS_JSON" || true

    kill "$LLAMA_PID" 2>/dev/null && echo "llama-cpp-python server stopped."
else
    echo "Skipping llama-cpp-python benchmark (server failed to start)."
fi

# ----------------------------------------------------------------
# 3. SUMMARISE + EMAIL
# ----------------------------------------------------------------
echo ""
echo "=== Results ==="
SUMMARY=$(python3 - "$RESULTS_JSON" << 'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    results = json.load(f)
lines = []
for r in results:
    native = r.get("ollama_native") or {}
    prefill = f"  prefill {native['prefill_tok_s']} tok/s, gen {native['generation_tok_s']} tok/s" if native.get("prefill_tok_s") else ""
    lines.append(
        f"{r['label']:12s}  avg {r['mean_s']:.2f}s  "
        f"(min {r['min_s']:.2f} max {r['max_s']:.2f} sd {r['stdev_s']:.2f})  "
        f"prompt={r['prompt_tokens']}tok  completion={r['completion_tokens']}tok"
    )
    if prefill:
        lines.append(prefill)
print("\n".join(lines))
PYEOF
)
echo "$SUMMARY"

python3 ~/bin/send_notify.py \
    "Backend benchmark done: $(hostname) $(date '+%Y-%m-%d %H:%M')" \
    "$(printf 'Backend benchmark results:\n\n%s\n\nLog: %s\nResults: %s' \
        "$SUMMARY" "$LOG" "$RESULTS_JSON")"

# ----------------------------------------------------------------
# 4. SYNC
# ----------------------------------------------------------------
echo ""
echo "=== Syncing to apape1 and pushing to GitHub ==="
bash "$REPO/sync-this.sh"

echo ""
echo "=== Done $(date) ==="
