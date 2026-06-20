#!/usr/bin/env bash
# Sequential queue runner for all GS-LLM-Traders batch scripts.
# Waits for a given PID to finish first (the already-running baseline).
# After each script: commits + pushes output, sends completion email.
# Usage: bash run_queue.sh [WAIT_PID] [MODEL] [RUNS] [SEED]
set -euo pipefail
cd "$(dirname "$0")/.."

WAIT_PID="${1:-}"
MODEL="${2:-gemma3-27b}"
RUNS="${3:-30}"
SEED="${4:-42}"

GMAIL_USER="dr.duus@gmail.com"
GMAIL_APP_PW="fdyuuyxvgzpvgnbu"
RECIPIENTS="dr.duus@gmail.com ndhamej1@binghamton.edu"

send_email() {
    local subject="$1"
    local body="$2"
    python3 - "$subject" "$body" <<'PYEOF'
import smtplib, sys
from email.message import EmailMessage
GMAIL_USER = "dr.duus@gmail.com"
GMAIL_APP_PW = "fdyuuyxvgzpvgnbu"
RECIPIENTS = ["dr.duus@gmail.com", "ndhamej1@binghamton.edu"]
msg = EmailMessage()
msg["From"] = GMAIL_USER
msg["To"] = ", ".join(RECIPIENTS)
msg["Subject"] = sys.argv[1]
msg.set_content(sys.argv[2])
with smtplib.SMTP("smtp.gmail.com", 587) as s:
    s.starttls()
    s.login(GMAIL_USER, GMAIL_APP_PW)
    s.send_message(msg)
print("Email sent.")
PYEOF
}

commit_and_push() {
    local label="$1"
    echo "Syncing to apape1 and pushing to GitHub: ${label}"
    bash "$(dirname "$0")/../sync-this.sh"
}

run_script() {
    local script="$1"
    local label="$2"
    local logfile

    echo "=== QUEUE: Starting ${label} ==="
    bash "batch/${script}" "$MODEL" "$RUNS" "$SEED"
    logfile=$(ls -t logs/ | head -1)

    commit_and_push "$label"

    local results=""
    results=$(ls -t output/results_*.txt output/attanasi_bootstrap_*.csv 2>/dev/null | head -5 | xargs -I{} sh -c 'echo "=== {} ==="; cat {}' 2>/dev/null || echo "(no results file found)")

    send_email \
        "GS-LLM-Traders: ${label} COMPLETE on promaxgb10" \
        "This is an automatically generated email sent by Claude AI (claude-sonnet-4-6) on behalf of apape.

=== ${label} — COMPLETE ===

Machine:   $(hostname)
Model:     ${MODEL}
Runs:      ${RUNS} (seed ${SEED})
Finished:  $(date)
Log:       /home/apape/zit/GS-LLM-Traders/logs/${logfile}

Output has been committed and pushed to:
  https://github.com/nencydhameja/GS-LLM-Traders

--- Recent results ---
${results}

---
This email was generated automatically by Claude AI running on promaxgb10-4ae4."

    echo "=== QUEUE: ${label} done, email sent ==="
}

# Wait for the already-running baseline if a PID was given
if [[ -n "$WAIT_PID" ]] && kill -0 "$WAIT_PID" 2>/dev/null; then
    echo "=== QUEUE: Waiting for PID ${WAIT_PID} (human baseline) to finish ==="
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
    echo "=== QUEUE: PID ${WAIT_PID} finished, committing baseline output ==="
    commit_and_push "human_baseline"
    # Send baseline completion email
    logfile=$(ls -t logs/human_baseline_*.log 2>/dev/null | head -1 || echo "unknown")
    results=$(ls -t output/results_*.txt 2>/dev/null | head -3 | xargs -I{} sh -c 'echo "=== {} ==="; cat {}' 2>/dev/null || echo "(no results file found)")
    send_email \
        "GS-LLM-Traders: Human Baseline COMPLETE on promaxgb10" \
        "This is an automatically generated email sent by Claude AI (claude-sonnet-4-6) on behalf of apape.

=== Human Baseline (run_human_baseline.sh) — COMPLETE ===

Machine:   $(hostname)
Model:     ${MODEL}
Runs:      ${RUNS} (seed ${SEED})
Finished:  $(date)
Log:       /home/apape/zit/GS-LLM-Traders/logs/${logfile}

Output committed and pushed to:
  https://github.com/nencydhameja/GS-LLM-Traders

--- Results ---
${results}

Starting next script: run_memory_ablation.sh (~18 h)

---
This email was generated automatically by Claude AI running on promaxgb10-4ae4."
fi

# Start llama-cpp-python server if not already running
LLAMA_SERVER_PID=""
if ! curl -sf http://localhost:8081/v1/models > /dev/null 2>&1; then
    echo "=== Starting llama-cpp-python server ==="
    ~/llms/venv/bin/python -m llama_cpp.server \
        --config_file ~/zit/GS-LLM-Traders/benchmark/llama_cpp_gemma3.yaml \
        > /tmp/llama_cpp_server.log 2>&1 &
    LLAMA_SERVER_PID=$!
    echo "Waiting for server (up to 5 min, PID ${LLAMA_SERVER_PID})..."
    for i in $(seq 1 30); do
        curl -sf http://localhost:8081/v1/models > /dev/null && echo "Server ready after $((i*10))s." && break
        if ! kill -0 "$LLAMA_SERVER_PID" 2>/dev/null; then
            echo "ERROR: llama-cpp-python server died. Check /tmp/llama_cpp_server.log" >&2
            exit 1
        fi
        sleep 10
    done
else
    echo "=== llama-cpp-python server already running on port 8081 ==="
fi

# Steps 1-3 (human baseline, memory ablation, persona ablation) already complete — skip.
# Run remaining scripts in order (full factorial last — it's the lion's share)
run_script "run_dial_risk_aversion.sh"        "Dial: Risk Aversion"
run_script "run_dial_aggressiveness.sh"       "Dial: Aggressiveness"
run_script "run_dial_profit_orientation.sh"   "Dial: Profit Orientation"
run_script "run_temperature_sweep.sh"         "Temperature Sweep"
run_script "run_human_baseline_all_models.sh" "Human Baseline (All Models)"
run_script "run_full_factorial.sh"            "Full Factorial"

# Shut down llama-cpp-python server if we started it
if [[ -n "$LLAMA_SERVER_PID" ]] && kill -0 "$LLAMA_SERVER_PID" 2>/dev/null; then
    kill "$LLAMA_SERVER_PID" && echo "llama-cpp-python server stopped."
fi

send_email \
    "GS-LLM-Traders: ALL RUNS COMPLETE on promaxgb10" \
    "This is an automatically generated email sent by Claude AI (claude-sonnet-4-6) on behalf of apape.

=== ALL GS-LLM-Traders batch runs COMPLETE ===

Machine:   $(hostname)
Finished:  $(date)

All scripts completed and output pushed to:
  https://github.com/nencydhameja/GS-LLM-Traders

Scripts completed (in order):
  1. run_human_baseline.sh           (~9 h)
  2. run_memory_ablation.sh          (~18 h)
  3. run_persona_ablation.sh         (~27 h)
  4. run_dial_risk_aversion.sh       (~45 h)
  5. run_dial_aggressiveness.sh      (~45 h)
  6. run_dial_profit_orientation.sh  (~45 h)
  7. run_temperature_sweep.sh        (~45 h)
  8. run_full_factorial.sh           (~190 h)
  9. run_human_baseline_all_models.sh (~90 h)

---
This email was generated automatically by Claude AI running on promaxgb10-4ae4."

echo "=== QUEUE: All done ==="
