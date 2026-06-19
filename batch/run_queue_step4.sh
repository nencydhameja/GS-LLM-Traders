#!/usr/bin/env bash
# Queue runner for steps 4-9 (picks up after persona ablation).
# Usage: bash batch/run_queue_step4.sh [MODEL] [RUNS] [SEED]
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-gemma3-27b}"
RUNS="${2:-30}"
SEED="${3:-42}"

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
    git add output/ logs/ 2>/dev/null || true
    if ! git diff --cached --quiet; then
        git commit -m "Add output: ${label} on $(date +%Y-%m-%d)"
        git push
        echo "Committed and pushed: ${label}"
    else
        echo "Nothing new to commit for: ${label}"
    fi
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

Machine:   promaxgb10-4ae4
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

run_script "run_dial_risk_aversion.sh"      "Dial: Risk Aversion"
run_script "run_dial_aggressiveness.sh"     "Dial: Aggressiveness"
run_script "run_dial_profit_orientation.sh" "Dial: Profit Orientation"
run_script "run_temperature_sweep.sh"       "Temperature Sweep"
run_script "run_full_factorial.sh"          "Full Factorial"
run_script "run_human_baseline_all_models.sh" "Human Baseline (All Models)"

send_email \
    "GS-LLM-Traders: ALL RUNS COMPLETE on promaxgb10" \
    "This is an automatically generated email sent by Claude AI (claude-sonnet-4-6) on behalf of apape.

=== ALL GS-LLM-Traders batch runs COMPLETE ===

Machine:   promaxgb10-4ae4
Finished:  $(date)

All scripts completed and output pushed to:
  https://github.com/nencydhameja/GS-LLM-Traders

Scripts completed (in order):
  4. run_dial_risk_aversion.sh           (~45 h)
  5. run_dial_aggressiveness.sh          (~45 h)
  6. run_dial_profit_orientation.sh      (~45 h)
  7. run_temperature_sweep.sh            (~45 h)
  8. run_full_factorial.sh               (~190 h)
  9. run_human_baseline_all_models.sh    (~90 h)

---
This email was generated automatically by Claude AI running on promaxgb10-4ae4."

echo "=== QUEUE: All done ==="
