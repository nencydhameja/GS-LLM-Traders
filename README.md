# GS-LLM-Traders

LLM replication of **Attanasi, Centorrino & Moscati (2021)** "Controlling monopoly power in a double-auction market experiment" (*JPET* 23(5), 1074-1101).

We replace human subjects with LLM agents and compare trading behavior across 5 market structures.

## Status

| Phase | Market Structure | Status |
|-------|-----------------|--------|
| 1 | Perfect Competition (4 sellers, 24 buyers) | **Complete** (30 runs, mistral-instruct) |
| 2 | Competition with quotas (capacity = 6/seller) | Not started |
| 3 | Cartel (sellers agree on price) | Not started |
| 4 | Cartel with quotas | Not started |
| 5 | Monopoly (1 seller, 6 buyers) | Not started |

### Ablation / Sweep Progress (gemma3-27b, promaxgb10)

| Step | Script | Status | Notes |
|------|--------|--------|-------|
| 1 | `run_human_baseline.sh` | **Complete** | Committed 2026-06-08 |
| 2 | `run_memory_ablation.sh` | **Complete** | Committed 2026-06-09 |
| 3 | `run_persona_ablation.sh` | **Complete** | Committed 2026-06-11 |
| 4 | `run_dial_risk_aversion.sh` | **Pending restart** | 11/30 done for `very_averse`; Binghamton server outage killed it. **Run fresh on apape2 with `--resume`** |
| 5 | `run_dial_aggressiveness.sh` | Pending | Prior run data exists (Apr–May); re-running fresh |
| 6 | `run_dial_profit_orientation.sh` | Pending | Prior run data exists (Apr–May); re-running fresh |
| 7 | `run_temperature_sweep.sh` | Pending | Prior run data exists (Apr–May); re-running fresh |
| 8 | `run_full_factorial.sh` | Pending | Queued |
| 9 | `run_human_baseline_all_models.sh` | Pending | Queued |

**Active machine: apape2** (`promaxgb10-4be9`). apape1's GPU is fully occupied by a PancsVriend `run_all_contexts.py` job started 2026-06-11 (100 runs × 1000 steps on gemma-4-31b). Do not interrupt it.

**LLM backend (2026-06-19 onwards):** Local Ollama on apape2 — no longer depends on `chat.binghamton.edu`. `attanasi_experiment.py` auto-detects hostname and routes to `http://localhost:11434/v1/chat/completions`. Both machines run Ollama **v0.30.10** (upgraded 2026-06-19; v0.23.2 did not detect the GB10 GPU and ran on CPU at 10× slower speeds).

**Previous crash (2026-06-11):** Queue died at start of step 4 because stale progress `.json` files committed from Nancy Dhameja's MacBook had `master_path` pointing to `/Users/nencydhameja/...`. Fixed 2026-06-19 by deleting all stale progress files and restarting from step 4.

**Prior campaign (Apr–May 2026):** Steps 4–7 were run in an earlier campaign before the June restart. Output CSVs survive in `output/` under their original timestamps. Coverage was incomplete — `very_seeking` is missing from the risk-aversion sweep and `very_passive` is missing from aggressiveness — so the June re-run is authoritative. The old files are retained for reference but should not be treated as the final dataset.

## Pending Tasks

### Task 1 — Switch to local inference on apape1/apape2 ✓ DONE (2026-06-19)

`attanasi_experiment.py` now auto-detects hostname at startup. On `apape1.rc.binghamton.edu` / `apape2.rc.binghamton.edu` (matching `promaxgb10-4ae4` / `promaxgb10-4be9`), all LLM calls are routed to `http://localhost:11434/v1/chat/completions` (local Ollama). All other hosts continue to use `chat.binghamton.edu`. No flag required.

### Task 2 — Download LLMs for local inference ✓ DONE (2026-06-19)

`gemma3:27b` (17 GB, Q4_K_M, ID `a418f5838eaf`) is now pulled and ready on both machines.

- **apape1** (`promaxgb10-4ae4`) — Ollama **v0.30.10** system service at `/usr/local/bin/ollama`; model at `/usr/share/ollama/.ollama/models`
- **apape2** (`promaxgb10-4be9`) — Ollama **v0.30.10** user binary at `~/.local/bin/ollama`; model at `~/.ollama/models`; start server with `nohup ~/.local/bin/ollama serve > /tmp/ollama.log 2>&1 &`

Both now correctly use the GB10 GPU via the `cuda_v13` backend (compute 12.1). v0.23.2 only had `cuda_v12` which does not support cc 12.1 and silently fell back to CPU.

### Task 3 — Recalculate expected runtime for local inference ⏳ IN PROGRESS (2026-06-19)

**CPU benchmark (2026-06-19, apape1, Ollama v0.23.2 — GPU not detected, ran on CPU):**
1 run, 30 steps, seed 0 → **181 min 41 sec**. This is the CPU baseline, not the target.

**GPU benchmark (2026-06-19, apape2, Ollama v0.30.10 — GB10 GPU active):**
Running now. Log: `apape2:~/zit/GS-LLM-Traders/logs/benchmark_apape2_gpu_20260619_211930.log`
Email will be sent to dr.duus@gmail.com and ndhamej1@binghamton.edu on completion with full results.
Expected completion: **~10–30 minutes** from benchmark start (GPU is ~10–30× faster than CPU).

| Baseline | Min/run | Notes |
|----------|---------|-------|
| Binghamton server (prior estimate) | ~18 min | Historical, remote inference |
| apape1 CPU (Ollama v0.23.2, GPU not detected) | ~182 min | **Not representative** |
| apape2 GPU (Ollama v0.30.10, GB10 cuda_v13) | **TBD** | Benchmark running — update when email arrives |

**Action when email arrives:** pull the `real` time from the log, update the table above and the Execution Plan estimates below.

---

## Experimental Design

Replicates the paper's Phase 1 (Perfect Competition):

- **Agents**: 4 sellers (cost = 12) + 24 buyers (valuations: 20, 17, 15, 13, 8, 0 x 4 groups)
- **Mechanism**: Continuous double auction with price-time priority
- **Periods**: 3 trading periods per run, 30 trading rounds per period
- **Runs**: 30 independent runs (seeds 42-71) for bootstrap 95% CIs
- **Prompts**: Mirror exactly what human subjects see on screen (Appendix Figs A.4/A.5) including market structure, transaction history, own situation, cumulated payoffs, and trading round countdown

### Human Benchmarks (Table B.1, Competition played first)

| Period | Human Avg Price | CE Price | CE Quantity |
|--------|----------------|----------|-------------|
| 1 | 14.49 | ~12 | 16 |
| 2 | 14.10 | ~12 | 16 |
| 3 | 13.50 | ~12 | 16 |

## Files

| File | What It Covers |
|------|---------------|
| `attanasi_config.yaml` | **All experimental parameters + LLM settings.** Market structure (4 sellers, cost=12; 24 buyers, valuations 20/17/15/13/8/0 x 4 groups; 3 periods, 30 steps). CE benchmarks (price ~12, quantity 16). Human benchmarks from Table B.1 (avg prices 14.49, 14.10, 13.50). 12 LLM model configs with endpoints, timeouts, temperature. Prompt templates mirroring exactly what human subjects see on screen (Appendix Figs A.4/A.5): market screen, order book state, transaction history, own situation, cumulated payoffs, trading round countdown. |
| `attanasi_experiment.py` | **The entire simulation engine.** (1) *Agent class* -- LLM-powered agents that receive prompts mimicking the human interface, query the Binghamton server, parse an integer price, and submit to the auction. Constraint validation, retry logic, error tracking. (2) *DoubleAuction class* -- continuous double auction with price-time priority matching, separate bid/ask books, trade execution. (3) *CompetitionExperiment class* -- orchestrates 28 agents across 3 periods of 30 steps, shuffles agent order each step, records every bid/ask decision (whether it matched or not). (4) *Batch runner* -- N replications with different seeds, one master CSV, `--resume` support, bootstrap 95% CIs. |
| `run_gui.py` | **Interactive console menu.** Shows numbered list of available models, prompts for runs/steps/seed/resume with defaults. Validates inputs, shows confirmation, then launches the experiment. Just run `python run_gui.py`. |
| `paper.tex` | **Limitations writeup.** Temporal reasoning: LLMs don't feel time pressure like human subjects with a countdown clock. We include "Trading round X of 30" in prompts as approximation. |
| `jpet12504-sup-0001-online_appendix.pdf` | **Original paper appendix** with human subject instructions and benchmark tables. |

## Output

All output goes to `output/`:

| File | Description |
|------|-------------|
| `attanasi_master_{model}_{timestamp}.csv` | **Every decision from every run in one file.** Columns: `run_id`, `seed`, `model`, `period`, `step`, `agent_id`, `agent_type`, `action` (bid/ask), `submitted_price`, `reservation_value`, `matched` (True/False), `trade_price`, `counterparty`, `profit`. |
| `attanasi_bootstrap_{model}.csv` | **Bootstrap 95% CI results.** Columns: `llm_mean_price`, `llm_sd_price`, `llm_ci_lo`, `llm_ci_hi`, `human_mean_price`, `llm_mean_trades`, `human_ce_trades`. |
| `results_{model}.txt` | **Human-readable results summary.** |
| `attanasi_progress_{model}.json` | Resume checkpoint (auto-deleted when batch completes). |

From the master CSV you can compute: avg/median prices, price std dev, num trades, quantity efficiency (trades/16), total surplus, surplus efficiency (actual/CE max of 68), per-buyer-type trading, price convergence, and bid/ask distributions.

## Setup

### Requirements
```bash
pip install requests pyyaml
```

### Configuration
All experimental parameters and LLM model configs are in `attanasi_config.yaml`. The API endpoint is the Binghamton University OpenWebUI server.

### Running

```bash
# Interactive menu (recommended)
python run_gui.py

# Or direct CLI
python attanasi_experiment.py --model mistral-instruct --steps 30 --runs 30 --seed 42

# Resume after server interruption
python attanasi_experiment.py --model mistral-instruct --steps 30 --runs 30 --seed 42 --resume
```

### Available Models
All hosted on Binghamton server: `mistral-instruct`, `llama3.1-70B`, `llama3.3`, `mixtral-8x22b`, `gemma3-27b`, `gemma3-4b`, `phi4`, `qwen2.5-coder-32B`, `qwq`, `gpt-oss-20b`, `hermes3`, `codellama-70B`

## Results (Phase 1 -- Mistral-Instruct, 30 runs)

```
Period  LLM Price [95% CI]         Human Price  LLM Trades  CE Trades
---------------------------------------------------------------------
  1     16.54  [16.33, 16.77]        14.49         7.0         16
  2     16.92  [16.70, 17.13]        14.10         6.2         16
  3     17.26  [16.96, 17.56]        13.50         5.7         16
```

- LLM prices are **2-4 points above** human averages, and the CIs don't overlap with human means
- **No convergence** -- LLM prices *increase* across periods (16.5->17.3) while humans *decrease* (14.5->13.5)
- **Severe under-trading** -- ~6 trades vs 16 at CE (38% quantity efficiency)
- Sellers exhibit **tacit collusion** -- consistently asking 17-19, never undercutting each other

---

## Execution Plan (apape @ promaxgb10)

### Machines

| Resource | apape1 | apape2 |
|----------|--------|--------|
| Hostname | `promaxgb10-4ae4` | `promaxgb10-4be9` |
| Ollama | v0.30.10 system service | v0.30.10 user binary (`~/.local/bin/ollama`) |
| GPU | NVIDIA GB10 (cuda_v13, cc 12.1) | NVIDIA GB10 (cuda_v13, cc 12.1) |
| RAM | 121 GiB unified | 121 GiB unified |
| GPU status | **Occupied** — PancsVriend job since 2026-06-11 | **Free** — use this for runs |
| Start Ollama | `sudo systemctl start ollama` | `nohup ~/.local/bin/ollama serve > /tmp/ollama.log 2>&1 &` |

**Active run machine: apape2.** SSH in, then run from `~/zit/GS-LLM-Traders/`.

### Syncing output back to apape1 and git

```bash
# From apape1:
bash ~/zit/GS-LLM-Traders/sync-this.sh
```

This rsyncs `output/` and `logs/` from apape2, commits, and pushes.

### Notifications

Progress emails to: `dr.duus@gmail.com`, `ndhamej1@binghamton.edu`

Send manually:
```bash
python3 ~/bin/send_notify.py "Subject" "Body"  # sends to dr.duus only
# For both recipients use inline Python (see watcher pattern below)
```

Notify on: smoke-test result, each script completion, and any fatal error.

---

### NEXT ACTION — Resume Step 4 on apape2

Step 4 (`run_dial_risk_aversion.sh`) was interrupted at 11/30 seeds for `very_averse`. Resume it on apape2:

```bash
ssh apape2.rc.binghamton.edu
cd ~/zit/GS-LLM-Traders

# First rsync any existing output/ from apape1 so --resume finds the progress file
rsync -av apape1:~/zit/GS-LLM-Traders/output/ ~/zit/GS-LLM-Traders/output/

# Then resume step 4
nohup bash batch/run_dial_risk_aversion.sh gemma3-27b 30 42 --resume \
  > logs/dial_risk_aversion_gemma3-27b_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

After step 4 completes, run steps 5–9 in order the same way.

---

### Step 0 — Smoke test (run on apape2 before any long job)

```bash
python3 attanasi_experiment.py --model gemma3-27b --steps 3 --runs 1 --seed 0
```

Expected: CSV output in `output/`, `Inference: local Ollama` in header, no HTTP errors.

---

### Subsequent runs (in order, all on apape2)

| Order | Script | Runs | Est. time (GPU TBD) | What it tests |
|-------|--------|------|---------------------|---------------|
| 4 | `run_dial_risk_aversion.sh` | 150 | TBD | 5 risk-aversion levels |
| 5 | `run_dial_aggressiveness.sh` | 150 | TBD | 5 aggressiveness levels |
| 6 | `run_dial_profit_orientation.sh` | 150 | TBD | 5 profit-orientation levels |
| 7 | `run_temperature_sweep.sh` | 150 | TBD | 5 temperature levels |
| 8 | `run_full_factorial.sh` | 625 | TBD | 125 dial×temp combos × 5 runs |
| 9 | `run_human_baseline_all_models.sh` | 300 | TBD | Human baseline across all 10 models |

**Time estimates are TBD** — waiting on GPU benchmark result (see Task 3 above). At 18 min/run (server baseline), total is ~514 h. GPU local speed will differ; update when benchmark email arrives.

For each script: use `--resume` if interrupted. After completion, run `sync-this.sh` from apape1 to commit and push.

