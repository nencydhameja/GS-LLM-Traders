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

### Machine

| Resource | Spec |
|----------|------|
| Host | `promaxgb10-4ae4` |
| OS | Ubuntu (Linux 6.17, aarch64) |
| CPU | ARM Cortex-X925 (4P) + Cortex-A725 (16E), 20 cores, 3.9 GHz max |
| RAM | 121 GiB total, ~74 GiB available |
| GPU | NVIDIA GB10 |
| Operator | `apape` |

Scripts are run from this machine. All output is committed and pushed to this repo as it accumulates.

### Notifications

Progress emails are sent to:
- `dr.duus@gmail.com`
- `ndhamej1@binghamton.edu`

Notify on: smoke-test result, each script completion, and any fatal error.

---

### Step 0 — Smoke test

Before any long run, verify the stack is working end-to-end with a 1-run, 3-step dry run:

```bash
python attanasi_experiment.py --model gemma3-27b --steps 3 --runs 1 --seed 0
```

Expected: one CSV row per agent decision in `output/`, no HTTP errors. If it passes, proceed. If it fails, stop and diagnose before committing to multi-hour runs.

---

### Step 1 — Human baseline (~9 h)

Shortest script; establishes the primary human-comparable benchmark.

```bash
chmod +x batch/*.sh
nohup ./batch/run_human_baseline.sh gemma3-27b 30 42 > /dev/null 2>&1 &
tail -f logs/human_baseline_gemma3-27b_*.log
```

**During the run:** commit and push `output/` and `logs/` files whenever new files appear (check every ~30 min).

**On completion:** commit all remaining output, push, then send email to both addresses with the `results_*.txt` summary attached.

---

### Subsequent runs (in order)

Run each script only after the previous one completes and its output has been pushed.

| Order | Script | Approx time | What it tests |
|-------|--------|-------------|---------------|
| 2 | `run_memory_ablation.sh` | ~18 h | Baseline vs refined memory |
| 3 | `run_persona_ablation.sh` | ~27 h | undergrad_econ / grad_econ / undergrad_human |
| 4 | `run_dial_risk_aversion.sh` | ~45 h | 5 risk-aversion levels |
| 5 | `run_dial_aggressiveness.sh` | ~45 h | 5 aggressiveness levels |
| 6 | `run_dial_profit_orientation.sh` | ~45 h | 5 profit-orientation levels |
| 7 | `run_temperature_sweep.sh` | ~45 h | 5 temperature levels |
| 8 | `run_full_factorial.sh` | ~190 h | 125 dial×temp combos × 5 runs |
| 9 | `run_human_baseline_all_models.sh` | ~90 h | Human baseline across all 10 models |

For each script: use `--resume` if interrupted, commit/push incrementally, and send a completion email with the results summary.

**Total estimated wall time (sequential):** ~514 h (~21 days).

