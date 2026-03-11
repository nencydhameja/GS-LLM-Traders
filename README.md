# GS-LLM-Traders

LLM replication of **Attanasi, Centorrino & Moscati (2021)** "Controlling monopoly power in a double-auction market experiment" (*JPET* 23(5), 1074-1101).

We replace human subjects with LLM agents and compare trading behavior across 5 market structures.

## Status

| Phase | Market Structure | Status |
|-------|-----------------|--------|
| 1 | Perfect Competition (4 sellers, 24 buyers) | Running 30-run batch |
| 2 | Competition with quotas (capacity = 6/seller) | Not started |
| 3 | Cartel (sellers agree on price) | Not started |
| 4 | Cartel with quotas | Not started |
| 5 | Monopoly (1 seller, 6 buyers) | Not started |

## Experimental Design

Replicates the paper's Phase 1 (Perfect Competition):

- **Agents**: 4 sellers (cost = 12) + 24 buyers (valuations: 20, 17, 15, 13, 8, 0 x 4 groups)
- **Mechanism**: Continuous double auction with price-time priority
- **Periods**: 3 trading periods per run, 30 trading rounds per period
- **LLM**: Llama 3.3 70B via Binghamton University server (OpenAI-compatible API)
- **Runs**: 30 independent runs (seeds 42-71) for bootstrap 95% CIs
- **Prompts**: Mirror exactly what human subjects see on screen (Appendix Figs A.4/A.5) including market structure, transaction history, own situation, cumulated payoffs, and trading round countdown

### Human Benchmarks (Table B.1, Competition played first)

| Period | Human Avg Price | CE Price | CE Quantity |
|--------|----------------|----------|-------------|
| 1 | 14.49 | ~12 | 16 |
| 2 | 14.10 | ~12 | 16 |
| 3 | 13.50 | ~12 | 16 |

## Setup

### Requirements
```bash
pip install requests pyyaml
```

### Configuration
All experimental parameters and LLM model configs are in `attanasi_config.yaml`. The API endpoint is the Binghamton University OpenWebUI server.

### Running

```bash
# Single run
python attanasi_experiment.py --model llama3.3 --steps 30 --runs 1 --seed 42

# Full 30-run batch
python attanasi_experiment.py --model llama3.3 --steps 30 --runs 30 --seed 42

# Resume after server interruption
python attanasi_experiment.py --model llama3.3 --steps 30 --runs 30 --seed 42 --resume
```

### Available Models
All hosted on Binghamton server: `mistral-instruct`, `llama3.1-70B`, `llama3.3`, `mixtral-8x22b`, `gemma3-27b`, `gemma3-4b`, `phi4`, `qwen2.5-coder-32B`, `qwq`, `gpt-oss-20b`, `hermes3`, `codellama-70B`

## Output

All output goes to `output/`:

| File | Description |
|------|-------------|
| `attanasi_trades_*.csv` | Every trade across all runs (with run_id, seed, model tags) |
| `attanasi_summary_*.csv` | Per-period averages per run + LLM error counts |
| `attanasi_aggregate_*.csv` | Cross-run means, SDs, bootstrap 95% CIs |
| `attanasi_progress_*.json` | Resume checkpoint (deleted when batch completes) |

## Key Files

| File | Description |
|------|-------------|
| `attanasi_experiment.py` | Main simulation: Agent, DoubleAuction, CompetitionExperiment classes |
| `attanasi_config.yaml` | Parameters, LLM configs, prompt templates |
| `paper.tex` | Limitations (temporal reasoning) |
| `jpet12504-sup-0001-online_appendix.pdf` | Original paper appendix with instructions and benchmarks |

## Early Findings (Phase 1)

From initial runs with Llama 3.3 70B:

- LLM sellers exhibit **tacit collusion** -- all 4 sellers consistently ask 17-19, refusing to undercut each other, even with 96 units available vs 24 buyers
- **Severe under-trading** -- 5-9 trades per period vs 16 expected at competitive equilibrium
- **No convergence** -- humans converge toward CE (14.49 -> 13.50); LLMs do not show consistent downward trend
- Some buyers attempt to bid **above their valuation** after observing high transaction prices (caught by constraint validation)

Full 30-run results with bootstrap CIs pending.
