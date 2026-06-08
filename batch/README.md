# Batch scripts

All scripts are self-contained, use `--resume` for fault tolerance, and log to `../logs/`.

Each takes optional positional args: `MODEL` (default: gemma3-27b), `RUNS` (default: 30), `SEED` (default: 42).

| Script | What it runs | Time (per model, ~18 min/run) |
|---|---|---|
| `run_human_baseline.sh` | **Exactly humans**: no dial, no persona, no temp, baseline memory. Single model. | ~9 h |
| `run_human_baseline_all_models.sh` | Above, looped over 10 models, sequential. | ~90 h |
| `run_memory_ablation.sh` | Baseline vs refined memory, same model, 30 runs each. | ~18 h |
| `run_persona_ablation.sh` | undergrad_econ / grad_econ / undergrad_human, 30 runs each. | ~27 h |
| `run_dial_risk_aversion.sh` | 5 risk-aversion levels × 30 runs. | ~45 h |
| `run_dial_aggressiveness.sh` | 5 aggressiveness levels × 30 runs. | ~45 h |
| `run_dial_profit_orientation.sh` | 5 profit-orientation levels × 30 runs. | ~45 h |
| `run_temperature_sweep.sh` | 5 temps (0.0, 0.3, 0.5, 0.7, 1.0) × 30 runs. | ~45 h |
| `run_full_factorial.sh` | 125 dial×temp combos × 5 runs (or override). | ~190 h @ 5 runs |

## Quick start

```bash
chmod +x batch/*.sh
nohup ./batch/run_human_baseline.sh gemma3-27b 30 42 > /dev/null 2>&1 &
tail -f logs/human_baseline_gemma3-27b_*.log
```

To run in parallel on a server, fire multiple scripts in separate `nohup` processes — each writes a distinct progress file, no conflicts.
