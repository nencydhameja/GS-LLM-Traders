# GS-LLM Findings

## Refined Memory Treatment (April 2026)

### Design

We compare two cross-period memory formats injected into the LLM system prompt:

- **Baseline:** One-line summary per period (e.g., "Period 2: You bought 1 unit at price 15, profit=5. Market: 12 trades, avg price=14.8.")
- **Refined:** Multi-line detailed recap including:
  - Full action log (every bid/ask, outcome)
  - Own outcome with valuation/cost
  - Market price distribution (range, median, avg)
  - Price trajectory in step order

### Results (gemma3-27b, 30 runs each)

| Period | Baseline | Refined | Human |
|--------|----------|---------|-------|
| 1      | 14.38    | 14.26   | 14.49 |
| 2      | 14.67    | 14.56   | 14.10 |
| 3      | 14.88    | 14.70   | 13.50 |
| **P1→P3 drift** | **+0.50** | **+0.44** | **−0.99** |

CE (competitive equilibrium) = 12.7. Human prices converge toward CE across periods; LLM prices drift away from it.

### Interpretation

1. **Refined memory does not fix the convergence failure.** The P1→P3 drift is +0.50 (baseline) vs +0.44 (refined) — no meaningful improvement.

2. **The problem is not information — it is reasoning.** Refined memory gives agents everything they need (full distribution, trajectory, own profit), but they cannot translate that into the adaptive undercutting behavior that drives human convergence.

3. **LLMs anchor rather than learn.** Instead of strategically adjusting to undercut competitors (as humans do), LLMs anchor on recently observed prices and reproduce them, causing mild upward drift.

4. **This rules out "insufficient memory" as an explanation for convergence failure.** The bottleneck is a deeper reasoning limitation in how LLMs process sequential market experience.

### Model comparison (refined memory)

| Model | P1 Mean | Trades/16 | Quality |
|-------|---------|-----------|---------|
| gemma3-27b | 14.26 | 15.3 | Good — close to human P1, high trade volume |
| qwen2.5-coder-32B | 14.98 | 9.6 | Moderate — slight overshoot, fewer trades |
| hermes3 | 17.19 | 4.8 | Poor — large overshoot, very low trade volume |
