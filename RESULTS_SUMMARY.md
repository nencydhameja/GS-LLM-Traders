# Attanasi et al. (2021) LLM Replication — Results Summary

## Competitive Equilibrium (CE)

The CE is a property of the market structure, not the players — identical for humans and LLMs.

- **24 buyers** = 4 groups × 6 buyers (valuations: 20, 17, 15, 13, 8, 0)
- **4 sellers**, effectively unlimited capacity, cost = 12
- Buyers with valuation > 12 want to trade: valuations 20, 17, 15, 13 → 4 × 4 groups = **16 trades**
- Marginal buyer value = 13, seller cost = 12 → **CE price = 12–13 (midpoint ~12.5)**
- Buyers at valuations 8 and 0 are extramarginal (cannot profitably trade)

## Human Benchmarks (Table B.1, Co first)
- Period 1: 14.49
- Period 2: 14.10
- Period 3: 13.50
- CE: ~12.50, 16 trades
- Humans converge toward CE across periods (14.49 → 14.10 → 13.50)

## Bootstrap 95% CI Results

### Complete (30/30 runs)

| Model | Period | LLM Mean | 95% CI | Human | Match? | Trades/16 |
|---|---|---|---|---|---|---|
| gemma3-27b | 1 | 14.38 | [14.16, 14.60] | 14.49 | **YES** | 15.2 |
| gemma3-27b | 2 | 14.67 | [14.46, 14.87] | 14.10 | no | 15.3 |
| gemma3-27b | 3 | 14.88 | [14.67, 15.09] | 13.50 | no | 14.8 |
| qwen2.5-coder-32B | 1 | 14.59 | [14.32, 14.88] | 14.49 | **YES** | 9.5 |
| qwen2.5-coder-32B | 2 | 14.85 | [14.52, 15.19] | 14.10 | no | 8.9 |
| qwen2.5-coder-32B | 3 | 14.81 | [14.41, 15.22] | 13.50 | no | 8.0 |
| hermes3 | 1 | 17.48 | [16.11, 18.83] | 14.49 | no | 3.3 |
| hermes3 | 2 | 17.27 | [16.40, 18.14] | 14.10 | no | 4.0 |
| hermes3 | 3 | 15.22 | [14.63, 15.95] | 13.50 | no | 2.4 |

### In Progress (19/30 runs)

| Model | Period | LLM Mean | 95% CI | Human | Match? | Trades/16 |
|---|---|---|---|---|---|---|
| llama3.1-70B | 1 | 14.77 | [14.54, 15.02] | 14.49 | no (barely) | 14.8 |
| llama3.1-70B | 2 | 14.72 | [14.46, 14.97] | 14.10 | no | 15.5 |
| llama3.1-70B | 3 | 14.52 | [14.35, 14.72] | 13.50 | no | 15.4 |

## Key Findings

### 1. LLMs replicate initial price discovery but lack cross-period competitive learning

- **Period 1:** Human benchmark falls inside the 95% bootstrap CI for gemma3-27b and qwen2.5-coder-32B. We cannot reject that LLM initial prices equal human prices at the 5% level.
- **Periods 2-3:** All models fail. LLM prices stay flat or drift upward (14.4 → 14.7 → 14.9) while human prices converge downward toward competitive equilibrium (14.49 → 14.10 → 13.50).
- **Interpretation:** LLMs can process market rules and discover plausible initial prices, but they do not learn from repeated market interaction the way human subjects do. Humans update beliefs across periods and push prices toward equilibrium; LLMs treat each period roughly independently.

### 2. Memory is present but not used effectively

LLM agents have two types of memory implemented:

- **Within-period (Spec 11):** Multi-turn conversation history. Each agent accumulates all its bids, asks, trade outcomes, and feedback during a period. The LLM sees its full interaction history when making each new decision.
- **Cross-period (Spec 12):** Period summaries carry forward. At the end of each period, a summary is saved (e.g., "Period 1: You bought 1 unit at price 15, profit=5. Market: 14 trades, avg price=14.3."). These summaries are prepended to the system prompt in subsequent periods.

**The information is there — the LLMs don't act on it.** In Period 2, the LLM knows that avg price was ~14.4 in Period 1. A human subject would use this to compete more aggressively and undercut. The LLM does not adjust its bidding behavior meaningfully. This is not a memory implementation gap — it is a **learning gap**. LLMs receive the same experiential feedback that drives human convergence toward CE, but fail to translate that feedback into adaptive price-setting behavior.

This finding is consistent with the broader LLM-as-agent literature: LLMs can follow rules and process information, but they lack the experiential learning loop that drives human behavior change across repeated interactions.

## Model Rankings

1. **gemma3-27b** — Best overall. P1 match, near-CE trade volume (15/16).
2. **qwen2.5-coder-32B** — P1 match, but only ~9/16 trades (low volume).
3. **llama3.1-70B** — Strong volume (15/16) but P1 barely misses at 19 runs. Pending full 30 runs.
4. **hermes3** — Poor. Prices too high (~17), barely trades (3/16).

## Methodology
- 30 independent runs per model (different random seeds)
- 10,000 bootstrap resamples for 95% percentile CIs
- Bootstrap resamples run-level mean prices (not individual trades)
- Server: Binghamton University (`chat.binghamton.edu`)
- CI interpretation (frequentist): "If we repeated this procedure many times, 95% of the resulting intervals would contain the true mean."
