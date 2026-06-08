# Software Design Specification (SDS)

## Attanasi et al. (2021) Perfect Competition Replication with LLM Agents

| Field | Value |
|-------|-------|
| **Project** | GS-LLM: LLM Agent-Based Double Auction |
| **Paper** | Attanasi, Centorrino & Moscati (2021), "Controlling monopoly power in a double-auction market experiment," JPET 23(5), 1074-1101 |
| **Version** | 3.0 |
| **Date** | 2026-03-29 |
| **Author** | Nency Dhameja |
| **Repository** | `https://github.com/nencydhameja/LLM_ABM_Project/tree/main/GS-LLM` |

---

## 1. Purpose

This document specifies the software design for replicating the perfect competition treatment of Attanasi et al. (2021) using LLM agents in place of human subjects. The system implements a continuous double auction (CDA) market where 4 sellers and 24 buyers trade via API calls to large language models hosted on a Binghamton University server.

## 2. System Overview

### 2.1 Source Files

| File | Purpose |
|------|---------|
| `attanasi_experiment.py` | Core experiment logic: Agent, DoubleAuction, CompetitionExperiment classes |
| `attanasi_config.yaml` | Experimental parameters, LLM model configs, prompt templates |
| `compute_table_b3.py` | Post-experiment analysis (replicates Table B.3 from paper) |
| `parse_trades.py` | Utility to extract trade data from output CSVs |
| `run_gui.py` | GUI wrapper for running experiments |

### 2.2 Architecture

```
attanasi_config.yaml
        |
        v
CompetitionExperiment
   |-- creates --> Agent (4 sellers + 24 buyers)
   |-- creates --> DoubleAuction (order book engine)
   |-- runs ----> _run_period() x 3 periods
                     |
                     +--> _reshuffle_buyer_valuations()  [Spec 1]
                     |
                     +--> Agent.decide_price()
                     |       |-- _build_system_prompt()    [first call + Spec 12]
                     |       |-- _build_decision_prompt()  [every call]
                     |       |-- _query_llm()              [multi-turn API, Spec 11]
                     |       +-- _extract_price()          [parse + validate]
                     |
                     +--> DoubleAuction.submit_bid() / submit_ask()
                     |       |-- improvement rule check   [Spec 2]
                     |       |-- match check (crossing spread)
                     |       +-- _execute_trade() [clears book, Spec 3]
                     |
                     +--> Agent.record_outcome()  [feedback, Spec 11]
                     |
                     +--> Agent.save_period_summary()  [Spec 12]
                     |
                     +--> Output: master CSV (every bid/ask/trade)
```

### 2.3 Experimental Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Sellers | 4 | Online Appendix p.1 |
| Buyers | 24 (4 at each of 6 valuations) | Online Appendix p.1, Table A.1 |
| Seller cost | 12 (uniform) | Online Appendix p.1 |
| Buyer valuations | 20, 17, 15, 13, 8, 0 | Online Appendix p.1, Figure A.2 |
| Seller capacity | 24 units/period | Online Appendix p.1 |
| Buyer capacity | 1 unit/period | Online Appendix p.1 |
| Periods | 3 | Main paper Section 3, p.1080 |
| Price range (buyer) | [0, own valuation] | Online Appendix p.4, Figure A.5 |
| Price range (seller) | [own cost, 999] | Online Appendix p.4, Figure A.4 |
| Idle-round stop | 5 consecutive no-trade rounds | Main paper Section 3, p.1080 |
| Max rounds/period | 30 (ceiling) | Config default |
| CE price | ~12.5 (range 12-13) | Online Appendix p.1, Figure A.2 |
| CE quantity | 16 units | Online Appendix p.1, Figure A.2 |

### 2.4 Human Benchmarks (Online Appendix p.5, Table B.1, Co played first)

| Period | Avg Price |
|--------|-----------|
| 1 | 14.49 |
| 2 | 14.10 |
| 3 | 13.50 |

---

## 3. Screen-to-Code Cross-Reference (Online Appendix p.4, Figures A.4 and A.5)

This table maps every element visible on the human subject's screen to the corresponding code implementation.

| Screen Element (Figs A.4/A.5) | Example Value | Code Location | Config Location |
|-------------------------------|---------------|---------------|-----------------|
| "You are the seller S1" / "buyer B6" | S1, B6 | `Agent.agent_id` (py:70) | buyer/seller_system: `{agent_id}` (yaml:141/158) |
| "Round: 1/3" | 1/3 | `market_state["period"]` (py:454) | buyer/seller_system: `{period}` (yaml:144/161) |
| "Cumulated payoffs: 0" | 0 | `Agent.cumulated_payoffs` (py:95) | decision: `{cumulated_payoffs}` (yaml:190) |
| "Period payoffs: 0" | 0 | `Agent.period_payoffs` (py:94) | decision: `{period_payoffs}` (yaml:189) |
| "Available units for purchase: 19" | 19 | `market_state["units_to_buy"]` (py:436-443) | decision: `{units_to_buy}` (yaml:177) |
| "Available units for sale: 91" | 91 | `market_state["units_to_sell"]` (py:436-443) | decision: `{units_to_sell}` (yaml:178) |
| "Units sold/bought: 5" | 5 | `market_state["units_traded"]` (py:453) | decision: `{units_traded}` (yaml:179) |
| Best buy offer "B10:15*" | B10:15 | `best_bid_agent` + `best_bid` (py:448-449) | decision: `{best_bid}` (yaml:181) |
| Best sell offer "S4:16*" | S4:16 | `best_ask_agent` + `best_ask` (py:448-449) | decision: `{best_ask}` (yaml:182) |
| Transaction list "B22:S4:19" | IDB:IDS:Price | `_build_decision_prompt()` (py:222-233) | decision: `{transaction_history}` (yaml:185) |
| Your situation table (Unit/Cost/Ask/Price/Payoffs) | See Fig A.4 | `_build_decision_prompt()` (py:236-267) | decision: `{your_situation}` (yaml:188) |
| "Please enter an integer between 12 and 999" (seller) | [cost, 999] | `_extract_price()` (py:371-385) | seller_system: `between {cost} and 999` (yaml:164) |
| "Please enter an integer between 0 and 16" (buyer) | [0, val] | `_extract_price()` (py:371-385) | buyer_system: `between 0 and {valuation}` (yaml:147) |
| Validate / Cancel buttons | submit or skip | `Agent.decide_price()` (py:116-156) | decision: `What integer price...` (yaml:192) |

---

## 4. Design Specifications

### Spec 1: Buyer Valuation Reshuffling

| Field | Detail |
|-------|--------|
| **Category** | Experimental Design |
| **Requirement** | Each period, the 24-value multiset (4 copies of each of 6 valuations) is shuffled and randomly reassigned to buyers. Buyers do not keep the same valuation across periods. |
| **Paper** | Online Appendix p.1, Figure A.2: demand function shows 6 valuation steps with 4 buyers each. Main paper Section 3 p.1080: "valuations...were reshuffled across buyers." |
| **Code** | `attanasi_experiment.py` : `CompetitionExperiment._reshuffle_buyer_valuations()` (lines 655-663). Called at line 687 in `_run_period()`. |
| **Config** | `attanasi_config.yaml` lines 13-20: `buyer_valuations` list + `num_groups: 4`. |

### Spec 2: Improvement Rule Enforcement

| Field | Detail |
|-------|--------|
| **Category** | Auction Mechanism |
| **Requirement** | New bids must strictly exceed the current best bid. New asks must be strictly below the current best ask. Non-improving orders are rejected. |
| **Paper** | Main paper Section 3 p.1080: "the CDA improvement rule...a new bid (ask) must improve upon the current best bid (ask)." |
| **Code** | `attanasi_experiment.py` : `DoubleAuction.submit_bid()` lines 470-476, `submit_ask()` lines 510-516. Returns `(None, "rejected")` for non-improving orders. |
| **Config** | `attanasi_config.yaml` buyer_system lines 155-156, seller_system lines 172-173: "IMPORTANT: Your new bid/ask must be strictly HIGHER/LOWER..." |

### Spec 3: Order Book Clearing After Trade

| Field | Detail |
|-------|--------|
| **Category** | Auction Mechanism |
| **Requirement** | After each trade, all standing bids and asks are removed. Trading restarts with an empty book. |
| **Paper** | Main paper Section 3 p.1080: "After each transaction, the book is cleared." |
| **Code** | `attanasi_experiment.py` : `DoubleAuction._execute_trade()` lines 577-579: `self.bids.clear(); self.asks.clear()`. |
| **Config** | `attanasi_config.yaml` buyer_system line 156, seller_system line 173: "After each trade, the order book is cleared..." |

### Spec 4: Buyer Creation Comment Fix

| Field | Detail |
|-------|--------|
| **Category** | Documentation |
| **Requirement** | Code comment corrected to "4 at each of 6 valuations (reshuffled each period)" matching Figure A.2 demand schedule. |
| **Paper** | Online Appendix p.1, Figure A.2: step function shows 4 buyers at valuations 20, 17, 15, 13, 8, 0. |
| **Code** | `attanasi_experiment.py` : `CompetitionExperiment._create_agents()` line 645. |

### Spec 5: Event-Based Stopping Rule

| Field | Detail |
|-------|--------|
| **Category** | Experimental Design |
| **Requirement** | Periods end when no trades occur for N consecutive rounds (default: 5), with a hard ceiling of 30 rounds. |
| **Paper** | Main paper Section 3 p.1080-1081: periods end after a sequence of idle rounds, not at a fixed step count. |
| **Code** | `attanasi_experiment.py` : `CompetitionExperiment._run_period()`. Counter `no_trade_rounds` initialized at line 705. Incremented at line 823, reset at line 821. Break at lines 825-830. |
| **Config** | `attanasi_config.yaml` line 26: `idle_rounds_to_stop: 5`. Line 25: `num_steps: 30` (ceiling). |

### Spec 6: Period vs Session Payoff Separation

| Field | Detail |
|-------|--------|
| **Category** | Experimental Design |
| **Requirement** | `period_payoffs` resets each period; `cumulated_payoffs` accumulates across session. Both shown on screen. |
| **Paper** | Online Appendix p.4, Fig A.4: "Round payoffs" column in situation table (period-level). Top bar: "Cumulated payoffs: 0" (session-level). Fig A.5: same layout for buyers. |
| **Code** | `attanasi_experiment.py` : `Agent.__init__()` lines 93-95, `reset_period()` line 108, `_execute_trade()` lines 569-573, `_build_decision_prompt()` lines 286-287. |
| **Config** | `attanasi_config.yaml` decision template lines 189-190. |

### Spec 7: Best Bid/Ask Displays Agent IDs

| Field | Detail |
|-------|--------|
| **Category** | Prompt Fidelity |
| **Requirement** | Best bid/ask shows agent ID + price (e.g., "B10: 15"), matching the "ID&Price" column on screen. |
| **Paper** | Online Appendix p.4, Fig A.4: "Current buy offers: Best buy offer" table shows "B10:15*". Fig A.5: same. |
| **Code** | `attanasi_experiment.py` : `DoubleAuction.market_state()` lines 448-449 (returns `best_bid_agent`, `best_ask_agent`). `Agent._build_decision_prompt()` lines 268-278 (formats display). |
| **Config** | `attanasi_config.yaml` decision template lines 181-182. |

### Spec 8: Correct Price Ranges

| Field | Detail |
|-------|--------|
| **Category** | Prompt Fidelity |
| **Requirement** | Buyer range [0, valuation], seller range [cost, 999]. |
| **Paper** | Online Appendix p.4, Fig A.4 seller screen: "Please enter an integer between 12 and 999 to submit a sell offer." Fig A.5 buyer screen: "Please enter an integer between 0 and 16 to submit a buy offer." |
| **Code** | `attanasi_experiment.py` : `Agent._extract_price()` lines 371-385 (removed [0,30] range, kept budget constraint only). |
| **Config** | `attanasi_config.yaml` buyer_system line 147: `between 0 and {valuation}`. seller_system line 164: `between {cost} and 999`. |

### Spec 9: Remove Profit Recommendation

| Field | Detail |
|-------|--------|
| **Category** | Prompt Fidelity |
| **Requirement** | Removed "We recommend making at least 1 point of profit." Not on experimental screen. |
| **Paper** | Online Appendix p.4, Figs A.4-A.5: no such recommendation anywhere on screen. Only "Validate" and "Cancel" buttons. |
| **Code** | N/A (removal only). |
| **Config** | `attanasi_config.yaml` buyer_system line 148, seller_system line 165: reduced to "You are not obligated to buy/sell." |

### Spec 10: Remove Cost Symmetry Information

| Field | Detail |
|-------|--------|
| **Category** | Prompt Fidelity |
| **Requirement** | Removed "The production cost is the same for each unit across sellers." Human subjects see only their own cost. |
| **Paper** | Online Appendix p.4, Figs A.4-A.5: seller sees "Cost: 12" in situation table but no statement about other sellers' costs. Main paper Section 3: subjects know own cost only. |
| **Code** | N/A (removal only). |
| **Config** | `attanasi_config.yaml`: line removed from both buyer_system and seller_system prompts. |

### Spec 11: Multi-Turn Conversation History

| Field | Detail |
|-------|--------|
| **Category** | LLM Architecture |
| **Requirement** | LLM agents maintain conversation history within each period. Each turn includes the market state prompt, the LLM's response, and feedback on the outcome (traded/posted/rejected). |
| **Paper** | Implicit: human subjects have persistent memory during a trading period. They remember their past submissions and outcomes. Without history, LLMs repeat rejected bids indefinitely (observed with mistral-instruct). |
| **Code** | `attanasi_experiment.py` : `Agent.decide_price()` lines 116-156 (builds multi-turn messages), `Agent.record_outcome()` lines 158-160 (stores feedback), `Agent._get_messages_for_api()` lines 294-303 (caps at 20 messages, preserves first), `Agent._query_llm()` lines 305-350 (sends full history). Feedback in `_run_period()` lines 778-817 (traded/posted/rejected) and lines 723-740 (constraint/parse errors). |
| **Fields** | `Agent.message_history` (line 97), `Agent.pending_feedback` (line 98). Reset in `reset_period()` lines 109-110. |

### Spec 12: Cross-Period Memory

| Field | Detail |
|-------|--------|
| **Category** | LLM Architecture |
| **Requirement** | At the end of each period, a summary of the agent's outcome and market results is saved and included in subsequent periods' system prompts. Drives monotonic price convergence. |
| **Paper** | Online Appendix p.5, Table B.1: human prices converge monotonically (p_Co: 14.49 -> 14.10 -> 13.50). Main paper p.1085: "prices converge toward the CE across periods." Table B.2: Period 2 coefficient -0.579*** (p<0.001), Period 3 coefficient -0.952*** (p<0.001), confirming significant cross-period learning. |
| **Code** | `attanasi_experiment.py` : `Agent.save_period_summary()` lines 162-184 (builds summary text), `Agent._build_system_prompt()` lines 204-208 (appends summaries to prompt), `CompetitionExperiment._run_period()` lines 846-851 (calls save for all agents). |
| **Fields** | `Agent.period_summaries: List[str]` (line 100) -- never reset across periods. |

**Example cross-period prompt (Period 3):**
```
=== YOUR EXPERIENCE FROM PREVIOUS PERIODS ===
  Period 1: You bought 1 unit at price 15, profit=5. Market: 16 trades, avg price=14.2.
  Period 2: You bought 1 unit at price 13, profit=7. Market: 15 trades, avg price=13.8.
```

### Spec 13: Cancel/Pass Option

| Field | Detail |
|-------|--------|
| **Category** | Prompt Fidelity |
| **Requirement** | Agents can choose not to submit a bid/ask in any given round, matching the "Cancel" button on the experimental screen. |
| **Paper** | Online Appendix p.4, Fig A.4: seller screen shows "Validate" and "Cancel" buttons side by side. Fig A.5: buyer screen shows the same. Subjects are not forced to submit an order every round. |
| **Code** | `attanasi_experiment.py` : `Agent._extract_price()` lines 362-365 (detects PASS/CANCEL/SKIP in LLM response, returns `(None, "pass")`). `CompetitionExperiment._run_period()` lines 746-750 (handles "pass" status, records feedback for multi-turn history). |
| **Config** | `attanasi_config.yaml` decision template line 192: "Reply with ONLY a single integer, or PASS to skip this round." |

---

## 5. Data Flow

### 5.1 Per-Step Flow

```
1. CompetitionExperiment._run_period() shuffles agent order
2. For each agent:
   a. Agent.decide_price(market_state)
      - Builds user message (system prompt on first call, feedback + state on subsequent)
      - Appends to message_history
      - Calls _query_llm() with full history (truncated to 20 messages)
      - Parses integer from LLM response via _extract_price()
      - Returns (price, status)
   b. If valid price: DoubleAuction.submit_bid() or submit_ask()
      - Checks improvement rule -> reject if non-improving
      - Checks for crossing spread -> trade
      - If trade: _execute_trade() clears book, updates agents
      - If no trade and no reject: post to order book
   c. Agent.record_outcome(feedback) stores result for next turn
   d. Decision logged to all_decisions list
3. Check idle-round stopping criterion (Spec 5)
4. Check if all profitable trades exhausted
```

### 5.2 Output Schema (Master CSV)

| Column | Type | Description |
|--------|------|-------------|
| run_id | int | Run number within batch |
| seed | int | Random seed for this run |
| model | str | LLM model key |
| timestamp | str | Batch timestamp |
| period | int | Trading period (1-3) |
| step | int | Trading round within period |
| agent_id | str | Agent identifier (B1-B24, S1-S4) |
| agent_type | str | "buyer" or "seller" |
| action | str | "bid" or "ask" |
| submitted_price | int | Price submitted by agent |
| reservation_value | int | Agent's valuation (buyer) or cost (seller) |
| matched | bool | Whether a trade occurred |
| trade_price | int | Transaction price (if traded) |
| counterparty | str | ID of trading partner (if traded) |
| profit | int | Agent's profit from trade (if traded) |

---

## 6. LLM Model Configurations

All models accessed via Binghamton University server (`chat.binghamton.edu/api/chat/completions`).

| Model Key | Model ID | Parameters | Timeout | Max Tokens |
|-----------|----------|------------|---------|------------|
| mistral-instruct | mistral:instruct | 7B | 60s | 50 |
| llama3.1-70B | llama3.1:70B | 70B | 120s | 100 |
| llama3.3 | llama3.3:latest | 70B | 120s | 100 |
| mixtral-8x22b | mixtral:8x22b-instruct | 8x22B | 180s | 100 |
| gemma3-27b | gemma3:27b | 27B | 60s | 100 |
| gemma3-4b | gemma3:4b | 4B | 30s | 50 |
| phi4 | phi4:latest | 14B | 60s | 75 |
| qwen2.5-coder-32B | qwen2.5-coder:32B | 32B | 120s | 100 |
| qwq | qwq:latest | 32B | 120s | 100 |
| gpt-oss-20b | gpt-oss:20b | 20B | 60s | 75 |
| hermes3 | hermes3:latest | 8B | 30s | 75 |
| codellama-70B | codellama:70B | 70B | 120s | 100 |

All models: `temperature=0.7`.

---

## 7. Prompt Templates

### 7.1 Buyer System Prompt (yaml lines 140-156)

```
You are participating in a market experiment as a BUYER ({agent_id}).
There are 4 sellers and 24 buyers in this market.
Each of the 4 sellers can sell up to 24 units. Each buyer can buy at most 1 unit per period.
This is Period {period} of 3.

Your valuation for one unit is {valuation}. Your profit = Valuation - Price.
You can only submit integer prices between 0 and {valuation}.
You are not obligated to buy.
You CANNOT bid above your valuation of {valuation}.

How trades work: If your bid meets or exceeds the current lowest ask, a trade occurs
between you and the seller holding that ask. Similarly, if a seller's ask meets or is
below the current highest bid, a trade occurs.

IMPORTANT: Your new bid must be strictly HIGHER than the current best bid, or it will
be rejected. After each trade, the order book is cleared and bidding starts fresh.
```

### 7.2 Seller System Prompt (yaml lines 157-173)

```
You are participating in a market experiment as a SELLER ({agent_id}).
There are 4 sellers and 24 buyers in this market.
Each of the 4 sellers can sell up to 24 units. Each buyer can buy at most 1 unit per period.
This is Period {period} of 3.

Your production cost per unit is {cost}. Your profit = Price - Cost.
You can only submit integer prices between {cost} and 999.
You are not obligated to sell.
You CANNOT ask below your cost of {cost}.

How trades work: If your ask meets or is below the current highest bid, a trade occurs
between you and the buyer holding that bid. Similarly, if a buyer's bid meets or exceeds
the current lowest ask, a trade occurs.

IMPORTANT: Your new ask must be strictly LOWER than the current best ask, or it will
be rejected. After each trade, the order book is cleared and asking starts fresh.
```

### 7.3 Decision Prompt (yaml lines 174-192)

```
=== MARKET SCREEN ===
Trading round: {current_step} of {total_steps}. No more trades can happen after round {total_steps}.
Available units for purchase: {units_to_buy}
Available units for sale: {units_to_sell}
Units sold/bought so far: {units_traded}

Current best buy offer (highest bid): {best_bid}
Current best sell offer (lowest ask): {best_ask}

Transactions this period:
{transaction_history}

=== YOUR SITUATION ===
{your_situation}
Period payoffs: {period_payoffs}
Cumulated session payoffs: {cumulated_payoffs}

What integer price do you want to {action_verb}? Reply with ONLY a single integer.
```

### 7.4 Outcome Feedback (appended to next user message, Spec 11)

| Outcome | Feedback Template |
|---------|-------------------|
| Traded (buyer) | "TRADED! You bought 1 unit at price {price}. Your profit: {profit}." |
| Traded (seller) | "TRADED! You sold 1 unit at price {price}. Your profit: {profit}." |
| Posted | "Your {action} of {price} has been POSTED to the order book." |
| Rejected | "Your {action} of {price} was REJECTED because it does not improve on the current best {action}." |
| Constraint error | "Your price was rejected because it violated your budget constraint." |
| Parse error | "Could not understand your response. Please reply with a single integer." |

---

## 8. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-29 | Initial implementation with fixes 1-6 (reshuffling, improvement rule, book clearing, comment, stopping rule, payoff separation) |
| 2.0 | 2026-03-29 | Added fixes 7-11 (best bid/ask IDs, correct price ranges, removed profit recommendation, removed cost symmetry info, multi-turn conversation history) |
| 2.1 | 2026-03-29 | Added fix 12 (cross-period memory for monotonic convergence) |
| 3.0 | 2026-03-29 | Restored all 12 specs after merge conflict revert. Updated all line references. Added precise paper page/section/figure citations. |
| 3.1 | 2026-03-29 | Added Spec 13 (Cancel/Pass option matching experiment Cancel button). Added persona treatment (--persona CLI flag, 3 personas in config). Added persona column to output CSV. |
