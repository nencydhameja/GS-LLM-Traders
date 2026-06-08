# Paper vs Code Audit: Attanasi et al. (2021)

Source: Appendix C Instructions (jpet12504-sup-0001-online_appendix.pdf), Figures A.2, A.4, A.5

## Parameter Matching

| # | Paper (Appendix C) | Code (config + experiment) | Match? |
|---|---|---|---|
| 1 | 4 sellers, 24 buyers | `num_sellers: 4`, `num_buyers: 24` | YES |
| 2 | Each seller can sell 24 units per period | `seller_capacity: 24` | YES |
| 3 | Each buyer can buy at maximum 1 unit per period | `buyer_capacity: 1` | YES |
| 4 | Seller cost = 12, same for all sellers, same across periods | `seller_cost: 12` | YES |
| 5 | Buyer valuations: 20, 17, 15, 13, 8, 0 (from Figure A.2) | `buyer_valuations: [20, 17, 15, 13, 8, 0]` | YES |
| 6 | 4 identical groups of 6 buyers | `num_groups: 4` | YES |
| 7 | Valuations can change across periods for same buyer | `_reshuffle_buyer_valuations()` called each period (Spec 1) | YES |
| 8 | Seller cost same across trading periods | Cost never changes in code | YES |
| 9 | 3 trading periods per phase | `num_periods: 3` | YES |
| 10 | Integer prices between 0 and 30 | `price_range: [0, 30]`, enforced in `_extract_price()` | YES |
| 11 | CE price = 12-13, CE quantity = 16 (Figure A.2) | `competitive_equilibrium: price: 12.5, quantity: 16` | YES |

## Trading Mechanism

| # | Paper (Appendix C) | Code | Match? |
|---|---|---|---|
| 12 | "If a seller's ask becomes the current lowest ask and it meets or is below the current highest bid, a trade is made" | `submit_ask()`: if `price <= self.bids[0].price` → trade | YES |
| 13 | "If a buyer's bid becomes the current highest bid and it meets or exceeds the current lowest ask, a trade occurs" | `submit_bid()`: if `price >= self.asks[0].price` → trade | YES |
| 14 | Improvement rule: new bid must beat current best bid; new ask must beat current best ask | Spec 2: bid rejected if `price <= self.bids[0].price`; ask rejected if `price >= self.asks[0].price` | YES |
| 15 | Order book visible: current best bid/ask, units available, transaction history | Decision prompt includes all these fields | YES |
| 16 | After trade, order book clears (from Figure A.4 behavior) | Spec 3: `self.bids.clear(); self.asks.clear()` after each trade | YES |

## Agent Behavior Rules

| # | Paper (Appendix C) | Code | Match? |
|---|---|---|---|
| 17 | "You are not obliged to sell/buy" + Cancel button (Figure A.4/A.5) | Spec 13: PASS option in prompt, `_extract_price()` parses pass/cancel/skip | YES |
| 18 | Seller profit = Price - Cost | `seller_profit = trade_price - seller.reservation_value` | YES |
| 19 | Buyer profit = Valuation - Price | `buyer_profit = buyer.reservation_value - trade_price` | YES |
| 20 | Cannot sell below cost (software enforces) | Code rejects ask < cost as constraint violation | YES |
| 21 | Cannot buy above valuation (software enforces) | Code rejects bid > valuation as constraint violation | YES |
| 22 | "We recommend you make at least 1 point of profit" (guidance, not enforced) | Not enforced in code — agents CAN trade at 0 profit, same as human software | YES |

## Information Shown on Screen (Figures A.4, A.5)

| # | Paper Screen Element | Code Prompt Field | Match? |
|---|---|---|---|
| 23 | "Available units for purchase: 19" | `{units_to_buy}` | YES |
| 24 | "Available units for sale: 91" | `{units_to_sell}` | YES |
| 25 | "Units sold/bought: 5" | `{units_traded}` | YES |
| 26 | Current best buy offer / best sell offer | `{best_bid}` / `{best_ask}` | YES |
| 27 | Transaction history (list of trades) | `{transaction_history}` | YES |
| 28 | Your situation (cost/valuation, offers, profits) | `{your_situation}` | YES |
| 29 | Period payoffs | `{period_payoffs}` | YES |
| 30 | Cumulated session payoffs | `{cumulated_payoffs}` | YES |
| 31 | Round: 1/3 | `Trading round: {current_step} of {total_steps}` | YES |

## Intentional Adaptations (LLM vs Human)

| # | Difference | Justification |
|---|---|---|
| A1 | Paper: 120 seconds real-time per period. Code: 30 discrete steps per period. | LLMs cannot operate in real-time. Turn-based is the standard adaptation in LLM-as-agent literature. 30 steps provides sufficient rounds for price discovery. |
| A2 | Paper: 28 human subjects on networked computers, real-time (submit anytime during 120s). Code: 28 LLM agents, sequential turns (random order each step). | LLM API calls are sequential. Random ordering prevents systematic position advantages. In practice, even human CDA is effectively sequential (one trade at a time, improvement rule forces one-at-a-time offers). The real difference is humans can time their entry strategically; LLMs cannot. |

Note: The human screen (Figures A.4/A.5) also shows only the **best** bid and **best** ask — labeled "best buy offer" and "best sell offer" — not the full order book. With the improvement rule + order book clearing after each trade, there is effectively only one standing order per side at any time. So our prompt showing best bid/ask matches exactly what humans see.

## Verdict

**Code matches paper on all 31 checkpoints.** Two intentional adaptations (A1-A2) are necessary for LLM implementation and are standard in the LLM-as-agent experimental literature. Neither affects the competitive equilibrium or the fundamental market structure.
