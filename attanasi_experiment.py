#!/usr/bin/env python3
"""
Attanasi, Centorrino & Moscati (2021) — Perfect Competition Replication
"Controlling monopoly power in a double-auction market experiment"
JPET 23(5), 1074-1101

Phase 1: Perfect Competition with LLM agents.
Compares LLM trading behavior with human results (Table B.1).

Design Specifications Implemented:
  1. Buyer valuation reshuffling each period
  2. Improvement rule enforcement (bid must beat best bid; ask must beat best ask)
  3. Order book clearing after each trade
  4. Correct buyer creation comment
  5. Event-based stopping (idle rounds)
  6. Period vs session payoff separation
  7. Best bid/ask shows agent IDs
  8. Correct price ranges ([0, 30] global per Appendix C; buyers further capped at valuation; sellers floored at cost)
  9. Remove profit recommendation (config)
 10. Remove cost symmetry information (config)
 11. Multi-turn conversation history within each period
 12. Cross-period memory (period summaries carried forward)

Usage:
    python attanasi_experiment.py --model llama3.3 --steps 30 --runs 1 --seed 42
"""

import argparse
import csv
import json
import math
import os
import random
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import socket

import requests
import yaml


def _atomic_json_write(filepath, data):
    """Write JSON atomically: write to temp file, then rename.
    Prevents data loss if process is killed mid-write."""
    filepath = Path(filepath)
    fd, tmp = tempfile.mkstemp(dir=filepath.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, filepath)  # atomic on POSIX
    except BaseException:
        os.unlink(tmp)
        raise


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class Trade:
    period: int
    step: int
    price: int
    buyer_id: str
    seller_id: str
    buyer_valuation: int
    seller_cost: int
    buyer_profit: int
    seller_profit: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class OrderBookEntry:
    price: int
    agent_id: str
    timestamp: float


# =============================================================================
# AGENT CLASS
# =============================================================================


class Agent:
    def __init__(
        self,
        agent_id: str,
        agent_type: str,  # "buyer" or "seller"
        reservation_value: int,  # valuation (buyer) or cost (seller)
        llm_config: dict,
        prompt_templates: dict,
        capacity: int = 1,
        refined_memory: bool = False,
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.reservation_value = reservation_value
        self.llm_config = llm_config
        self.prompt_templates = prompt_templates
        self.capacity = capacity
        self.units_remaining = capacity
        self.trades: List[Trade] = []
        self.period_trades: List[Trade] = []  # trades in current period only
        # Spec 6: separate period and session payoffs
        self.period_payoffs = 0
        self.cumulated_payoffs = 0
        # Spec 11: multi-turn conversation history within each period
        self.message_history: List[dict] = []
        self.pending_feedback: Optional[str] = None
        # Spec 12: cross-period memory (never reset)
        self.period_summaries: List[str] = []
        # Refined memory: track own submitted actions (including rejected) this period
        self.refined_memory = refined_memory
        self.period_actions: List[dict] = []
        self.llm_successes = 0
        self.llm_errors = 0

    def reset_period(self):
        """Reset for a new trading period. Preserves cumulated_payoffs and period_summaries."""
        self.units_remaining = self.capacity
        self.period_trades = []
        self.period_payoffs = 0  # Spec 6: reset each period
        self.message_history = []  # Spec 11: fresh conversation each period
        self.pending_feedback = None
        self.period_actions = []  # refined memory: fresh action log each period
        # NOTE: cumulated_payoffs and period_summaries are NOT reset

    def can_trade(self) -> bool:
        return self.units_remaining > 0

    def decide_price(self, market_state: dict) -> Tuple[Optional[int], str]:
        """Query LLM for a price decision using multi-turn conversation (Spec 11).

        Returns (price, status) where status is one of:
          'ok'                - valid price returned
          'skip'              - agent can't trade (no remaining units)
          'llm_error'         - LLM call failed (timeout, connection, HTTP error)
          'parse_error'       - couldn't extract integer from LLM response
          'constraint_error'  - price violated budget/range constraint
        """
        if not self.can_trade():
            return None, "skip"

        period = market_state.get("period", 1)
        decision = self._build_decision_prompt(market_state)

        # Spec 11: multi-turn conversation
        if not self.message_history:
            # First call in this period: include system prompt
            system = self._build_system_prompt(period)
            user_content = f"{system}\n{decision}"
        else:
            # Subsequent calls: prepend feedback from last action
            if self.pending_feedback:
                user_content = f"{self.pending_feedback}\n\n{decision}"
            else:
                user_content = decision
        self.pending_feedback = None

        self.message_history.append({"role": "user", "content": user_content})
        raw_response = self._query_llm()
        if raw_response is None:
            self.message_history.pop()  # remove failed user message
            return None, "llm_error"
        self.message_history.append({"role": "assistant", "content": raw_response})

        price, extract_status = self._extract_price(raw_response)
        if price is None:
            return None, extract_status

        return price, "ok"

    def record_outcome(self, feedback: str):
        """Store feedback to prepend to the next decision prompt (Spec 11)."""
        self.pending_feedback = feedback

    def save_period_summary(self, period: int, all_trades: List[Trade]):
        """Build a summary for cross-period memory (Spec 12).

        Baseline: one-line terse summary (original behavior).
        Refined (self.refined_memory=True): multi-line detailed recap including
        own action history, market price distribution, and intra-period trajectory.
        """
        if self.refined_memory:
            self.period_summaries.append(
                self._build_refined_summary(period, all_trades)
            )
        else:
            self.period_summaries.append(
                self._build_baseline_summary(period, all_trades)
            )

    def _build_baseline_summary(self, period: int, all_trades: List[Trade]) -> str:
        """Original one-line summary."""
        if self.period_trades:
            if self.agent_type == "buyer":
                t = self.period_trades[-1]
                own = (f"You bought 1 unit at price {t.price}, "
                       f"profit={t.buyer_profit}")
            else:
                n = len(self.period_trades)
                total_profit = sum(t.seller_profit for t in self.period_trades)
                own = f"You sold {n} unit(s), total profit={total_profit}"
        else:
            own = "You did not trade"

        if all_trades:
            avg_p = sum(t.price for t in all_trades) / len(all_trades)
            market = f"{len(all_trades)} trades, avg price={avg_p:.1f}"
        else:
            market = "0 trades"

        return f"Period {period}: {own}. Market: {market}."

    def _build_refined_summary(self, period: int, all_trades: List[Trade]) -> str:
        """Multi-line detailed recap with distribution and trajectory."""
        lines = [f"Period {period} recap:"]

        # --- Your own actions this period ---
        if self.period_actions:
            act_lines = []
            for a in self.period_actions:
                verb = "bid" if self.agent_type == "buyer" else "ask"
                if a["matched"]:
                    act_lines.append(
                        f"{verb} {a['price']} -> TRADED with {a['counterparty']} "
                        f"(profit={a['profit']})"
                    )
                elif a["status"] == "ok":
                    act_lines.append(f"{verb} {a['price']} -> posted, not matched")
                elif a["status"] == "pass":
                    act_lines.append("PASS")
                else:
                    # rejected (constraint / parse / llm error)
                    p_str = a["price"] if a["price"] is not None else "-"
                    act_lines.append(f"{verb} {p_str} -> rejected ({a['status']})")
            lines.append(f"  Your actions ({len(self.period_actions)}): "
                         + "; ".join(act_lines))
        else:
            lines.append("  Your actions: none (no submissions)")

        # --- Own outcome roll-up ---
        if self.period_trades:
            if self.agent_type == "buyer":
                t = self.period_trades[-1]
                lines.append(
                    f"  Your outcome: bought 1 unit at price {t.price}, "
                    f"profit={t.buyer_profit} (valuation={self.reservation_value})"
                )
            else:
                n = len(self.period_trades)
                total = sum(t.seller_profit for t in self.period_trades)
                prices_sold = [t.price for t in self.period_trades]
                lines.append(
                    f"  Your outcome: sold {n} unit(s) at prices {prices_sold}, "
                    f"total profit={total} (cost={self.reservation_value})"
                )
        else:
            lines.append(
                f"  Your outcome: did not trade "
                f"({'valuation' if self.agent_type == 'buyer' else 'cost'}"
                f"={self.reservation_value})"
            )

        # --- Market-wide distribution ---
        if all_trades:
            prices = sorted(t.price for t in all_trades)
            n = len(prices)
            avg_p = sum(prices) / n
            median_p = prices[n // 2] if n % 2 else (prices[n//2 - 1] + prices[n//2]) / 2
            lines.append(
                f"  Market: {n} trades, price range {prices[0]}-{prices[-1]}, "
                f"median={median_p:g}, avg={avg_p:.1f}"
            )
            # Trajectory: sort trades by step and list prices in order
            traj = [t.price for t in sorted(all_trades, key=lambda x: x.step)]
            lines.append(f"  Price trajectory (step order): {traj}")
        else:
            lines.append("  Market: 0 trades this period")

        return "\n".join(lines)

    def _build_system_prompt(self, period: int) -> str:
        """Build system/instruction text (Specs 7, 8, 9, 10, 12)."""
        if self.agent_type == "buyer":
            prompt = self.prompt_templates["buyer_system"].format(
                agent_id=self.agent_id,
                valuation=self.reservation_value,
                period=period,
            )
        else:
            prompt = self.prompt_templates["seller_system"].format(
                agent_id=self.agent_id,
                cost=self.reservation_value,
                period=period,
            )

        # Spec 12: cross-period memory
        if self.period_summaries:
            prompt += "\n\n=== YOUR EXPERIENCE FROM PREVIOUS PERIODS ===\n"
            for s in self.period_summaries:
                prompt += f"  {s}\n"

        return prompt

    def _build_decision_prompt(self, market_state: dict) -> str:
        """Build market screen text mirroring Figures A.4/A.5 (Specs 6, 7)."""
        action_verb = "bid" if self.agent_type == "buyer" else "offer"

        # Transaction history — humans see ALL trades this period
        all_trades = market_state.get("all_trades", [])
        if all_trades:
            tx_lines = []
            for i, t in enumerate(all_trades, 1):
                tx_lines.append(
                    f"  Trade {i}: {t.buyer_id} bought from {t.seller_id} @ {t.price}"
                )
            transaction_history = "\n".join(tx_lines)
        else:
            transaction_history = "  No trades yet"

        # Your situation — mirrors the private table humans see
        if self.agent_type == "buyer":
            if self.period_trades:
                t = self.period_trades[-1]
                your_situation = (
                    f"Unit 1: Value={self.reservation_value}, "
                    f"Your bid={t.price}, Transaction price={t.price}, "
                    f"Payoff={t.buyer_profit}"
                )
            else:
                your_situation = (
                    f"Unit 1: Value={self.reservation_value}, "
                    f"Your bid=-, Transaction price=-, Payoff=-"
                )
        else:
            if self.period_trades:
                sit_lines = []
                for i, t in enumerate(self.period_trades, 1):
                    sit_lines.append(
                        f"Unit {i}: Cost={self.reservation_value}, "
                        f"Your ask={t.price}, Transaction price={t.price}, "
                        f"Payoff={t.seller_profit}"
                    )
                your_situation = "\n".join(sit_lines)
                your_situation += (
                    f"\n  Units sold this period: "
                    f"{len(self.period_trades)}/{self.capacity}"
                )
            else:
                your_situation = (
                    f"Unit 1: Cost={self.reservation_value}, "
                    f"Your ask=-, Transaction price=-, Payoff=-\n"
                    f"  Units sold this period: 0/{self.capacity}"
                )

        # Spec 7: best bid/ask with agent IDs
        best_bid = market_state.get("best_bid")
        best_ask = market_state.get("best_ask")
        best_bid_agent = market_state.get("best_bid_agent")
        best_ask_agent = market_state.get("best_ask_agent")

        if best_bid is not None and best_bid_agent:
            best_bid_display = f"{best_bid_agent}: {best_bid}"
        else:
            best_bid_display = "None"

        if best_ask is not None and best_ask_agent:
            best_ask_display = f"{best_ask_agent}: {best_ask}"
        else:
            best_ask_display = "None"

        decision = self.prompt_templates["decision"].format(
            best_bid=best_bid_display,
            best_ask=best_ask_display,
            units_to_buy=market_state.get("units_to_buy", "?"),
            units_to_sell=market_state.get("units_to_sell", "?"),
            units_traded=market_state.get("units_traded", 0),
            current_step=market_state.get("current_step", "?"),
            total_steps=market_state.get("total_steps", "?"),
            transaction_history=transaction_history,
            your_situation=your_situation,
            period_payoffs=self.period_payoffs,
            cumulated_payoffs=self.cumulated_payoffs,
            action_verb=action_verb,
        )

        return decision

    def _get_messages_for_api(self) -> List[dict]:
        """Truncate conversation history for API call (Spec 11).

        Caps at 20 messages (10 turns), always keeping the first message
        (system context) to preserve instructions.
        """
        if len(self.message_history) <= 20:
            return list(self.message_history)
        # Keep first message + last 19
        return [self.message_history[0]] + self.message_history[-19:]

    def _query_llm(self) -> Optional[str]:
        """POST to Binghamton API using full conversation history (Spec 11)."""
        try:
            headers = {"Content-Type": "application/json"}
            if self.llm_config.get("api_key"):
                headers["Authorization"] = f"Bearer {self.llm_config['api_key']}"

            messages = self._get_messages_for_api()

            response = requests.post(
                self.llm_config["url"],
                headers=headers,
                json={
                    "model": self.llm_config["model"],
                    "messages": messages,
                    "temperature": self.llm_config.get("temperature", 0.7),
                    "max_tokens": self.llm_config.get("max_tokens", 100),
                },
                timeout=self.llm_config.get("timeout", 60),
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            self.llm_successes += 1
            return content

        except requests.exceptions.Timeout:
            print(
                f"  ERROR {self.agent_id}: TIMEOUT "
                f"(>{self.llm_config.get('timeout', 60)}s)"
            )
            self.llm_errors += 1
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"  ERROR {self.agent_id}: CONNECTION FAILED - {e}")
            self.llm_errors += 1
            return None
        except requests.exceptions.HTTPError as e:
            print(
                f"  ERROR {self.agent_id}: HTTP {e.response.status_code} "
                f"- {e.response.text[:200]}"
            )
            self.llm_errors += 1
            return None
        except Exception as e:
            print(f"  ERROR {self.agent_id}: {type(e).__name__}: {e}")
            self.llm_errors += 1
            return None

    def _extract_price(self, text: str) -> Tuple[Optional[int], str]:
        """Extract integer price from LLM response, validate constraints (Spec 8).

        Returns (price, status) where status is 'ok', 'parse_error', or 'constraint_error'.
        Spec 8: No artificial [0,30] range — only budget constraints enforced.
        """
        if not text or not text.strip():
            return None, "parse_error"

        # Check for PASS/CANCEL (mirrors Cancel button on experiment screen)
        cleaned_check = text.strip().lower()
        if any(w in cleaned_check for w in ("pass", "cancel", "skip", "no bid", "no ask")):
            return None, "pass"

        # Clean text
        cleaned = re.sub(r"[\$,]", "", text.strip().lower())
        cleaned = re.sub(r"\s*(dollars?|usd|points?)\s*", "", cleaned)

        # Find first integer
        match = re.search(r"\b(\d+)\b", cleaned)
        if not match:
            print(
                f"  ERROR {self.agent_id}: Could not parse integer "
                f"(got: '{text[:50]}')"
            )
            return None, "parse_error"

        price = int(match.group(1))

        # Price range [0, 30] per paper instructions (Appendix C):
        # "You can make offers, either to sell or buy, only using integer
        #  numbers within 0 and 30."
        if price < 0 or price > 30:
            print(f"  ERROR {self.agent_id}: Price {price} outside [0, 30]")
            return None, "constraint_error"

        # Budget constraints (software-enforced per paper):
        # "Negative profits are not allowed"
        if self.agent_type == "buyer" and price > self.reservation_value:
            print(
                f"  ERROR {self.agent_id}: Bid {price} > valuation "
                f"{self.reservation_value}"
            )
            return None, "constraint_error"
        if self.agent_type == "seller" and price < self.reservation_value:
            print(
                f"  ERROR {self.agent_id}: Ask {price} < cost "
                f"{self.reservation_value}"
            )
            return None, "constraint_error"

        return price, "ok"


# =============================================================================
# DOUBLE AUCTION CLASS
# =============================================================================


class DoubleAuction:
    """Continuous double auction with price-time priority matching.

    Implements:
      - Spec 2: Improvement rule (bids must beat best bid; asks must beat best ask)
      - Spec 3: Order book clearing after each trade
      - Spec 7: Agent IDs in market state
    """

    def __init__(self):
        self.bids: List[OrderBookEntry] = []  # buy orders (highest first)
        self.asks: List[OrderBookEntry] = []  # sell orders (lowest first)
        self.trades: List[Trade] = []
        self.all_period_trades: List[Trade] = []

    def reset_period(self):
        """Clear the order book for a new period.

        trades (current-period list) and all_period_trades (cumulative) are both
        cleared. The period filter on all_period_trades (L986) makes the cumulative
        list redundant across periods anyway; clearing it here is defense-in-depth
        against stale data if period tagging ever drifts.
        """
        self.bids.clear()
        self.asks.clear()
        self.trades.clear()
        self.all_period_trades.clear()

    def best_bid(self) -> Optional[int]:
        return self.bids[0].price if self.bids else None

    def best_ask(self) -> Optional[int]:
        return self.asks[0].price if self.asks else None

    def recent_prices(self, n: int = 10) -> List[int]:
        return [t.price for t in self.trades[-n:]]

    def market_state(self, agents: Dict[str, "Agent"] = None, period: int = 1,
                     current_step: int = 1, total_steps: int = 30) -> dict:
        """Build market state dict. Spec 7: includes agent IDs for best bid/ask."""
        units_to_buy = 0
        units_to_sell = 0
        if agents:
            for a in agents.values():
                if a.agent_type == "buyer":
                    units_to_buy += a.units_remaining
                else:
                    units_to_sell += a.units_remaining

        return {
            "best_bid": self.best_bid(),
            "best_ask": self.best_ask(),
            "best_bid_agent": self.bids[0].agent_id if self.bids else None,
            "best_ask_agent": self.asks[0].agent_id if self.asks else None,
            "all_trades": list(self.trades),
            "units_to_buy": units_to_buy,
            "units_to_sell": units_to_sell,
            "units_traded": len(self.trades),
            "period": period,
            "current_step": current_step,
            "total_steps": total_steps,
        }

    def submit_bid(
        self, price: int, agent_id: str, agents: Dict[str, "Agent"],
        period: int, step: int,
    ) -> Tuple[Optional[Trade], str]:
        """Submit a buy order. Returns (trade_or_None, status).

        status is 'traded', 'posted', or 'rejected'.
        Spec 2: Improvement rule — bid must strictly exceed current best bid.
        """
        now = time.time()

        # Spec 2: Improvement rule
        if self.bids and price <= self.bids[0].price:
            print(
                f"  {agent_id} ({agents[agent_id].agent_type}): "
                f"bid {price} REJECTED (does not improve on standing bid)"
            )
            return None, "rejected"

        # Check for match: bid >= lowest ask
        if self.asks and price >= self.asks[0].price:
            ask_entry = self.asks.pop(0)
            trade_price = ask_entry.price  # Price priority: earlier order sets price
            trade = self._execute_trade(
                trade_price, agent_id, ask_entry.agent_id, agents, period, step
            )
            return trade, "traded"

        # No match — add to order book (sorted: highest bid first, then time)
        entry = OrderBookEntry(price=price, agent_id=agent_id, timestamp=now)
        inserted = False
        for i, existing in enumerate(self.bids):
            if price > existing.price:
                self.bids.insert(i, entry)
                inserted = True
                break
        if not inserted:
            self.bids.append(entry)
        return None, "posted"

    def submit_ask(
        self, price: int, agent_id: str, agents: Dict[str, "Agent"],
        period: int, step: int,
    ) -> Tuple[Optional[Trade], str]:
        """Submit a sell order. Returns (trade_or_None, status).

        status is 'traded', 'posted', or 'rejected'.
        Spec 2: Improvement rule — ask must be strictly below current best ask.
        """
        now = time.time()

        # Spec 2: Improvement rule
        if self.asks and price >= self.asks[0].price:
            print(
                f"  {agent_id} ({agents[agent_id].agent_type}): "
                f"ask {price} REJECTED (does not improve on standing ask)"
            )
            return None, "rejected"

        # Check for match: ask <= highest bid
        if self.bids and price <= self.bids[0].price:
            bid_entry = self.bids.pop(0)
            trade_price = bid_entry.price  # Price priority: earlier order sets price
            trade = self._execute_trade(
                trade_price, bid_entry.agent_id, agent_id, agents, period, step
            )
            return trade, "traded"

        # No match — add to order book (sorted: lowest ask first, then time)
        entry = OrderBookEntry(price=price, agent_id=agent_id, timestamp=now)
        inserted = False
        for i, existing in enumerate(self.asks):
            if price < existing.price:
                self.asks.insert(i, entry)
                inserted = True
                break
        if not inserted:
            self.asks.append(entry)
        return None, "posted"

    def _execute_trade(
        self, price: int, buyer_id: str, seller_id: str,
        agents: Dict[str, "Agent"], period: int, step: int,
    ) -> Trade:
        """Record a trade and update agent state.

        Spec 3: Clears all standing bids and asks after each trade.
        Spec 6: Updates both period_payoffs and cumulated_payoffs.
        """
        buyer = agents[buyer_id]
        seller = agents[seller_id]

        trade = Trade(
            period=period,
            step=step,
            price=price,
            buyer_id=buyer_id,
            seller_id=seller_id,
            buyer_valuation=buyer.reservation_value,
            seller_cost=seller.reservation_value,
            buyer_profit=buyer.reservation_value - price,
            seller_profit=price - seller.reservation_value,
        )

        buyer.units_remaining -= 1
        seller.units_remaining -= 1
        buyer.trades.append(trade)
        seller.trades.append(trade)
        buyer.period_trades.append(trade)
        seller.period_trades.append(trade)
        # Spec 6: update both period and session payoffs
        buyer.period_payoffs += trade.buyer_profit
        seller.period_payoffs += trade.seller_profit
        buyer.cumulated_payoffs += trade.buyer_profit
        seller.cumulated_payoffs += trade.seller_profit
        self.trades.append(trade)
        self.all_period_trades.append(trade)

        # Spec 3: clear ALL standing orders after each trade
        self.bids.clear()
        self.asks.clear()

        print(
            f"  *** TRADE: {buyer_id} buys from {seller_id} @ {price} "
            f"(buyer profit: {trade.buyer_profit}, "
            f"seller profit: {trade.seller_profit})"
        )
        return trade


# =============================================================================
# EXPERIMENT CLASS
# =============================================================================


class CompetitionExperiment:
    """Orchestrates the 3-period perfect competition experiment."""

    HUMAN_BENCHMARKS = [14.49, 14.10, 13.50]

    def __init__(self, config: dict, model_key: str, num_steps: int, seed: int,
                 refined_memory: bool = False):
        self.config = config
        self.exp = config["experiment"]
        self.num_steps = num_steps
        self.seed = seed
        self.rng = random.Random(seed)
        self.refined_memory = refined_memory

        # Get LLM config for chosen model
        self.llm_config = config["llm_models"][model_key]
        self.model_key = model_key
        self.prompt_templates = config["prompts"]

        # Create agents
        self.agents: Dict[str, Agent] = {}
        self._create_agents()

        # Auction engine
        self.auction = DoubleAuction()

        # Results
        self.period_results: List[dict] = []
        self.all_decisions: List[dict] = []  # every bid/ask, traded or not

    def _create_agents(self):
        """Create 4 sellers + 24 buyers per experimental design."""
        cost = self.exp["seller_cost"]
        seller_cap = self.exp["seller_capacity"]

        # 4 sellers
        for i in range(self.exp["num_sellers"]):
            aid = f"S{i+1}"
            self.agents[aid] = Agent(
                agent_id=aid,
                agent_type="seller",
                reservation_value=cost,
                llm_config=self.llm_config,
                prompt_templates=self.prompt_templates,
                capacity=seller_cap,
                refined_memory=self.refined_memory,
            )

        # Spec 4: 24 buyers: 4 at each of 6 valuations (reshuffled each period)
        valuations = self.exp["buyer_valuations"]
        buyer_idx = 1
        for g in range(self.exp["num_groups"]):
            for v in valuations:
                aid = f"B{buyer_idx}"
                self.agents[aid] = Agent(
                    agent_id=aid,
                    agent_type="buyer",
                    reservation_value=v,
                    llm_config=self.llm_config,
                    prompt_templates=self.prompt_templates,
                    capacity=self.exp["buyer_capacity"],
                    refined_memory=self.refined_memory,
                )
                buyer_idx += 1

    def _reshuffle_buyer_valuations(self):
        """Spec 1: Reshuffle the 24-value multiset and reassign to buyers each period."""
        valuations = []
        for v in self.exp["buyer_valuations"]:
            valuations.extend([v] * self.exp["num_groups"])  # 4 copies of each
        self.rng.shuffle(valuations)
        buyers = [a for a in self.agents.values() if a.agent_type == "buyer"]
        for buyer, val in zip(buyers, valuations):
            buyer.reservation_value = val

    def run(self) -> List[dict]:
        """Run all 3 periods."""
        num_periods = self.exp["num_periods"]
        print(f"\n{'='*60}")
        print(f"Attanasi et al. (2021) Perfect Competition Replication")
        print(f"Model: {self.llm_config.get('name', self.model_key)}")
        print(f"Periods: {num_periods}, Steps/period: {self.num_steps}, Seed: {self.seed}")
        print(f"Agents: {self.exp['num_sellers']} sellers (cost={self.exp['seller_cost']}) "
              f"+ {self.exp['num_buyers']} buyers")
        print(f"{'='*60}")

        for period in range(1, num_periods + 1):
            self._run_period(period)

        self._print_summary()
        return self.period_results

    def _run_period(self, period: int):
        """Run one trading period with all design specs enforced."""
        print(f"\n--- Period {period} ---")

        # Spec 1: reshuffle buyer valuations each period
        self._reshuffle_buyer_valuations()

        # Reset agents and auction
        for agent in self.agents.values():
            agent.reset_period()
        self.auction.reset_period()

        # Get list of active agent IDs
        agent_ids = list(self.agents.keys())
        trades_this_period = 0
        period_llm_errors = 0
        period_constraint_errors = 0
        period_parse_errors = 0
        period_llm_calls = 0
        period_rejected = 0

        # Spec 5: event-based stopping
        idle_limit = self.exp.get("idle_rounds_to_stop", 5)
        no_trade_rounds = 0

        for step in range(1, self.num_steps + 1):
            # Randomize order each step
            self.rng.shuffle(agent_ids)
            step_had_trade = False

            for aid in agent_ids:
                agent = self.agents[aid]
                if not agent.can_trade():
                    continue

                # Get price decision from LLM
                period_llm_calls += 1
                price, status = agent.decide_price(
                    self.auction.market_state(self.agents, period, step, self.num_steps)
                )

                if status == "llm_error":
                    period_llm_errors += 1
                    agent.record_outcome(
                        "Your request failed due to a system error. Please try again."
                    )
                    continue
                elif status == "constraint_error":
                    period_constraint_errors += 1
                    agent.record_outcome(
                        "Your price was rejected because it violated your budget constraint."
                    )
                    continue
                elif status == "parse_error":
                    period_parse_errors += 1
                    agent.record_outcome(
                        "Could not understand your response. Please reply with a single integer."
                    )
                    continue
                elif status == "pass":
                    agent.record_outcome(
                        "You chose to PASS this round (no bid/ask submitted)."
                    )
                    continue
                elif price is None:
                    continue

                # Submit to auction
                if agent.agent_type == "buyer":
                    action = "bid"
                    trade, result = self.auction.submit_bid(
                        price, aid, self.agents, period, step
                    )
                else:
                    action = "ask"
                    trade, result = self.auction.submit_ask(
                        price, aid, self.agents, period, step
                    )

                # Log every decision
                decision_log = {
                    "period": period,
                    "step": step,
                    "agent_id": aid,
                    "agent_type": agent.agent_type,
                    "action": action,
                    "submitted_price": price,
                    "reservation_value": agent.reservation_value,
                    "matched": trade is not None,
                    "trade_price": trade.price if trade else "",
                    "counterparty": (
                        (trade.seller_id if action == "bid" else trade.buyer_id)
                        if trade else ""
                    ),
                    "profit": (
                        (trade.buyer_profit if action == "bid" else trade.seller_profit)
                        if trade else ""
                    ),
                }
                self.all_decisions.append(decision_log)

                # Refined memory: per-agent action log for cross-period summary
                if agent.refined_memory:
                    agent.period_actions.append({
                        "step": step,
                        "price": price,
                        "status": "ok",
                        "matched": trade is not None,
                        "counterparty": (
                            (trade.seller_id if action == "bid" else trade.buyer_id)
                            if trade else ""
                        ),
                        "profit": (
                            (trade.buyer_profit if action == "bid" else trade.seller_profit)
                            if trade else ""
                        ),
                    })

                # Spec 11: feedback for multi-turn conversation
                if result == "traded":
                    trades_this_period += 1
                    step_had_trade = True
                    if agent.agent_type == "buyer":
                        agent.record_outcome(
                            f"TRADED! You bought 1 unit at price {trade.price}. "
                            f"Your profit: {trade.buyer_profit}."
                        )
                        # Feedback to counterparty seller
                        self.agents[trade.seller_id].record_outcome(
                            f"TRADED! You sold 1 unit at price {trade.price}. "
                            f"Your profit: {trade.seller_profit}."
                        )
                    else:
                        agent.record_outcome(
                            f"TRADED! You sold 1 unit at price {trade.price}. "
                            f"Your profit: {trade.seller_profit}."
                        )
                        # Feedback to counterparty buyer
                        self.agents[trade.buyer_id].record_outcome(
                            f"TRADED! You bought 1 unit at price {trade.price}. "
                            f"Your profit: {trade.buyer_profit}."
                        )
                elif result == "rejected":
                    period_rejected += 1
                    agent.record_outcome(
                        f"Your {action} of {price} was REJECTED because it does not "
                        f"improve on the current best {action}."
                    )
                elif result == "posted":
                    agent.record_outcome(
                        f"Your {action} of {price} has been POSTED to the order book."
                    )
                    # Print posted orders (not rejected, not traded)
                    model = self.llm_config.get("model", "?")
                    print(
                        f"  {aid} ({agent.agent_type}): [{model}] "
                        f"{action} {price} (limit: {agent.reservation_value})"
                    )

            # Spec 5: event-based stopping
            if step_had_trade:
                no_trade_rounds = 0
            else:
                no_trade_rounds += 1

            if no_trade_rounds >= idle_limit:
                print(
                    f"  [No trades for {idle_limit} consecutive rounds "
                    f"-- stopping period at step {step}]"
                )
                break

            # Check if all possible trades are done
            active_buyers = [
                a for a in self.agents.values()
                if a.agent_type == "buyer" and a.can_trade()
                and a.reservation_value > self.exp["seller_cost"]
            ]
            active_sellers = [
                a for a in self.agents.values()
                if a.agent_type == "seller" and a.can_trade()
            ]
            if not active_buyers or not active_sellers:
                print(f"  [All profitable trades exhausted at step {step}]")
                break

        # Spec 12: save period summaries for cross-period memory
        period_trades = [
            t for t in self.auction.all_period_trades if t.period == period
        ]
        for agent in self.agents.values():
            agent.save_period_summary(period, period_trades)

        # Period summary
        if period_trades:
            avg_price = sum(t.price for t in period_trades) / len(period_trades)
            min_price = min(t.price for t in period_trades)
            max_price = max(t.price for t in period_trades)
        else:
            avg_price = min_price = max_price = float("nan")

        result = {
            "period": period,
            "num_trades": len(period_trades),
            "avg_price": avg_price,
            "min_price": min_price,
            "max_price": max_price,
            "human_avg": self.HUMAN_BENCHMARKS[period - 1],
            "llm_calls": period_llm_calls,
            "llm_errors": period_llm_errors,
            "constraint_errors": period_constraint_errors,
            "parse_errors": period_parse_errors,
            "rejected": period_rejected,
        }
        self.period_results.append(result)

        print(
            f"\n  Period {period} summary: {len(period_trades)} trades, "
            f"avg price = {avg_price:.2f} "
            f"(human: {self.HUMAN_BENCHMARKS[period-1]:.2f}) "
            f"[LLM calls: {period_llm_calls}, errors: {period_llm_errors}, "
            f"constraint: {period_constraint_errors}, rejected: {period_rejected}]"
        )

    def _print_summary(self):
        """Print comparison table: LLM vs Human."""
        print(f"\n{'='*60}")
        print("RESULTS COMPARISON")
        print(f"{'='*60}")
        print(f"{'Period':<10} {'LLM Avg':>10} {'Human Avg':>12} {'CE':>8} {'N Trades':>10}")
        print(f"{'-'*50}")
        for r in self.period_results:
            print(
                f"{r['period']:<10} {r['avg_price']:>10.2f} "
                f"{r['human_avg']:>12.2f} {'~12':>8} {r['num_trades']:>10}"
            )
        print(f"{'='*60}")

        # LLM error summary
        total_success = sum(a.llm_successes for a in self.agents.values())
        total_errors = sum(a.llm_errors for a in self.agents.values())
        total_calls = total_success + total_errors
        if total_calls > 0:
            print(
                f"LLM calls: {total_calls} total, "
                f"{total_success} success ({100*total_success/total_calls:.1f}%), "
                f"{total_errors} errors"
            )

    DIAL_LEVEL_MAP = {
        "risk_aversion": {"none": 0, "very_averse": 1, "averse": 2, "neutral": 3, "seeking": 4, "very_seeking": 5},
        "aggressiveness": {"none": 0, "very_passive": 1, "passive": 2, "moderate": 3, "aggressive": 4, "very_aggressive": 5},
        "profit_orientation": {"none": 0, "very_conservative": 1, "conservative": 2, "balanced": 3, "maximizer": 4, "extreme_maximizer": 5},
    }

    def get_decision_rows(self, run_id: int, run_seed: int, batch_timestamp: str,
                          persona: str = None,
                          memory_type: str = "baseline",
                          dial_values: dict = None,
                          temperature: float = None) -> List[dict]:
        """Return all decision rows (every bid/ask) with run metadata."""
        dv = dial_values or {}
        temp = temperature if temperature is not None else self.llm_config.get("temperature", 0.7)
        rows = []
        for d in self.all_decisions:
            ra = dv.get("risk_aversion", "none")
            ag = dv.get("aggressiveness", "none")
            po = dv.get("profit_orientation", "none")
            rows.append({
                "run_id": run_id,
                "seed": run_seed,
                "model": self.model_key,
                "persona": persona or "baseline",
                "memory_type": memory_type,
                "temperature": temp,
                "risk_aversion": ra,
                "risk_aversion_level": self.DIAL_LEVEL_MAP["risk_aversion"][ra],
                "aggressiveness": ag,
                "aggressiveness_level": self.DIAL_LEVEL_MAP["aggressiveness"][ag],
                "profit_orientation": po,
                "profit_orientation_level": self.DIAL_LEVEL_MAP["profit_orientation"][po],
                "timestamp": batch_timestamp,
                **d,
            })
        return rows


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Attanasi et al. (2021) Perfect Competition Replication"
    )
    parser.add_argument(
        "--model", type=str, default="llama3.3",
        help="LLM model key from config (default: llama3.3)",
    )
    parser.add_argument(
        "--steps", type=int, default=30,
        help="Trading rounds per period (default: 30)",
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of independent runs (default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume a previous batch -- skip completed runs, append to existing CSVs",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (default: attanasi_config.yaml in same dir)",
    )
    parser.add_argument(
        "--persona", type=str, default=None,
        help="Persona key from config personas section (e.g. undergrad_econ). "
             "Omit for clean baseline replication.",
    )
    parser.add_argument(
        "--refined-memory", action="store_true",
        help="Use refined (detailed) cross-period memory instead of the "
             "baseline one-line summary. Includes own action log, market "
             "price distribution, and intra-period price trajectory.",
    )
    parser.add_argument(
        "--risk-aversion", type=str, default=None,
        choices=["very_averse", "averse", "neutral", "seeking", "very_seeking"],
        help="Behavioral dial: risk aversion level (default: none)",
    )
    parser.add_argument(
        "--aggressiveness", type=str, default=None,
        choices=["very_passive", "passive", "moderate", "aggressive", "very_aggressive"],
        help="Behavioral dial: aggressiveness level (default: none)",
    )
    parser.add_argument(
        "--profit-orientation", type=str, default=None,
        choices=["very_conservative", "conservative", "balanced", "maximizer", "extreme_maximizer"],
        help="Behavioral dial: profit orientation level (default: none)",
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Override LLM temperature (0.0-1.0). If not set, uses config default.",
    )
    args = parser.parse_args()

    # Load config
    script_dir = Path(__file__).parent
    config_path = Path(args.config) if args.config else script_dir / "attanasi_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate model
    if args.model not in config["llm_models"]:
        available = ", ".join(config["llm_models"].keys())
        print(f"ERROR: Unknown model '{args.model}'. Available: {available}")
        return

    # On apape1/apape2, route inference to local Ollama instead of Binghamton server.
    # apape1.rc.binghamton.edu == promaxgb10-4ae4, apape2.rc.binghamton.edu == promaxgb10-4be9
    _LOCAL_HOSTS = {
        "apape1.rc.binghamton.edu", "apape2.rc.binghamton.edu",
        "promaxgb10-4ae4", "promaxgb10-4be9",
    }
    _hostname = socket.gethostname()
    if _hostname in _LOCAL_HOSTS or socket.getfqdn() in _LOCAL_HOSTS:
        _local_url = "http://localhost:11434/v1/chat/completions"
        config["llm_models"][args.model]["url"] = _local_url
        config["llm_models"][args.model].pop("api_key", None)
        print(f"Inference: local Ollama on {_hostname} ({_local_url})")
    else:
        print(f"Inference: {config['llm_models'][args.model]['url']}")

    # Override temperature if specified
    if args.temperature is not None:
        if not 0.0 <= args.temperature <= 1.0:
            print(f"ERROR: Temperature must be between 0.0 and 1.0 (got {args.temperature})")
            return
        config["llm_models"][args.model]["temperature"] = args.temperature
        print(f"Temperature: {args.temperature}")
    else:
        print(f"Temperature: {config['llm_models'][args.model].get('temperature', 0.7)} (config default)")

    # Validate and inject persona (robustness treatment)
    if args.persona:
        personas = config.get("personas", {})
        if args.persona not in personas:
            available = ", ".join(personas.keys()) if personas else "(none defined)"
            print(f"ERROR: Unknown persona '{args.persona}'. Available: {available}")
            return
        persona_text = personas[args.persona].strip()
        print(f"Persona: {args.persona}")
    else:
        persona_text = None
        print("Persona: none (baseline replication)")

    # Validate and collect behavioral dial texts
    dials_config = config.get("behavioral_dials", {})
    dial_values = {
        "risk_aversion": args.risk_aversion,
        "aggressiveness": args.aggressiveness,
        "profit_orientation": args.profit_orientation,
    }
    dial_texts = []
    for dial_name, dial_level in dial_values.items():
        if dial_level:
            if dial_name not in dials_config:
                print(f"ERROR: Behavioral dial '{dial_name}' not found in config.")
                return
            if dial_level not in dials_config[dial_name]:
                available = ", ".join(dials_config[dial_name].keys())
                print(f"ERROR: Unknown level '{dial_level}' for dial '{dial_name}'. "
                      f"Available: {available}")
                return
            dial_texts.append(dials_config[dial_name][dial_level].strip())
            print(f"Dial {dial_name}: {dial_level}")

    # Build unified prefix and prepend to system prompts
    prefix_parts = []
    if persona_text:
        prefix_parts.append(persona_text)
    prefix_parts.extend(dial_texts)

    if prefix_parts:
        prefix = "\n\n".join(prefix_parts)
        for key in ("buyer_system", "seller_system"):
            config["prompts"][key] = prefix + "\n\n" + config["prompts"][key]

    if args.refined_memory:
        print("Memory: refined (detailed cross-period summaries)")
    else:
        print("Memory: baseline (one-line cross-period summaries)")

    # Output directory
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # --- Progress tracking ---
    # Progress file stores: batch_timestamp, model, completed seeds, file paths.
    # Refined-memory runs get their own namespace so they don't collide with baseline.
    # Dial suffix ensures parallel factorial combos don't collide.
    mem_suffix = "_refined" if args.refined_memory else ""
    dial_parts = []
    if args.risk_aversion:
        dial_parts.append(f"r{args.risk_aversion[:2]}")
    if args.aggressiveness:
        dial_parts.append(f"a{args.aggressiveness[:2]}")
    if args.profit_orientation:
        dial_parts.append(f"p{args.profit_orientation[:2]}")
    dial_suffix = "_d" + "".join(dial_parts) if dial_parts else ""
    temp_suffix = f"_t{args.temperature:.1f}".replace(".", "") if args.temperature is not None else ""
    progress_file = output_dir / f"attanasi_progress_{args.model}{mem_suffix}{dial_suffix}{temp_suffix}.json"

    master_fields = [
        "run_id", "seed", "model", "persona", "memory_type",
        "temperature",
        "risk_aversion", "risk_aversion_level",
        "aggressiveness", "aggressiveness_level",
        "profit_orientation", "profit_orientation_level",
        "timestamp",
        "period", "step", "agent_id", "agent_type", "action",
        "submitted_price", "reservation_value",
        "matched", "trade_price", "counterparty", "profit",
    ]

    if args.resume and progress_file.exists():
        # Resume existing batch
        with open(progress_file) as f:
            progress = json.load(f)
        batch_timestamp = progress["batch_timestamp"]
        completed_seeds = set(progress["completed_seeds"])
        master_path = Path(progress["master_path"])
        # The stored master_path is absolute and may come from a checkpoint
        # committed on a different machine (e.g. a laptop path resumed on a
        # server). If it doesn't resolve here, relocate by basename to the
        # local output dir so --resume works cross-machine instead of crashing
        # with FileNotFoundError when appending rows.
        if not master_path.exists():
            relocated = output_dir / master_path.name
            if relocated.exists():
                print(f"  Stored master_path not found here; using local copy "
                      f"{relocated}")
                master_path = relocated
            else:
                # Neither the stored path nor a local copy exists, so the
                # completed-seed rows are gone. Reset progress and recreate a
                # fresh master CSV rather than crash or silently skip seeds
                # whose data no longer exists.
                print(f"  WARNING: master CSV missing at stored path and "
                      f"locally; resetting progress and recreating {relocated}")
                master_path = relocated
                completed_seeds = set()
                with open(master_path, "w", newline="") as f:
                    csv.DictWriter(f, fieldnames=master_fields).writeheader()
        print(f"Resuming batch {batch_timestamp}: "
              f"{len(completed_seeds)}/{args.runs} runs already completed")
    else:
        # New batch
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        completed_seeds = set()
        master_path = output_dir / f"attanasi_master_{args.model}{mem_suffix}{dial_suffix}{temp_suffix}_{batch_timestamp}.csv"

        # Write CSV header
        with open(master_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=master_fields).writeheader()

        # Save initial progress
        progress = {
            "batch_timestamp": batch_timestamp,
            "model": args.model,
            "persona": args.persona,
            "completed_seeds": [],
            "master_path": str(master_path),
            "total_runs": args.runs,
            "base_seed": args.seed,
        }
        _atomic_json_write(progress_file, progress)

    # --- Run loop ---
    all_run_results = []
    new_completed = 0

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        run_id = run_idx + 1

        # Skip already-completed runs
        if run_seed in completed_seeds:
            print(f"\n  Skipping run {run_id}/{args.runs} (seed={run_seed}) -- already completed")
            continue

        if args.runs > 1:
            remaining = args.runs - len(completed_seeds) - new_completed
            print(f"\n{'#'*60}")
            print(f"# RUN {run_id}/{args.runs} (seed={run_seed}) -- {remaining} remaining")
            print(f"{'#'*60}")

        experiment = CompetitionExperiment(
            config=config,
            model_key=args.model,
            num_steps=args.steps,
            seed=run_seed,
            refined_memory=args.refined_memory,
        )
        experiment.run()

        # Check if run was successful (had any trades)
        total_trades = sum(r["num_trades"] for r in experiment.period_results)
        total_errors = sum(r.get("llm_errors", 0) for r in experiment.period_results)

        if total_trades == 0 and total_errors > 0:
            print(f"\n  *** Run {run_id} FAILED (0 trades, {total_errors} LLM errors). "
                  f"Skipping -- will retry on --resume. ***")
            continue

        # --- Save immediately (append to master CSV) ---
        persona_tag = args.persona or "baseline"
        mem_type = "refined" if args.refined_memory else "baseline"
        dial_vals_for_csv = {
            "risk_aversion": args.risk_aversion or "none",
            "aggressiveness": args.aggressiveness or "none",
            "profit_orientation": args.profit_orientation or "none",
        }
        decision_rows = experiment.get_decision_rows(
            run_id, run_seed, batch_timestamp, persona_tag,
            memory_type=mem_type, dial_values=dial_vals_for_csv,
            temperature=args.temperature,
        )

        with open(master_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=master_fields)
            writer.writerows(decision_rows)

        # Update progress file
        completed_seeds.add(run_seed)
        new_completed += 1
        progress["completed_seeds"] = sorted(completed_seeds)
        _atomic_json_write(progress_file, progress)

        all_run_results.append(experiment.period_results)

        print(f"\n  Run {run_id} saved. "
              f"Total completed: {len(completed_seeds)}/{args.runs}")

    # --- Final summary ---
    print(f"\nMaster CSV: {master_path}")
    print(f"Completed: {len(completed_seeds)}/{args.runs} runs")

    if len(completed_seeds) < args.runs:
        print(f"\n  {args.runs - len(completed_seeds)} runs remaining. "
              f"Re-run with --resume to continue.")

    # Aggregate across all successful runs
    if len(completed_seeds) > 1:
        all_run_results = _load_results_from_master(master_path)
        # Tag the bootstrap summary with the FULL condition (dial/temp/persona),
        # mirroring the master/progress filenames. Previously only mem_suffix was
        # used, so every persona/dial/temperature run shared one filename
        # (attanasi_bootstrap_<model>.csv) and silently overwrote the others.
        persona_suffix = f"_{args.persona}" if args.persona else ""
        _print_aggregate(
            all_run_results,
            args.model + mem_suffix + dial_suffix + temp_suffix + persona_suffix,
            output_dir,
        )

    # Clean up progress file when all runs complete
    if len(completed_seeds) >= args.runs:
        progress_file.unlink(missing_ok=True)
        print("All runs complete -- progress file removed.")


def _load_results_from_master(master_path: Path) -> list:
    """Reload per-run period results from the master CSV."""
    from collections import defaultdict
    runs = defaultdict(lambda: defaultdict(lambda: {"prices": [], "num_trades": 0}))
    human_benchmarks = {1: 14.49, 2: 14.10, 3: 13.50}

    with open(master_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_id = int(row["run_id"])
            period = int(row["period"])
            if row["matched"] == "True":
                runs[run_id][period]["num_trades"] += 1
                runs[run_id][period]["prices"].append(float(row["trade_price"]))

    result = []
    for run_id in sorted(runs):
        periods = []
        for p in sorted(runs[run_id]):
            info = runs[run_id][p]
            avg_p = sum(info["prices"]) / len(info["prices"]) if info["prices"] else float("nan")
            periods.append({
                "period": p,
                "num_trades": info["num_trades"],
                "avg_price": avg_p,
                "human_avg": human_benchmarks.get(p, 0),
            })
        result.append(periods)
    return result


def _bootstrap_ci(data: list, n_boot: int = 10000, alpha: float = 0.05, seed: int = 999) -> tuple:
    """Compute bootstrap percentile CI. Returns (ci_lo, ci_hi)."""
    rng = random.Random(seed)
    n = len(data)
    boot_means = []
    for _ in range(n_boot):
        sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo_idx = int(n_boot * alpha / 2)
    hi_idx = int(n_boot * (1 - alpha / 2))
    return boot_means[lo_idx], boot_means[hi_idx]


def _print_aggregate(all_run_results: list, model_key: str, output_dir: Path):
    """Compute and print cross-run statistics with bootstrap 95% CIs, save to CSV."""
    num_runs = len(all_run_results)
    num_periods = len(all_run_results[0])
    human_benchmarks = [14.49, 14.10, 13.50]

    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS ({num_runs} runs, bootstrap 95% CI)")
    print(f"{'='*70}")
    print(
        f"{'Period':<8} {'Mean Price':>11} {'SD':>8} {'95% CI':>18} "
        f"{'Human':>8} {'Mean Trades':>12}"
    )
    print(f"{'-'*65}")

    agg_rows = []
    for p in range(num_periods):
        prices = [
            run[p]["avg_price"]
            for run in all_run_results
            if len(run) > p and not math.isnan(run[p]["avg_price"])
        ]
        trades = [run[p]["num_trades"] for run in all_run_results if len(run) > p]

        n = len(prices)
        if n == 0:
            print(f"{p+1:<8} {'NO TRADES':>11}")
            continue

        mean_p = sum(prices) / n
        mean_t = sum(trades) / len(trades)

        if n > 1:
            var = sum((x - mean_p) ** 2 for x in prices) / (n - 1)
            sd = math.sqrt(var)
            ci_lo, ci_hi = _bootstrap_ci(prices)
        else:
            sd = 0.0
            ci_lo = ci_hi = mean_p

        ci_str = f"[{ci_lo:.2f}, {ci_hi:.2f}]"
        human = human_benchmarks[p]

        print(
            f"{p+1:<8} {mean_p:>11.2f} {sd:>8.2f} {ci_str:>18} "
            f"{human:>8.2f} {mean_t:>12.1f}"
        )

        agg_rows.append({
            "period": p + 1,
            "model": model_key,
            "n_runs": n,
            "llm_mean_price": round(mean_p, 3),
            "llm_sd_price": round(sd, 3),
            "llm_ci_lo": round(ci_lo, 3),
            "llm_ci_hi": round(ci_hi, 3),
            "human_mean_price": human,
            "llm_mean_trades": round(mean_t, 2),
            "human_ce_trades": 16,
            "ci_method": "bootstrap_10000",
        })

    print(f"{'='*70}")

    # Save bootstrap results
    bootstrap_path = output_dir / f"attanasi_bootstrap_{model_key}.csv"
    bootstrap_fields = [
        "period", "model", "n_runs",
        "llm_mean_price", "llm_sd_price", "llm_ci_lo", "llm_ci_hi",
        "human_mean_price", "llm_mean_trades", "human_ce_trades",
        "ci_method",
    ]
    with open(bootstrap_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=bootstrap_fields)
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"Bootstrap results saved to: {bootstrap_path}")


if __name__ == "__main__":
    main()
