#!/usr/bin/env python3
"""
Attanasi, Centorrino & Moscati (2021) — Perfect Competition Replication
"Controlling monopoly power in a double-auction market experiment"
JPET 23(5), 1074-1101

Phase 1: Perfect Competition with LLM agents.
Compares LLM trading behavior with human results (Table B.1).

Usage:
    python attanasi_experiment.py --model llama3.3 --steps 30 --runs 1 --seed 42
"""

import argparse
import csv
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import yaml

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
        self.cumulated_payoffs = 0
        self.llm_successes = 0
        self.llm_errors = 0

    def reset_period(self):
        """Reset units remaining for a new trading period."""
        self.units_remaining = self.capacity
        self.period_trades = []

    def can_trade(self) -> bool:
        return self.units_remaining > 0

    def decide_price(self, market_state: dict) -> Tuple[Optional[int], str]:
        """Query LLM for a price decision.

        Returns (price, status) where status is one of:
          'ok'                - valid price returned
          'skip'              - agent can't trade (no remaining units)
          'llm_error'         - LLM call failed (timeout, connection, HTTP error)
          'parse_error'       - couldn't extract integer from LLM response
          'constraint_error'  - price violated budget/range constraint
        """
        if not self.can_trade():
            return None, "skip"

        prompt = self._build_prompt(market_state)
        raw_response = self._query_llm(prompt)
        if raw_response is None:
            return None, "llm_error"

        price, extract_status = self._extract_price(raw_response)
        if price is None:
            return None, extract_status

        return price, "ok"

    def _build_prompt(self, market_state: dict) -> str:
        """Build prompt mirroring exactly what humans see on screen (Figs A.4/A.5)."""
        period = market_state.get("period", 1)

        # System context (top-level instructions, like what humans read at start)
        if self.agent_type == "buyer":
            system = self.prompt_templates["buyer_system"].format(
                agent_id=self.agent_id,
                valuation=self.reservation_value,
                period=period,
            )
            action_verb = "bid"
        else:
            system = self.prompt_templates["seller_system"].format(
                agent_id=self.agent_id,
                cost=self.reservation_value,
                period=period,
            )
            action_verb = "offer"

        # Transaction history — humans see ALL trades this period
        all_trades = market_state.get("all_trades", [])
        if all_trades:
            tx_lines = []
            for i, t in enumerate(all_trades, 1):
                tx_lines.append(f"  Trade {i}: {t.buyer_id} bought from {t.seller_id} @ {t.price}")
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
                units_sold = len(self.period_trades)
                your_situation = "\n".join(sit_lines)
                your_situation += f"\n  Units sold this period: {units_sold}/{self.capacity}"
            else:
                your_situation = (
                    f"Unit 1: Cost={self.reservation_value}, "
                    f"Your ask=-, Transaction price=-, Payoff=-\n"
                    f"  Units sold this period: 0/{self.capacity}"
                )

        best_bid = market_state.get("best_bid")
        best_ask = market_state.get("best_ask")

        decision = self.prompt_templates["decision"].format(
            best_bid=best_bid if best_bid is not None else "None",
            best_ask=best_ask if best_ask is not None else "None",
            units_to_buy=market_state.get("units_to_buy", "?"),
            units_to_sell=market_state.get("units_to_sell", "?"),
            units_traded=market_state.get("units_traded", 0),
            current_step=market_state.get("current_step", "?"),
            total_steps=market_state.get("total_steps", "?"),
            transaction_history=transaction_history,
            your_situation=your_situation,
            cumulated_payoffs=self.cumulated_payoffs,
            action_verb=action_verb,
        )

        return f"{system}\n{decision}"

    def _query_llm(self, prompt: str) -> Optional[str]:
        """POST to Binghamton API, return raw text response."""
        try:
            headers = {"Content-Type": "application/json"}
            if self.llm_config.get("api_key"):
                headers["Authorization"] = f"Bearer {self.llm_config['api_key']}"

            response = requests.post(
                self.llm_config["url"],
                headers=headers,
                json={
                    "model": self.llm_config["model"],
                    "messages": [{"role": "user", "content": prompt}],
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
        """Extract integer price from LLM response, validate constraints.

        Returns (price, status) where status is 'ok', 'parse_error', or 'constraint_error'.
        """
        if not text or not text.strip():
            return None, "parse_error"

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

        # Validate range [0, 30]
        if price < 0 or price > 30:
            print(
                f"  ERROR {self.agent_id}: Price {price} outside range [0, 30]"
            )
            return None, "constraint_error"

        # Validate budget constraint
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
    """Continuous double auction with price-time priority matching."""

    def __init__(self):
        self.bids: List[OrderBookEntry] = []  # buy orders (highest first)
        self.asks: List[OrderBookEntry] = []  # sell orders (lowest first)
        self.trades: List[Trade] = []
        self.all_period_trades: List[Trade] = []

    def reset_period(self):
        """Clear the order book for a new period (keep cumulative trades)."""
        self.bids.clear()
        self.asks.clear()
        self.trades.clear()

    def best_bid(self) -> Optional[int]:
        return self.bids[0].price if self.bids else None

    def best_ask(self) -> Optional[int]:
        return self.asks[0].price if self.asks else None

    def recent_prices(self, n: int = 10) -> List[int]:
        return [t.price for t in self.trades[-n:]]

    def market_state(self, agents: Dict[str, "Agent"] = None, period: int = 1,
                     current_step: int = 1, total_steps: int = 30) -> dict:
        # Compute available units for purchase/sale (what humans see on screen)
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
            "all_trades": list(self.trades),  # full transaction history this period
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
    ) -> Optional[Trade]:
        """Submit a buy order. Check for immediate match with lowest ask."""
        now = time.time()

        # Check for match: bid >= lowest ask
        if self.asks and price >= self.asks[0].price:
            ask_entry = self.asks.pop(0)
            trade_price = ask_entry.price  # Price priority: earlier order sets price
            trade = self._execute_trade(
                trade_price, agent_id, ask_entry.agent_id, agents, period, step
            )
            return trade

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
        return None

    def submit_ask(
        self, price: int, agent_id: str, agents: Dict[str, "Agent"],
        period: int, step: int,
    ) -> Optional[Trade]:
        """Submit a sell order. Check for immediate match with highest bid."""
        now = time.time()

        # Check for match: ask <= highest bid
        if self.bids and price <= self.bids[0].price:
            bid_entry = self.bids.pop(0)
            trade_price = bid_entry.price  # Price priority: earlier order sets price
            trade = self._execute_trade(
                trade_price, bid_entry.agent_id, agent_id, agents, period, step
            )
            return trade

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
        return None

    def _execute_trade(
        self, price: int, buyer_id: str, seller_id: str,
        agents: Dict[str, "Agent"], period: int, step: int,
    ) -> Trade:
        """Record a trade and update agent state."""
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
        buyer.cumulated_payoffs += trade.buyer_profit
        seller.cumulated_payoffs += trade.seller_profit
        self.trades.append(trade)
        self.all_period_trades.append(trade)

        # Remove any stale orders from agents who just traded
        self.bids = [b for b in self.bids if b.agent_id != buyer_id or buyer.can_trade()]
        self.asks = [a for a in self.asks if a.agent_id != seller_id or seller.can_trade()]

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

    def __init__(self, config: dict, model_key: str, num_steps: int, seed: int):
        self.config = config
        self.exp = config["experiment"]
        self.num_steps = num_steps
        self.seed = seed
        self.rng = random.Random(seed)

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
            )

        # 24 buyers: 4 groups of 6, each group gets the same valuation schedule
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
                )
                buyer_idx += 1

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
        """Run one trading period."""
        print(f"\n--- Period {period} ---")

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

        for step in range(1, self.num_steps + 1):
            # Randomize order each step
            self.rng.shuffle(agent_ids)

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
                    continue
                elif status == "constraint_error":
                    period_constraint_errors += 1
                    continue
                elif status == "parse_error":
                    period_parse_errors += 1
                    continue
                elif price is None:
                    continue

                # Submit to auction
                if agent.agent_type == "buyer":
                    action = "bid"
                    trade = self.auction.submit_bid(
                        price, aid, self.agents, period, step
                    )
                else:
                    action = "ask"
                    trade = self.auction.submit_ask(
                        price, aid, self.agents, period, step
                    )

                # Log every decision
                decision = {
                    "period": period,
                    "step": step,
                    "agent_id": aid,
                    "agent_type": agent.agent_type,
                    "action": action,
                    "submitted_price": price,
                    "reservation_value": agent.reservation_value,
                    "matched": trade is not None,
                    "trade_price": trade.price if trade else "",
                    "counterparty": (trade.seller_id if action == "bid" else trade.buyer_id) if trade else "",
                    "profit": (trade.buyer_profit if action == "bid" else trade.seller_profit) if trade else "",
                }
                self.all_decisions.append(decision)

                if trade is None:
                    model = self.llm_config.get("model", "?")
                    print(
                        f"  {aid} ({agent.agent_type}): [{model}] "
                        f"{action} {price} (limit: {agent.reservation_value})"
                    )
                else:
                    trades_this_period += 1

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

        # Period summary
        period_trades = [
            t for t in self.auction.all_period_trades if t.period == period
        ]
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
        }
        self.period_results.append(result)

        print(
            f"\n  Period {period} summary: {len(period_trades)} trades, "
            f"avg price = {avg_price:.2f} "
            f"(human: {self.HUMAN_BENCHMARKS[period-1]:.2f}) "
            f"[LLM calls: {period_llm_calls}, errors: {period_llm_errors}, "
            f"constraint: {period_constraint_errors}]"
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

    def get_decision_rows(self, run_id: int, run_seed: int, batch_timestamp: str) -> List[dict]:
        """Return all decision rows (every bid/ask) with run metadata."""
        rows = []
        for d in self.all_decisions:
            rows.append({
                "run_id": run_id,
                "seed": run_seed,
                "model": self.model_key,
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
        help="Resume a previous batch — skip completed runs, append to existing CSVs",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (default: attanasi_config.yaml in same dir)",
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

    # Output directory
    output_dir = script_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # --- Progress tracking ---
    # Progress file stores: batch_timestamp, model, completed seeds, file paths
    progress_file = output_dir / f"attanasi_progress_{args.model}.json"

    master_fields = [
        "run_id", "seed", "model", "timestamp",
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
        print(f"Resuming batch {batch_timestamp}: "
              f"{len(completed_seeds)}/{args.runs} runs already completed")
    else:
        # New batch
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        completed_seeds = set()
        master_path = output_dir / f"attanasi_master_{args.model}_{batch_timestamp}.csv"

        # Write CSV header
        with open(master_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=master_fields).writeheader()

        # Save initial progress
        progress = {
            "batch_timestamp": batch_timestamp,
            "model": args.model,
            "completed_seeds": [],
            "master_path": str(master_path),
            "total_runs": args.runs,
            "base_seed": args.seed,
        }
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)

    # --- Run loop ---
    all_run_results = []
    new_completed = 0

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        run_id = run_idx + 1

        # Skip already-completed runs
        if run_seed in completed_seeds:
            print(f"\n  Skipping run {run_id}/{args.runs} (seed={run_seed}) — already completed")
            continue

        if args.runs > 1:
            remaining = args.runs - len(completed_seeds) - new_completed
            print(f"\n{'#'*60}")
            print(f"# RUN {run_id}/{args.runs} (seed={run_seed}) — {remaining} remaining")
            print(f"{'#'*60}")

        experiment = CompetitionExperiment(
            config=config,
            model_key=args.model,
            num_steps=args.steps,
            seed=run_seed,
        )
        experiment.run()

        # Check if run was successful (had any trades)
        total_trades = sum(r["num_trades"] for r in experiment.period_results)
        total_errors = sum(r.get("llm_errors", 0) for r in experiment.period_results)

        if total_trades == 0 and total_errors > 0:
            print(f"\n  *** Run {run_id} FAILED (0 trades, {total_errors} LLM errors). "
                  f"Skipping — will retry on --resume. ***")
            continue

        # --- Save immediately (append to master CSV) ---
        decision_rows = experiment.get_decision_rows(run_id, run_seed, batch_timestamp)

        with open(master_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=master_fields)
            writer.writerows(decision_rows)

        # Update progress file
        completed_seeds.add(run_seed)
        new_completed += 1
        progress["completed_seeds"] = sorted(completed_seeds)
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)

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
        _print_aggregate(all_run_results, args.model)

    # Clean up progress file when all runs complete
    if len(completed_seeds) >= args.runs:
        progress_file.unlink(missing_ok=True)
        print("All runs complete — progress file removed.")


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


def _print_aggregate(all_run_results: list, model_key: str):
    """Compute and print cross-run statistics with bootstrap 95% CIs."""
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

    for p in range(num_periods):
        prices = [
            run[p]["avg_price"]
            for run in all_run_results
            if not math.isnan(run[p]["avg_price"])
        ]
        trades = [run[p]["num_trades"] for run in all_run_results]

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

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
