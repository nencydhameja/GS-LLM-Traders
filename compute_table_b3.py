#!/usr/bin/env python3
"""
Compute Table B.3 equivalent for LLM Competition results.

Replicates Table B.3 from Attanasi, Centorrino & Moscati (2021):
"Fractions of intramarginal sellers and buyers that do not trade"

Intramarginal agents:
  - Sellers: all 4 (cost=12, can profit at any price > 12)
  - Buyers: 16 with valuation > 12 (valuations 20, 17, 15, 13 x 4 groups)

Human benchmarks from Table B.3 (Competition played first):
  Sellers: 4.17%, 6.25%, 5.56%
  Buyers: 20.42%, 20.00%, 20.97%
"""

import csv
import glob
import os
from collections import defaultdict

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

HUMAN_SELLERS = {1: 4.17, 2: 6.25, 3: 5.56}
HUMAN_BUYERS = {1: 20.42, 2: 20.00, 3: 20.97}


def find_master_csv(model="mistral-instruct"):
    pattern = os.path.join(OUTPUT_DIR, f"attanasi_master_{model}_*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No master CSV found for {model} in {OUTPUT_DIR}")
    return files[-1]  # most recent


def compute_table_b3(master_path):
    with open(master_path) as f:
        rows = list(csv.DictReader(f))

    model = rows[0]["model"]

    # Identify agents
    agent_info = {}
    for r in rows:
        aid = r["agent_id"]
        if aid not in agent_info:
            agent_info[aid] = (r["agent_type"], int(r["reservation_value"]))

    sellers = [a for a, (t, v) in agent_info.items() if t == "seller"]
    intra_buyers = [a for a, (t, v) in agent_info.items() if t == "buyer" and v > 12]

    # Track which agents traded per run x period
    traded = defaultdict(set)
    for r in rows:
        if r["matched"] == "True":
            traded[(r["run_id"], r["period"])].add(r["agent_id"])

    # Count runs
    run_ids = sorted(set(r["run_id"] for r in rows), key=int)
    n_runs = len(run_ids)

    # Compute fractions
    results = []
    for period in [1, 2, 3]:
        s_no_trade = 0
        s_total = 0
        b_no_trade = 0
        b_total = 0

        for run_id in run_ids:
            key = (run_id, str(period))
            traded_set = traded.get(key, set())

            for s in sellers:
                s_total += 1
                if s not in traded_set:
                    s_no_trade += 1

            for b in intra_buyers:
                b_total += 1
                if b not in traded_set:
                    b_no_trade += 1

        s_frac = s_no_trade / s_total * 100
        b_frac = b_no_trade / b_total * 100
        results.append({
            "period": period,
            "model": model,
            "n_runs": n_runs,
            "llm_sellers_not_trading_pct": round(s_frac, 2),
            "human_sellers_not_trading_pct": HUMAN_SELLERS[period],
            "llm_buyers_not_trading_pct": round(b_frac, 2),
            "human_buyers_not_trading_pct": HUMAN_BUYERS[period],
        })

    return results


def save_csv(results):
    out_path = os.path.join(OUTPUT_DIR, f"attanasi_table_b3_{results[0]['model']}.csv")
    fields = list(results[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    return out_path


def print_table(results):
    model = results[0]["model"]
    n = results[0]["n_runs"]
    print(f"\nTable B.3 equivalent: {model.upper()} vs HUMAN ({n} runs)")
    print(f"Fractions of intramarginal sellers/buyers that do NOT trade (Competition)\n")
    print(f"              Sellers                         Buyers")
    print(f"         t=1      t=2      t=3           t=1      t=2      t=3")
    print(f"Human   {HUMAN_SELLERS[1]:5.2f}%   {HUMAN_SELLERS[2]:5.2f}%   {HUMAN_SELLERS[3]:5.2f}%"
          f"        {HUMAN_BUYERS[1]:5.2f}%   {HUMAN_BUYERS[2]:5.2f}%   {HUMAN_BUYERS[3]:5.2f}%")
    s = [r["llm_sellers_not_trading_pct"] for r in results]
    b = [r["llm_buyers_not_trading_pct"] for r in results]
    print(f"LLM    {s[0]:5.2f}%  {s[1]:5.2f}%  {s[2]:5.2f}%"
          f"       {b[0]:5.2f}%  {b[1]:5.2f}%  {b[2]:5.2f}%")
    print()


if __name__ == "__main__":
    master = find_master_csv()
    print(f"Reading: {master}")
    results = compute_table_b3(master)
    out = save_csv(results)
    print(f"Saved:   {out}")
    print_table(results)
