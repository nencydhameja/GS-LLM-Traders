"""
Parse experiment output logs to reconstruct trade CSV files and summary files.
"""
import re
import csv
import os
from collections import defaultdict

# Buyer valuations lookup
BUYER_VALUATIONS = {}
val_map = {
    20: [1, 7, 13, 19],
    17: [2, 8, 14, 20],
    15: [3, 9, 15, 21],
    13: [4, 10, 16, 22],
    8:  [5, 11, 17, 23],
    0:  [6, 12, 18, 24],
}
for val, ids in val_map.items():
    for bid in ids:
        BUYER_VALUATIONS[f"B{bid}"] = val

SELLER_COST = 12

# Human benchmark averages
HUMAN_AVG = {1: 14.49, 2: 14.10, 3: 13.50}

# Run configurations
RUNS = [
    {
        "run_id": 1,
        "label": "basic_prompts",
        "log_path": "/private/tmp/claude-501/-Users-nencydhameja/tasks/b45a0a1.output",
        "model": "llama3.3",
        "seed": 42,
    },
    {
        "run_id": 2,
        "label": "human_prompts",
        "log_path": "/private/tmp/claude-501/-Users-nencydhameja/tasks/be5f10c.output",
        "model": "llama3.3",
        "seed": 42,
    },
]

OUTPUT_DIR = "/Users/nencydhameja/PycharmProjects/LLM_ABM_Project/GS-LLM/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Regex patterns
PERIOD_RE = re.compile(r"---\s*Period\s+(\d+)\s*---")
TRADE_RE = re.compile(
    r"\*\*\*\s*TRADE:\s*(B\d+)\s+buys\s+from\s+(S\d+)\s+@\s+(\d+)"
    r"\s+\(buyer profit:\s*(\d+),\s*seller profit:\s*(\d+)\)"
)


def parse_log(log_path):
    """Parse a log file and return list of trade dicts."""
    trades = []
    current_period = None
    step_counter = 0

    with open(log_path, "r") as f:
        for line in f:
            # Check for period header
            m_period = PERIOD_RE.search(line)
            if m_period:
                current_period = int(m_period.group(1))
                step_counter = 0
                continue

            # Check for trade
            m_trade = TRADE_RE.search(line)
            if m_trade and current_period is not None:
                step_counter += 1
                buyer_id = m_trade.group(1)
                seller_id = m_trade.group(2)
                price = int(m_trade.group(3))
                buyer_profit = int(m_trade.group(4))
                seller_profit = int(m_trade.group(5))
                buyer_valuation = BUYER_VALUATIONS.get(buyer_id, None)

                trades.append({
                    "period": current_period,
                    "step": step_counter,
                    "price": price,
                    "buyer_id": buyer_id,
                    "seller_id": seller_id,
                    "buyer_valuation": buyer_valuation,
                    "seller_cost": SELLER_COST,
                    "buyer_profit": buyer_profit,
                    "seller_profit": seller_profit,
                })

    return trades


def write_trade_csv(trades, run_cfg, output_path):
    """Write trade-level CSV."""
    fieldnames = [
        "run_id", "seed", "model", "period", "step", "price",
        "buyer_id", "seller_id", "buyer_valuation", "seller_cost",
        "buyer_profit", "seller_profit",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            row = {
                "run_id": run_cfg["run_id"],
                "seed": run_cfg["seed"],
                "model": run_cfg["model"],
                **t,
            }
            writer.writerow(row)
    print(f"  Wrote {len(trades)} trades to {output_path}")


def write_summary_csv(trades, run_cfg, output_path):
    """Write period-level summary CSV."""
    by_period = defaultdict(list)
    for t in trades:
        by_period[t["period"]].append(t)

    fieldnames = [
        "run_id", "seed", "model", "period", "num_trades",
        "avg_price", "min_price", "max_price", "human_avg",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for period in sorted(by_period.keys()):
            period_trades = by_period[period]
            prices = [t["price"] for t in period_trades]
            avg_price = sum(prices) / len(prices)
            row = {
                "run_id": run_cfg["run_id"],
                "seed": run_cfg["seed"],
                "model": run_cfg["model"],
                "period": period,
                "num_trades": len(prices),
                "avg_price": f"{avg_price:.2f}",
                "min_price": min(prices),
                "max_price": max(prices),
                "human_avg": HUMAN_AVG.get(period, ""),
            }
            writer.writerow(row)
    print(f"  Wrote {len(by_period)} period summaries to {output_path}")


def verify_trades(trades, run_cfg):
    """Verify parsed trades against expected summaries."""
    by_period = defaultdict(list)
    for t in trades:
        by_period[t["period"]].append(t)

    print(f"\n  Verification for Run {run_cfg['run_id']} ({run_cfg['label']}):")
    all_ok = True
    for period in sorted(by_period.keys()):
        period_trades = by_period[period]
        prices = [t["price"] for t in period_trades]
        avg_price = sum(prices) / len(prices)
        print(f"    Period {period}: {len(prices)} trades, avg price {avg_price:.2f}")

        # Verify buyer profit = valuation - price, seller profit = price - cost
        for t in period_trades:
            expected_bp = t["buyer_valuation"] - t["price"]
            expected_sp = t["price"] - SELLER_COST
            if expected_bp != t["buyer_profit"]:
                print(f"    WARNING: {t['buyer_id']} buyer_profit mismatch: "
                      f"expected {expected_bp}, got {t['buyer_profit']}")
                all_ok = False
            if expected_sp != t["seller_profit"]:
                print(f"    WARNING: {t['seller_id']} seller_profit mismatch: "
                      f"expected {expected_sp}, got {t['seller_profit']}")
                all_ok = False

    if all_ok:
        print("    All profit calculations verified OK.")


# Main
for run_cfg in RUNS:
    label = run_cfg["label"]
    run_id = run_cfg["run_id"]
    print(f"\n=== Processing Run {run_id} ({label}) ===")

    trades = parse_log(run_cfg["log_path"])

    trade_path = os.path.join(OUTPUT_DIR, f"attanasi_trades_run{run_id}_{label}.csv")
    summary_path = os.path.join(OUTPUT_DIR, f"attanasi_summary_run{run_id}_{label}.csv")

    write_trade_csv(trades, run_cfg, trade_path)
    write_summary_csv(trades, run_cfg, summary_path)
    verify_trades(trades, run_cfg)

print("\nDone.")
