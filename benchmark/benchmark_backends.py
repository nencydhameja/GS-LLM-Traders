#!/usr/bin/env python3
"""
Benchmark LLM backend latency using a realistic attanasi buyer prompt.

Usage:
    python3 benchmark_backends.py --url URL --model MODEL --label LABEL \
        [--n-calls N] [--out-json FILE] [--ollama-native-url URL]

Outputs a JSON array (appended to --out-json if it already exists).
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Representative attanasi buyer prompt (~900 tokens).
# Mimics a period-2, step-8 buyer in the perfect-competition treatment.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are participating in a market experiment as a BUYER.

Market structure:
- There are 4 sellers and 24 buyers in this market.
- Each of the 4 sellers can sell up to 24 units. Each buyer can buy at most 1 unit per period.
- This is a continuous double auction. Buyers submit bids and sellers submit asks.
- A trade occurs immediately when a bid meets or exceeds the lowest standing ask, or when an ask meets or falls below the highest standing bid.
- Price priority: earlier orders at the same price are matched first.
- Improvement rule: your bid must strictly exceed the current best bid to be accepted into the order book.

Your role:
- You are a buyer. Your private valuation for one unit is 20 points.
- Your payoff if you buy = (your valuation) - (the price you pay).
- You cannot bid above your valuation (the system will reject it).
- You may respond with PASS if you do not wish to bid this round.

Past period summary:
Period 1: You bought 1 unit at price 17, profit=3. Market: 16 trades, avg price=14.8.

Goal: maximize your total payoff across all periods. Think strategically about the current order book and whether to bid aggressively or wait."""

DECISION_PROMPT = """=== Trading Round 8 of 30 (Period 2) ===

Current Order Book:
  Best bid : 13  (buyer B7)
  Best ask : 16  (seller S2)
  Standing bids  : 13 (B7), 12 (B3), 11 (B21), 10 (B16)
  Standing asks  : 16 (S2), 17 (S1), 17 (S4)

Trades so far this period:
  Round 2: B2  bought from S3 at price 15  (B2 profit: 2,  S3 profit: 3)
  Round 3: B11 bought from S1 at price 16  (B11 profit: 1, S1 profit: 4)
  Round 5: B4  bought from S4 at price 15  (B4 profit: 2,  S4 profit: 3)
  Round 7: B22 bought from S2 at price 14  (B22 profit: 6, S2 profit: 2)

Your situation:
  Your valuation       : 20 points
  Units bought so far  : 0  (you have not traded this period)
  Profit this period   : 0
  Cumulated profit     : 3  (from period 1)

What is your bid? Reply with a single integer only."""


def build_messages() -> list[dict]:
    return [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{DECISION_PROMPT}"},
    ]


def run_benchmark(url: str, model: str, n_calls: int) -> dict:
    messages = build_messages()
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 20,
    }
    headers = {"Content-Type": "application/json"}

    latencies = []
    prompt_tokens_list = []
    completion_tokens_list = []

    for i in range(n_calls):
        t0 = time.perf_counter()
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        t1 = time.perf_counter()
        resp.raise_for_status()
        data = resp.json()

        latencies.append(t1 - t0)
        usage = data.get("usage", {})
        prompt_tokens_list.append(usage.get("prompt_tokens", 0))
        completion_tokens_list.append(usage.get("completion_tokens", 0))

        reply = data["choices"][0]["message"]["content"].strip()
        print(f"  call {i+1}/{n_calls}: {t1-t0:.2f}s  reply={repr(reply[:40])}")

    return {
        "mean_s":             round(statistics.mean(latencies), 3),
        "median_s":           round(statistics.median(latencies), 3),
        "min_s":              round(min(latencies), 3),
        "max_s":              round(max(latencies), 3),
        "stdev_s":            round(statistics.stdev(latencies), 3) if n_calls > 1 else 0,
        "calls":              n_calls,
        "prompt_tokens":      round(statistics.mean(prompt_tokens_list)),
        "completion_tokens":  round(statistics.mean(completion_tokens_list)),
    }


def ollama_native_timing(native_url: str, model: str) -> dict | None:
    """One call to Ollama's /api/generate to get prefill/generation tok/s."""
    prompt = f"{SYSTEM_PROMPT}\n\n{DECISION_PROMPT}"
    try:
        resp = requests.post(
            native_url,
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.7, "num_predict": 10}},
            timeout=120,
        )
        resp.raise_for_status()
        d = resp.json()
        prefill_tok_s = (
            d["prompt_eval_count"] / d["prompt_eval_duration"] * 1e9
            if d.get("prompt_eval_duration") else None
        )
        gen_tok_s = (
            d["eval_count"] / d["eval_duration"] * 1e9
            if d.get("eval_duration") else None
        )
        return {
            "prefill_tok_s":      round(prefill_tok_s, 1) if prefill_tok_s else None,
            "generation_tok_s":   round(gen_tok_s, 1) if gen_tok_s else None,
            "prompt_eval_count":  d.get("prompt_eval_count"),
            "eval_count":         d.get("eval_count"),
        }
    except Exception as e:
        print(f"  Ollama native timing call failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",              required=True,  help="OpenAI-compat endpoint")
    parser.add_argument("--model",            required=True,  help="Model name/alias")
    parser.add_argument("--label",            required=True,  help="Label for results (e.g. 'ollama')")
    parser.add_argument("--n-calls",          type=int, default=10)
    parser.add_argument("--out-json",         default=None,   help="Append results to this file")
    parser.add_argument("--ollama-native-url",default=None,   help="Ollama /api/generate URL for tok/s")
    args = parser.parse_args()

    print(f"\n--- Benchmarking {args.label} ({args.url}, model={args.model}, n={args.n_calls}) ---")
    stats = run_benchmark(args.url, args.model, args.n_calls)

    native = None
    if args.ollama_native_url:
        print("  Running Ollama native timing call...")
        native = ollama_native_timing(args.ollama_native_url, args.model)

    result = {
        "label":   args.label,
        "url":     args.url,
        "model":   args.model,
        **stats,
        **({"ollama_native": native} if native else {}),
    }

    print(f"\n  Result: {result}")

    if args.out_json:
        path = Path(args.out_json)
        existing = json.loads(path.read_text()) if path.exists() else []
        existing.append(result)
        path.write_text(json.dumps(existing, indent=2))
        print(f"  Saved to {path}")

    return result


if __name__ == "__main__":
    main()
