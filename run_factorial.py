#!/usr/bin/env python3
"""
Factorial batch runner for behavioral dials and temperature.

Iterates all combinations of behavioral dials (5x5x5 = 125) and/or
temperature levels. Calls attanasi_experiment.py as a subprocess for
each combination, tracking progress in a JSON file so it can be resumed.

Usage:
    # Full dial factorial (125 combos x N runs each):
    python run_factorial.py --model gemma3-27b --runs-per-combo 5

    # Single-dial sweep (5 combos):
    python run_factorial.py --model gemma3-27b --runs-per-combo 5 --subset risk_aversion

    # Temperature sweep only (no dials):
    python run_factorial.py --model gemma3-27b --runs-per-combo 5 --subset temperature

    # Temperature + one dial (e.g., 5 temps x 5 risk levels = 25 combos):
    python run_factorial.py --model gemma3-27b --runs-per-combo 5 --subset risk_aversion --temperatures 0.0 0.3 0.5 0.7 1.0

    # Resume after interruption:
    python run_factorial.py --model gemma3-27b --runs-per-combo 5 --resume
"""

import argparse
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
EXPERIMENT_SCRIPT = SCRIPT_DIR / "attanasi_experiment.py"

DIALS = {
    "risk_aversion": ["very_averse", "averse", "neutral", "seeking", "very_seeking"],
    "aggressiveness": ["very_passive", "passive", "moderate", "aggressive", "very_aggressive"],
    "profit_orientation": ["very_conservative", "conservative", "balanced", "maximizer", "extreme_maximizer"],
}

DEFAULT_TEMPERATURES = [0.0, 0.3, 0.5, 0.7, 1.0]


def build_combos(subset: str = None, temperatures: list = None) -> list:
    """Build the list of dial/temperature combinations to run.

    If subset is 'temperature', only temperature varies (no dials).
    If subset is a dial name, only that dial varies.
    If temperatures are provided, they cross with whatever dials are active.
    Otherwise, full factorial of all 3 dials.
    """
    temps = temperatures or [None]  # None = use config default

    if subset == "temperature":
        # Temperature-only sweep, no dials
        if not temperatures:
            temps = DEFAULT_TEMPERATURES
        combos = []
        for t in temps:
            combos.append({
                "risk_aversion": None,
                "aggressiveness": None,
                "profit_orientation": None,
                "temperature": t,
            })
        return combos

    if subset and subset in DIALS:
        # Single-dial sweep, optionally crossed with temperature
        combos = []
        for level in DIALS[subset]:
            for t in temps:
                combo = {
                    "risk_aversion": None,
                    "aggressiveness": None,
                    "profit_orientation": None,
                    "temperature": t,
                }
                combo[subset] = level
                combos.append(combo)
        return combos

    # Full factorial of all 3 dials, optionally crossed with temperature
    combos = []
    for r, a, p in itertools.product(
        DIALS["risk_aversion"],
        DIALS["aggressiveness"],
        DIALS["profit_orientation"],
    ):
        for t in temps:
            combos.append({
                "risk_aversion": r,
                "aggressiveness": a,
                "profit_orientation": p,
                "temperature": t,
            })
    return combos


def combo_key(combo: dict) -> str:
    """Stable string key for a combo (used in progress tracking)."""
    parts = []
    for dial in ("risk_aversion", "aggressiveness", "profit_orientation"):
        parts.append(combo.get(dial) or "none")
    t = combo.get("temperature")
    parts.append(f"t{t}" if t is not None else "tdefault")
    return "|".join(parts)


def run_combo(combo: dict, model: str, runs: int, seed: int, steps: int,
              persona: str = None, refined_memory: bool = False) -> bool:
    """Run a single combo. Returns True if subprocess exited successfully."""
    cmd = [
        sys.executable, str(EXPERIMENT_SCRIPT),
        "--model", model,
        "--runs", str(runs),
        "--steps", str(steps),
        "--seed", str(seed),
        "--resume",  # always resume within a combo's runs
    ]
    if persona:
        cmd.extend(["--persona", persona])
    if refined_memory:
        cmd.append("--refined-memory")

    # Temperature
    t = combo.get("temperature")
    if t is not None:
        cmd.extend(["--temperature", str(t)])

    # Dials
    for dial_name in ("risk_aversion", "aggressiveness", "profit_orientation"):
        dial_level = combo.get(dial_name)
        if dial_level:
            flag = f"--{dial_name.replace('_', '-')}"
            cmd.extend([flag, dial_level])

    print(f"\n{'='*70}")
    print(f"FACTORIAL: {combo_key(combo)}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Factorial batch runner for behavioral dials and temperature"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="LLM model key from config",
    )
    parser.add_argument(
        "--runs-per-combo", type=int, default=5,
        help="Number of independent runs per combination (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--steps", type=int, default=30,
        help="Trading rounds per period (default: 30)",
    )
    parser.add_argument(
        "--subset", type=str, default=None,
        choices=list(DIALS.keys()) + ["temperature"],
        help="Vary only one dimension (5 combos instead of 125). "
             "Use 'temperature' for temperature-only sweep.",
    )
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=None,
        help="Temperature levels to sweep (default: 0.0 0.3 0.5 0.7 1.0). "
             "Cross with dials if --subset is a dial name.",
    )
    parser.add_argument(
        "--persona", type=str, default=None,
        help="Persona key to use for all combos (default: none)",
    )
    parser.add_argument(
        "--refined-memory", action="store_true",
        help="Use refined memory for all combos",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume a previous factorial batch (skip completed combos)",
    )
    args = parser.parse_args()

    # Build combo list
    combos = build_combos(args.subset, args.temperatures)
    print(f"Factorial design: {len(combos)} combinations x {args.runs_per_combo} runs each")

    # Progress tracking
    output_dir = SCRIPT_DIR / "output"
    output_dir.mkdir(exist_ok=True)
    subset_tag = f"_{args.subset}" if args.subset else "_full"
    if args.temperatures:
        subset_tag += "_temps"
    progress_file = output_dir / f"factorial_progress_{args.model}{subset_tag}.json"

    if args.resume and progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        completed = set(progress.get("completed", []))
        print(f"Resuming: {len(completed)}/{len(combos)} combos already done")
    else:
        progress = {
            "model": args.model,
            "subset": args.subset,
            "temperatures": args.temperatures,
            "runs_per_combo": args.runs_per_combo,
            "seed": args.seed,
            "started": datetime.now().isoformat(),
            "completed": [],
            "failed": [],
        }
        completed = set()

    # Run loop
    total = len(combos)
    successes = 0
    failures = 0

    for i, combo in enumerate(combos, 1):
        key = combo_key(combo)

        if key in completed:
            print(f"[{i}/{total}] SKIP (already done): {key}")
            continue

        print(f"\n[{i}/{total}] Running: {key}")
        t0 = time.time()

        ok = run_combo(
            combo, args.model, args.runs_per_combo, args.seed, args.steps,
            persona=args.persona, refined_memory=args.refined_memory,
        )

        elapsed = time.time() - t0
        if ok:
            successes += 1
            completed.add(key)
            progress["completed"] = sorted(completed)
            print(f"  Done in {elapsed:.0f}s. [{len(completed)}/{total} complete]")
        else:
            failures += 1
            progress.setdefault("failed", []).append(key)
            print(f"  FAILED after {elapsed:.0f}s. Will retry on --resume.")

        # Save progress after each combo
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)

    # Final summary
    print(f"\n{'='*70}")
    print(f"FACTORIAL COMPLETE")
    print(f"  Total combos: {total}")
    print(f"  Completed: {len(completed)}")
    print(f"  Failed this session: {failures}")
    print(f"  Remaining: {total - len(completed)}")
    print(f"{'='*70}")

    if len(completed) >= total:
        progress_file.unlink(missing_ok=True)
        print("All combos complete -- progress file removed.")
    elif failures > 0:
        print(f"\nRe-run with --resume to retry {total - len(completed)} remaining combos.")


if __name__ == "__main__":
    main()
