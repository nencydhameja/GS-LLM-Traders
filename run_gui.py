#!/usr/bin/env python3
"""
Interactive console menu for launching Attanasi et al. (2021) experiments.
Wraps attanasi_experiment.py so team members don't need to memorize CLI flags.
"""

import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "attanasi_config.yaml"
EXPERIMENT_SCRIPT = SCRIPT_DIR / "attanasi_experiment.py"


def load_models():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    return config["llm_models"]


def prompt_choice(label, options, default=1):
    """Prompt user to pick from a numbered list. Returns the key."""
    keys = list(options.keys())
    while True:
        raw = input(f"\n{label} [{default}]: ").strip()
        if raw == "":
            return keys[default - 1]
        try:
            idx = int(raw)
            if 1 <= idx <= len(keys):
                return keys[idx - 1]
        except ValueError:
            pass
        print(f"  Please enter a number between 1 and {len(keys)}.")


def prompt_int(label, default):
    """Prompt for a positive integer with a default."""
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if val > 0:
                return val
        except ValueError:
            pass
        print("  Please enter a positive integer.")


def prompt_yn(label, default="n"):
    """Prompt for y/n with a default."""
    while True:
        raw = input(f"{label} [{default}]: ").strip().lower()
        if raw == "":
            return default == "y"
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please enter y or n.")


def main():
    models = load_models()

    print("\n=== Attanasi et al. (2021) Experiment Runner ===")
    print("\nAvailable models:")
    for i, (key, info) in enumerate(models.items(), 1):
        print(f"  {i}. {key} ({info['name']})")

    model = prompt_choice("Select model", models, default=1)
    runs = prompt_int("Number of runs", 30)
    steps = prompt_int("Steps per period", 30)
    seed = prompt_int("Random seed", 42)
    resume = prompt_yn("Resume previous batch? (y/n)", "n")

    # Confirmation
    print("\n--- Confirm ---")
    print(f"  Model:  {model} ({models[model]['name']})")
    print(f"  Runs:   {runs}")
    print(f"  Steps:  {steps}")
    print(f"  Seed:   {seed}")
    print(f"  Resume: {'Yes' if resume else 'No'}")
    print()

    if not prompt_yn("Start experiment? (y/n)", "y"):
        print("Aborted.")
        return

    # Build command
    cmd = [
        sys.executable, str(EXPERIMENT_SCRIPT),
        "--model", model,
        "--runs", str(runs),
        "--steps", str(steps),
        "--seed", str(seed),
    ]
    if resume:
        cmd.append("--resume")

    print(f"\nLaunching: {' '.join(cmd)}\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
