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


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_models():
    return load_config()["llm_models"]


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
    config = load_config()
    models = config["llm_models"]
    personas = config.get("personas", {})

    print("\n=== Attanasi et al. (2021) Experiment Runner ===")
    print("\nAvailable models:")
    for i, (key, info) in enumerate(models.items(), 1):
        print(f"  {i}. {key} ({info['name']})")

    model = prompt_choice("Select model", models, default=1)
    runs = prompt_int("Number of runs", 30)
    steps = prompt_int("Steps per period", 30)
    seed = prompt_int("Random seed", 42)
    resume = prompt_yn("Resume previous batch? (y/n)", "n")

    # Persona selection
    persona = None
    if personas:
        print("\nPersona treatment (robustness):")
        print("  0. None (baseline replication)")
        for i, key in enumerate(personas.keys(), 1):
            # Show first line of persona text as description
            desc = personas[key].strip().split("\n")[0]
            print(f"  {i}. {key} — {desc}")
        while True:
            raw = input(f"\nSelect persona [0]: ").strip()
            if raw == "" or raw == "0":
                break
            try:
                idx = int(raw)
                if 1 <= idx <= len(personas):
                    persona = list(personas.keys())[idx - 1]
                    break
            except ValueError:
                pass
            print(f"  Please enter a number between 0 and {len(personas)}.")

    # Memory type selection
    print("\nMemory type:")
    print("  1. Baseline (one-line cross-period summary)")
    print("  2. Refined (detailed: own actions, market distribution, price trajectory)")
    while True:
        raw = input("\nSelect memory type [1]: ").strip()
        if raw == "" or raw == "1":
            refined_memory = False
            break
        if raw == "2":
            refined_memory = True
            break
        print("  Please enter 1 or 2.")

    # Temperature selection
    print("\nTemperature (controls randomness / 'learning'):")
    print("  0. Use config default (0.7)")
    print("  1. 0.0 (deterministic)")
    print("  2. 0.3 (low randomness)")
    print("  3. 0.5 (moderate)")
    print("  4. 0.7 (config default)")
    print("  5. 1.0 (high randomness)")
    temp_options = {0: None, 1: 0.0, 2: 0.3, 3: 0.5, 4: 0.7, 5: 1.0}
    temperature = None
    while True:
        raw = input("\nSelect temperature [0]: ").strip()
        if raw == "" or raw == "0":
            break
        try:
            idx = int(raw)
            if idx in temp_options:
                temperature = temp_options[idx]
                break
        except ValueError:
            pass
        print("  Please enter a number between 0 and 5.")

    # Behavioral dials selection
    dials_config = config.get("behavioral_dials", {})
    dial_selections = {}  # dial_name -> level or None

    dial_labels = {
        "risk_aversion": "Risk Aversion",
        "aggressiveness": "Aggressiveness",
        "profit_orientation": "Profit Orientation",
    }

    if dials_config:
        print("\nBehavioral dials (factorial treatments):")
        for dial_name in ("risk_aversion", "aggressiveness", "profit_orientation"):
            if dial_name not in dials_config:
                continue
            levels = list(dials_config[dial_name].keys())
            print(f"\n  {dial_labels[dial_name]}:")
            print("    0. None (no behavioral text)")
            for i, level in enumerate(levels, 1):
                print(f"    {i}. {level}")
            while True:
                raw = input(f"  Select {dial_labels[dial_name]} [0]: ").strip()
                if raw == "" or raw == "0":
                    dial_selections[dial_name] = None
                    break
                try:
                    idx = int(raw)
                    if 1 <= idx <= len(levels):
                        dial_selections[dial_name] = levels[idx - 1]
                        break
                except ValueError:
                    pass
                print(f"    Please enter a number between 0 and {len(levels)}.")

    # Confirmation
    print("\n--- Confirm ---")
    print(f"  Model:   {model} ({models[model]['name']})")
    print(f"  Runs:    {runs}")
    print(f"  Steps:   {steps}")
    print(f"  Seed:    {seed}")
    print(f"  Resume:  {'Yes' if resume else 'No'}")
    print(f"  Persona: {persona or 'None (baseline)'}")
    print(f"  Memory:  {'Refined' if refined_memory else 'Baseline'}")
    print(f"  Temp:    {temperature if temperature is not None else 'config default (0.7)'}")
    for dial_name, dial_level in dial_selections.items():
        print(f"  {dial_labels[dial_name]}: {dial_level or 'None'}")
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
    if persona:
        cmd.extend(["--persona", persona])
    if refined_memory:
        cmd.append("--refined-memory")
    if temperature is not None:
        cmd.extend(["--temperature", str(temperature)])
    for dial_name, dial_level in dial_selections.items():
        if dial_level:
            flag = f"--{dial_name.replace('_', '-')}"
            cmd.extend([flag, dial_level])

    print(f"\nLaunching: {' '.join(cmd)}\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
