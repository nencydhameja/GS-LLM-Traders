#!/usr/bin/env python3
"""End-to-end convergence analysis for the Attanasi (2021) LLM replication.

Reproduces the full human-vs-LLM analysis in one pass and writes a single PDF:
  page 1 = price-trajectory figure (each condition vs the human benchmark, CE band)
  page 2 = condition-summary table (convergence slope + P3-P1 drift, clustered SE)

It auto-discovers every attanasi_master_*.csv in the data dir, derives each
condition label from the file's own columns (memory_type / persona / dial levels
/ temperature), and pools files that share a condition. So when the step 4-9
dial/temperature sweeps land, just re-run this and they appear automatically.

Methods (match the experiment pipeline exactly):
  - trade = row with matched == True; price = trade_price; run = seed
  - period mean price per run -> mean across runs; bootstrap 95% CI is the
    percentile CI over runs (n_boot=10000, random.Random(seed=999)), matching
    attanasi_experiment._bootstrap_ci
  - convergence slope: trade-level OLS price ~ period, CR1 SE clustered by run
  - net drift: saturated OLS price ~ C(period), P3-P1 coef, CR1 clustered by run

Usage:
  python run_analysis.py [--data-dir DIR] [--out PDF] [--model gemma3-27b]
                         [--plot COND1,COND2,...]
"""
import argparse
import csv
import glob
import math
import os
import random
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Human benchmark, Attanasi et al. (2021) Table B.1, Competition played first.
HUMAN = {1: 14.49, 2: 14.10, 3: 13.50}
CE_LO, CE_HI = 12.0, 13.0

# Canonical clean cells: the 6 gemma3-27b masters produced by the *fixed* code
# (June 2026, 30 runs each). Explicit on purpose -- auto-discovering every
# attanasi_master_*.csv would pool old-code partial runs under the same labels
# and mix code versions. To add a condition (e.g. a completed dial sweep),
# append (label, filename) here.
CANONICAL = [
    ("baseline",                 "attanasi_master_gemma3-27b_20260608_134243.csv"),
    ("baseline_replicate",       "attanasi_master_gemma3-27b_20260608_214223.csv"),
    ("refined",                  "attanasi_master_gemma3-27b_refined_20260609_055319.csv"),
    ("persona_undergrad_econ",   "attanasi_master_gemma3-27b_20260609_131827.csv"),
    ("persona_grad_econ",        "attanasi_master_gemma3-27b_20260609_232417.csv"),
    ("persona_undergrad_human",  "attanasi_master_gemma3-27b_20260610_075116.csv"),
]
# Conditions shown in the trajectory figure (baseline_replicate omitted to avoid
# a near-duplicate line; it stays in the CSV tables as a replication check).
DEFAULT_PLOT = ["baseline", "refined", "persona_undergrad_econ",
                "persona_grad_econ", "persona_undergrad_human"]

# Behavioral-dial sweeps. IMPORTANT: these were run on the llama-cpp-python
# backend (Jun 20-24), a DIFFERENT engine/model artifact than the Ollama
# baseline/persona cells above (Jun 8-11). They are therefore analysed and
# plotted SEPARATELY (own pages, own CSVs); dial cells are never placed on the
# same axis as the Ollama cells, only compared within-dial and vs the human
# benchmark. Files are the verified single 30-run masters per level (the drve/
# dave filename codes collide very_averse/very_seeking and very_passive/
# very_aggressive, so the exact file is pinned, not the suffix).
DIAL_BACKEND = "llama-cpp-python (Jun 20-24)"
DIAL_CELLS = [
    # (dimension, level, order, master filename)
    ("risk_aversion",  "very_averse",  1, "attanasi_master_gemma3-27b_drve_20260620_141051.csv"),
    ("risk_aversion",  "averse",       2, "attanasi_master_gemma3-27b_drav_20260621_075134.csv"),
    ("risk_aversion",  "neutral",      3, "attanasi_master_gemma3-27b_drne_20260621_221142.csv"),
    ("risk_aversion",  "seeking",      4, "attanasi_master_gemma3-27b_drse_20260622_100047.csv"),
    ("risk_aversion",  "very_seeking", 5, "attanasi_master_gemma3-27b_drve_20260622_210328.csv"),
    ("aggressiveness", "very_passive",    1, "attanasi_master_gemma3-27b_dave_20260623_053253.csv"),
    ("aggressiveness", "passive",         2, "attanasi_master_gemma3-27b_dapa_20260623_155534.csv"),
    ("aggressiveness", "moderate",        3, "attanasi_master_gemma3-27b_damo_20260624_003816.csv"),
    ("aggressiveness", "aggressive",      4, "attanasi_master_gemma3-27b_daag_20260624_091138.csv"),
    ("aggressiveness", "very_aggressive", 5, "attanasi_master_gemma3-27b_dave_20260624_201047.csv"),
]


# --------------------------------------------------------------------------- #
# Loading + condition labelling
# --------------------------------------------------------------------------- #
def condition_label(rows):
    """Derive a condition key from a master file's own columns.

    Returns a short label like 'baseline', 'refined', 'persona_grad_econ',
    'risk_very_averse', 'temp_0.7' built from whichever knobs are non-default.
    """
    r = rows[0]
    parts = []
    if (r.get("memory_type") or "baseline") == "refined":
        parts.append("refined")
    persona = (r.get("persona") or "none")
    if persona not in ("", "none", "baseline"):
        parts.append(f"persona_{persona}")
    for col, tag in (("risk_aversion_level", "risk"),
                     ("aggressiveness_level", "aggr"),
                     ("profit_orientation_level", "profit")):
        v = r.get(col)
        if v not in (None, "", "none", "neutral", "moderate", "balanced"):
            parts.append(f"{tag}_{v}")
    temp = r.get("temperature")
    if temp not in (None, "", "default"):
        try:
            if abs(float(temp) - 0.0) > 1e-9:   # treat 0.0 / blank as default
                parts.append(f"temp_{float(temp):g}")
        except ValueError:
            pass
    return "_".join(parts) if parts else "baseline"


def load_master(fn):
    """Return (rows, label) where rows = [(period, price, seed), ...] matched only."""
    raw = []
    trades = []
    with open(fn) as f:
        for r in csv.DictReader(f):
            raw.append(r)
            if str(r.get("matched", "")).strip().lower() != "true":
                continue
            try:
                trades.append((int(r["period"]), float(r["trade_price"]), r["seed"]))
            except (ValueError, KeyError):
                pass
    if not raw or not trades:
        return None, None
    return trades, condition_label(raw)


def discover(data_dir, model):
    """Map condition label -> pooled list of (period, price, seed) trade tuples."""
    pat = os.path.join(data_dir, f"attanasi_master_{model}*.csv")
    pooled = defaultdict(list)
    for fn in sorted(glob.glob(pat)):
        trades, label = load_master(fn)
        if trades:
            # namespace seeds by file so pooled runs stay distinct clusters
            tag = os.path.basename(fn)
            pooled[label].extend((p, pr, f"{tag}:{s}") for p, pr, s in trades)
    return pooled


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def per_run_period_means(trades):
    by = defaultdict(lambda: defaultdict(list))
    nt = defaultdict(lambda: defaultdict(int))
    for p, price, s in trades:
        by[s][p].append(price)
        nt[s][p] += 1
    return by, nt


def bootstrap_ci(data, n_boot=10000, alpha=0.05, seed=999):
    """Percentile bootstrap CI over the sample (matches the pipeline exactly)."""
    rng = random.Random(seed)
    n = len(data)
    bm = []
    for _ in range(n_boot):
        bm.append(sum(data[rng.randint(0, n - 1)] for _ in range(n)) / n)
    bm.sort()
    return bm[int(n_boot * alpha / 2)], bm[int(n_boot * (1 - alpha / 2))]


def clustered_ols(X, y, cl):
    """OLS with CR1 cluster-robust SEs. Returns (beta, se)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ (X.T @ y)
    u = y - X @ beta
    n, k = X.shape
    g = defaultdict(lambda: np.zeros(k))
    for i, c in enumerate(cl):
        g[c] += X[i] * u[i]
    meat = sum(np.outer(v, v) for v in g.values())
    G = len(g)
    if G < 2 or n <= k:
        return beta, np.full(k, float("nan"))   # not enough clusters for CR1 SE
    V = (G / (G - 1)) * ((n - 1) / (n - k)) * (XtXi @ meat @ XtXi)
    return beta, np.sqrt(np.diag(V))


def analyse(trades):
    """Return dict of per-period and condition-level statistics for one condition."""
    by, nt = per_run_period_means(trades)
    seeds = list(by)
    periods = sorted({p for s in seeds for p in by[s]})
    out = {"n_runs": len(seeds), "periods": {}}
    for p in periods:
        pr = [sum(by[s][p]) / len(by[s][p]) for s in seeds if p in by[s]]
        tr = [nt[s][p] for s in seeds if p in nt[s]]
        m = sum(pr) / len(pr)
        sd = math.sqrt(sum((x - m) ** 2 for x in pr) / (len(pr) - 1)) if len(pr) > 1 else 0.0
        lo, hi = bootstrap_ci(pr) if len(pr) > 1 else (m, m)
        out["periods"][p] = dict(mean=m, sd=sd, ci_lo=lo, ci_hi=hi,
                                 trades=sum(tr) / len(tr), n=len(pr))
    # slope (continuous period) + P3-P1 (dummies), clustered by run
    Xs = [[1, p] for p, _, _ in trades]
    ys = [pr for _, pr, _ in trades]
    cl = [s for _, _, s in trades]
    b, se = clustered_ols(Xs, ys, cl)
    out["slope"], out["slope_se"] = b[1], se[1]
    if set(periods) >= {1, 2, 3}:
        Xd = [[1, 1 if p == 2 else 0, 1 if p == 3 else 0] for p, _, _ in trades]
        bd, sd_ = clustered_ols(Xd, ys, cl)
        out["p3m1"], out["p3m1_se"] = bd[2], sd_[2]
    else:
        out["p3m1"], out["p3m1_se"] = float("nan"), float("nan")
    return out


# --------------------------------------------------------------------------- #
# PDF output
# --------------------------------------------------------------------------- #
def _dial_sweep_page(pdf, dim, cells, model):
    """One trajectory page for a dial sweep: 5 ordered levels + human, CE band."""
    cells = sorted(cells, key=lambda c: c[1])   # by order index
    cmap = plt.get_cmap("coolwarm")
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.axhspan(CE_LO, CE_HI, color="grey", alpha=0.18, zorder=0)
    ax.text(2.0, (CE_LO + CE_HI) / 2, f"Competitive equilibrium ({CE_LO:g}-{CE_HI:g})",
            va="center", ha="center", fontsize=10, color="grey")
    ax.plot([1, 2, 3], [HUMAN[p] for p in (1, 2, 3)], "-o", color="#111111",
            lw=3.2, ms=8, label="Humans (Table B.1)", zorder=6)
    n = len(cells)
    for i, (level, order, r) in enumerate(cells):
        ps = sorted(r["periods"])
        y = [r["periods"][p]["mean"] for p in ps]
        lo = [r["periods"][p]["mean"] - r["periods"][p]["ci_lo"] for p in ps]
        hi = [r["periods"][p]["ci_hi"] - r["periods"][p]["mean"] for p in ps]
        c = cmap(i / (n - 1)) if n > 1 else cmap(0.5)
        ax.plot(ps, y, marker="o", color=c, lw=2.0, ms=7,
                label=f"{order}. {level}", zorder=4)
        ax.errorbar(ps, y, yerr=[lo, hi], fmt="none", ecolor=c, alpha=0.5,
                    capsize=3, zorder=3.9)
    ax.set_xticks([1, 2, 3])
    ax.set_xlim(0.9, 3.2)
    ax.set_xlabel("Trading period", fontsize=12)
    ax.set_ylabel("Mean transaction price", fontsize=12)
    pretty = dim.replace("_", " ")
    ax.set_title(f"{pretty} dial sweep ({model}, {DIAL_BACKEND})\n"
                 "5 levels vs. human benchmark — within-backend comparison", fontsize=12)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9,
              framealpha=0.95, title=f"{pretty} level")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _dial_table_page(pdf, dial_results):
    """Summary table page for all dial levels (slope + P3-P1, clustered)."""
    rows = []
    for dim in ("risk_aversion", "aggressiveness"):
        for level, order, r in sorted(dial_results.get(dim, []), key=lambda c: c[1]):
            pm = r["periods"]
            def cell(p): return f"{pm[p]['mean']:.2f}" if p in pm else "-"
            p3 = (f"{r['p3m1']:+.3f} ({r['p3m1_se']:.3f})"
                  if not math.isnan(r["p3m1"]) else "n/a")
            rows.append([f"{dim.split('_')[0]}: {level}", r["n_runs"],
                         cell(1), cell(2), cell(3),
                         f"{r['slope']:+.3f} ({r['slope_se']:.3f})", p3,
                         "diverges from CE" if r["slope"] > 0 else "converges to CE"])
    rows.append(["HUMANS (Table B.1)", "-", "14.49", "14.10", "13.50",
                 "-0.495 (n/a)", "-0.990 (n/a)", "converges to CE"])
    header = ["dial: level", "n_runs", "P1", "P2", "P3",
              "slope (SE)", "P3-P1 (SE)", "direction"]
    fig, ax = plt.subplots(figsize=(11, 0.5 + 0.42 * (len(rows) + 2)))
    ax.axis("off")
    colw = [0.22, 0.07, 0.07, 0.07, 0.07, 0.16, 0.16, 0.18]
    tbl = ax.table(cellText=rows, colLabels=header, loc="center",
                   cellLoc="center", colWidths=colw)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for j in range(len(header)):
        tbl[0, j].set_facecolor("#305496")
        tbl[0, j].set_text_props(color="white", weight="bold")
    for (rr, cc), cell in tbl.get_celld().items():
        if cc == 0 and rr > 0:
            cell.set_text_props(ha="left")
            cell.PAD = 0.03
    ax.set_title(f"Behavioral-dial sweeps ({DIAL_BACKEND}): convergence slope & "
                 "net drift\n(trade-level OLS, SE clustered by run)", fontsize=11, pad=14)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def make_pdf(results, model, out_path, plot_filter=None, dial_results=None):
    # stable ordering: baseline, refined, then alphabetical
    def rank(name):
        return (0 if name == "baseline" else 1 if name == "refined" else 2, name)
    names = sorted(results, key=rank)
    plot_names = [n for n in names if (plot_filter is None or n in plot_filter)]

    cmap = plt.get_cmap("tab10")
    markers = ["s", "^", "D", "v", "P", "X", "*", "o", "<", ">"]

    with PdfPages(out_path) as pdf:
        # ---- page 1: trajectory figure ----
        fig, ax = plt.subplots(figsize=(9.5, 6))
        ax.axhspan(CE_LO, CE_HI, color="grey", alpha=0.18, zorder=0)
        ax.text(2.0, (CE_LO + CE_HI) / 2, f"Competitive equilibrium ({CE_LO:g}-{CE_HI:g})",
                va="center", ha="center", fontsize=10, color="grey")
        # human benchmark
        hp = [HUMAN[p] for p in (1, 2, 3)]
        ax.plot([1, 2, 3], hp, "-o", color="#111111", lw=3.2, ms=8,
                label="Humans (Table B.1)", zorder=6)
        for i, name in enumerate(plot_names):
            r = results[name]
            ps = sorted(r["periods"])
            y = [r["periods"][p]["mean"] for p in ps]
            lo = [r["periods"][p]["mean"] - r["periods"][p]["ci_lo"] for p in ps]
            hi = [r["periods"][p]["ci_hi"] - r["periods"][p]["mean"] for p in ps]
            c = cmap(i % 10)
            ls = "-" if name in ("baseline", "refined") else "--"
            ax.plot(ps, y, marker=markers[i % len(markers)], color=c, ls=ls,
                    lw=2.0, ms=7, label=name, zorder=4)
            ax.errorbar(ps, y, yerr=[lo, hi], fmt="none", ecolor=c, alpha=0.5,
                        capsize=3, zorder=3.9)
        ax.set_xticks([1, 2, 3])
        ax.set_xlim(0.9, 3.2)
        ax.set_xlabel("Trading period", fontsize=12)
        ax.set_ylabel("Mean transaction price", fontsize=12)
        ax.set_title(f"Price convergence: humans vs. LLM traders ({model})\n"
                     "Attanasi et al. (2021) perfect-competition market", fontsize=12)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9,
                  framealpha=0.95, title="Condition")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ---- page 2: condition-summary table ----
        fig, ax = plt.subplots(figsize=(11, 0.5 + 0.42 * (len(names) + 2)))
        ax.axis("off")
        header = ["condition", "n_runs", "P1", "P2", "P3",
                  "slope (SE)", "P3-P1 (SE)", "direction"]
        body = []
        for name in names:
            r = results[name]
            pm = r["periods"]
            def cell(p): return f"{pm[p]['mean']:.2f}" if p in pm else "-"
            slope_s = f"{r['slope']:+.3f} ({r['slope_se']:.3f})"
            p3_s = (f"{r['p3m1']:+.3f} ({r['p3m1_se']:.3f})"
                    if not math.isnan(r["p3m1"]) else "n/a")
            direction = "diverges from CE" if r["slope"] > 0 else "converges to CE"
            body.append([name, r["n_runs"], cell(1), cell(2), cell(3),
                         slope_s, p3_s, direction])
        body.append(["HUMANS (Table B.1)", "-", "14.49", "14.10", "13.50",
                     "-0.495 (n/a)", "-0.990 (n/a)", "converges to CE"])
        colw = [0.20, 0.07, 0.08, 0.08, 0.08, 0.16, 0.16, 0.17]
        tbl = ax.table(cellText=body, colLabels=header, loc="center",
                       cellLoc="center", colWidths=colw)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.5)
        for (rr, cc), cell in tbl.get_celld().items():
            if cc == 0 and rr > 0:        # left-align condition names
                cell.set_text_props(ha="left")
                cell.PAD = 0.03
        for j in range(len(header)):
            tbl[0, j].set_facecolor("#305496")
            tbl[0, j].set_text_props(color="white", weight="bold")
        ax.set_title("Condition summary: convergence slope & net drift "
                     "(trade-level OLS, SE clustered by run)", fontsize=11, pad=14)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ---- dial-sweep pages (separate backend, never mixed with above) ----
        if dial_results:
            for dim in ("risk_aversion", "aggressiveness"):
                if dial_results.get(dim):
                    _dial_sweep_page(pdf, dim, dial_results[dim], model)
            _dial_table_page(pdf, dial_results)
    return out_path


def write_csvs(results, out_dir):
    """Write the three result CSVs (period-level, condition-summary, vs-humans-wide)."""
    def rank(n):
        return (0 if n == "baseline" else 1 if n == "refined" else 2, n)
    names = sorted(results, key=rank)

    # 1. period level
    p1 = os.path.join(out_dir, "RESULTS_period_level.csv")
    with open(p1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "period", "n_runs", "mean_price", "sd_price",
                    "ci_lo", "ci_hi", "mean_trades", "human_price", "gap_vs_human"])
        for name in names:
            for p in sorted(results[name]["periods"]):
                d = results[name]["periods"][p]
                hp = HUMAN.get(p, "")
                gap = round(d["mean"] - hp, 3) if hp != "" else ""
                w.writerow([name, p, d["n"], round(d["mean"], 3), round(d["sd"], 3),
                            round(d["ci_lo"], 3), round(d["ci_hi"], 3),
                            round(d["trades"], 2), hp, gap])
        for p in (1, 2, 3):
            w.writerow(["HUMANS", p, "", HUMAN[p], "", "", "", 16, HUMAN[p], 0.0])

    # 2. condition summary
    p2 = os.path.join(out_dir, "RESULTS_condition_summary.csv")
    with open(p2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "n_runs", "slope_beta", "slope_se", "slope_t",
                    "P3_minus_P1", "P3m1_se", "P3m1_t", "slope_sig_5pct", "direction"])
        for name in names:
            r = results[name]
            t = r["slope"] / r["slope_se"]
            p3t = (r["p3m1"] / r["p3m1_se"]) if not math.isnan(r["p3m1"]) else float("nan")
            w.writerow([name, r["n_runs"], round(r["slope"], 4), round(r["slope_se"], 4),
                        round(t, 2), round(r["p3m1"], 4), round(r["p3m1_se"], 4),
                        round(p3t, 2), "yes" if abs(t) > 1.96 else "no",
                        "diverges from CE" if r["slope"] > 0 else "converges to CE"])
        w.writerow(["HUMANS", "", -0.495, "", "", -0.990, "", "", "", "converges to CE"])

    # 3. vs-humans wide (price block + gap block)
    p3 = os.path.join(out_dir, "RESULTS_vs_humans_wide.csv")
    with open(p3, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "condition", "P1", "P2", "P3", "P3_minus_P1"])
        def price(name, p):
            return round(results[name]["periods"][p]["mean"], 3) if p in results[name]["periods"] else ""
        w.writerow(["mean price", "HUMANS", HUMAN[1], HUMAN[2], HUMAN[3],
                    round(HUMAN[3] - HUMAN[1], 3)])
        for name in names:
            w.writerow(["mean price", name, price(name, 1), price(name, 2), price(name, 3),
                        round(price(name, 3) - price(name, 1), 3)])
        for name in names:
            g = {p: round(results[name]["periods"][p]["mean"] - HUMAN[p], 3)
                 for p in (1, 2, 3) if p in results[name]["periods"]}
            w.writerow(["gap vs human", name, g.get(1, ""), g.get(2, ""), g.get(3, ""),
                        round(g[3] - g[1], 3) if 1 in g and 3 in g else ""])
    return [p1, p2, p3]


def write_dial_csvs(dial_results, out_dir):
    """Write per-level dial CSVs (kept separate from the Ollama RESULTS_* files)."""
    paths = []
    pl = os.path.join(out_dir, "DIAL_period_level.csv")
    with open(pl, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dial", "level", "order", "period", "n_runs", "mean_price",
                    "sd_price", "ci_lo", "ci_hi", "mean_trades", "human_price",
                    "gap_vs_human", "backend"])
        for dim in ("risk_aversion", "aggressiveness"):
            for level, order, r in sorted(dial_results.get(dim, []), key=lambda c: c[1]):
                for p in sorted(r["periods"]):
                    d = r["periods"][p]
                    w.writerow([dim, level, order, p, d["n"], round(d["mean"], 3),
                                round(d["sd"], 3), round(d["ci_lo"], 3),
                                round(d["ci_hi"], 3), round(d["trades"], 2),
                                HUMAN[p], round(d["mean"] - HUMAN[p], 3), DIAL_BACKEND])
    paths.append(pl)

    cs = os.path.join(out_dir, "DIAL_condition_summary.csv")
    with open(cs, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dial", "level", "order", "n_runs", "slope_beta", "slope_se",
                    "slope_t", "P3_minus_P1", "P3m1_se", "slope_sig_5pct",
                    "direction", "backend"])
        for dim in ("risk_aversion", "aggressiveness"):
            for level, order, r in sorted(dial_results.get(dim, []), key=lambda c: c[1]):
                t = r["slope"] / r["slope_se"]
                w.writerow([dim, level, order, r["n_runs"], round(r["slope"], 4),
                            round(r["slope_se"], 4), round(t, 2),
                            round(r["p3m1"], 4), round(r["p3m1_se"], 4),
                            "yes" if abs(t) > 1.96 else "no",
                            "diverges from CE" if r["slope"] > 0 else "converges to CE",
                            DIAL_BACKEND])
    paths.append(cs)
    return paths


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data-dir", default=os.path.join(here, "..", "output"),
                    help="dir containing attanasi_master_*.csv")
    ap.add_argument("--model", default="gemma3-27b")
    ap.add_argument("--out-dir", default=here,
                    help="dir to write the PDF + result CSVs")
    ap.add_argument("--plot", default=None,
                    help="comma-separated condition labels to plot (default: all)")
    args = ap.parse_args()

    pooled = {}
    for label, fn in CANONICAL:
        path = os.path.join(args.data_dir, fn)
        if not os.path.exists(path):
            print(f"  [skip] missing master for '{label}': {fn}")
            continue
        trades, _ = load_master(path)
        if trades:
            pooled[label] = trades
    if not pooled:
        raise SystemExit(f"No canonical master CSVs found in {args.data_dir}")

    results = {name: analyse(tr) for name, tr in pooled.items()}
    plot_filter = (set(args.plot.split(",")) if args.plot
                   else [n for n in DEFAULT_PLOT if n in results])

    # console summary
    def rank(n):
        return (0 if n == "baseline" else 1 if n == "refined" else 2, n)
    print(f"\nModel: {args.model}   conditions found: {len(results)}")
    print(f"{'condition':26s} {'n':>3} {'slope (SE)':>16} {'P3-P1 (SE)':>16}  direction")
    print("-" * 88)
    for name in sorted(results, key=rank):
        r = results[name]
        d = "diverges from CE" if r["slope"] > 0 else "converges to CE"
        slope_s = f"{r['slope']:+.3f} ({r['slope_se']:.3f})"
        p3 = (f"{r['p3m1']:+.3f} ({r['p3m1_se']:.3f})"
              if not math.isnan(r["p3m1"]) else "n/a")
        print(f"{name:26s} {r['n_runs']:>3} {slope_s:>16} {p3:>16}  {d}")
    print(f"{'HUMANS (Table B.1)':26s} {'-':>3} {'-0.495 (n/a)':>16} "
          f"{'-0.990 (n/a)':>16}  converges to CE")

    # ---- behavioral-dial sweeps (separate backend) ----
    dial_results = {"risk_aversion": [], "aggressiveness": []}
    print(f"\nDial sweeps ({DIAL_BACKEND}) -- analysed/plotted separately from the "
          f"Ollama cells above:")
    print(f"{'dial: level':28s} {'n':>3} {'slope (SE)':>16} {'P3-P1 (SE)':>16}  direction")
    print("-" * 90)
    for dim, level, order, fn in DIAL_CELLS:
        path = os.path.join(args.data_dir, fn)
        if not os.path.exists(path):
            print(f"  [skip] missing {dim}/{level}: {fn}")
            continue
        trades, _ = load_master(path)
        # verify the file really is the single expected level (filename codes collide)
        with open(path) as f:
            levels = {row.get(dim, "") for row in csv.DictReader(f)
                      if str(row.get("matched", "")).strip().lower() == "true"}
        if levels != {level}:
            print(f"  [SKIP-CONTAMINATED] {dim}/{level}: file has levels {levels}")
            continue
        r = analyse(trades)
        dial_results[dim].append((level, order, r))
        p3 = (f"{r['p3m1']:+.3f} ({r['p3m1_se']:.3f})"
              if not math.isnan(r["p3m1"]) else "n/a")
        d = "diverges from CE" if r["slope"] > 0 else "converges to CE"
        print(f"{dim.split('_')[0]+': '+level:28s} {r['n_runs']:>3} "
              f"{r['slope']:+.3f} ({r['slope_se']:.3f})".ljust(48)
              + f"{p3:>16}  {d}")

    csvs = write_csvs(results, args.out_dir)
    csvs += write_dial_csvs(dial_results, args.out_dir)
    pdf = make_pdf(results, args.model,
                   os.path.join(args.out_dir, "convergence_analysis.pdf"),
                   plot_filter, dial_results)
    print("\nWrote:")
    for p in csvs + [pdf]:
        print("  " + p)


if __name__ == "__main__":
    main()
