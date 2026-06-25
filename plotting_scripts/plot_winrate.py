#!/usr/bin/env python3
"""Plot win-rate (and mean_p) vs training step from a run's winrate_history.jsonl.

The in-loop eval in es_lora_multinode.py appends one JSON line per eval point:
  {"es_step": N, "win_rate": .., "mean_p": .., "n": ..}   (percentages out of 100)

The output PNG goes to ../outputs/ (gitignored) by default.

Usage (the erml env has matplotlib):
  /workspace/miniconda/envs/erml/bin/python plotting_scripts/plot_winrate.py <run_dir-or-glob> [--out PATH]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "outputs")

ap = argparse.ArgumentParser()
ap.add_argument("run", help="run dir (or glob) containing winrate_history.jsonl")
ap.add_argument("--out", default=None, help="output PNG (default: ../outputs/<run>_winrate.png)")
args = ap.parse_args()

matches = sorted(glob.glob(os.path.expandvars(args.run)))
run_dir = (matches[-1] if matches else args.run).rstrip("/")
hist_path = os.path.join(run_dir, "winrate_history.jsonl")
if not os.path.isfile(hist_path):
    raise SystemExit(f"no winrate_history.jsonl in {run_dir}")

rows = sorted((json.loads(l) for l in open(hist_path) if l.strip()), key=lambda r: r["es_step"])
if not rows:
    raise SystemExit(f"{hist_path} is empty")
steps = [r["es_step"] for r in rows]
win = [r["win_rate"] for r in rows]
mp = [r["mean_p"] for r in rows]
n = rows[-1].get("n", 0)

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(steps, win, "-o", color="#1a73e8", label="win_rate", zorder=3)
ax.plot(steps, mp, "-s", color="#34a853", label="mean_p", alpha=0.85, zorder=3)
ax.axhline(50, ls="--", color="#d93025", lw=1.2, label="50 = tie with init")
ax.set_xlabel("ES training step")
ax.set_ylabel("Preferred vs initial model (out of 100)")
ax.set_title(f"Win-rate vs init over training (n={n}/eval)\n{os.path.basename(run_dir)[:62]}", fontsize=10)
ax.set_ylim(0, 100)
ax.legend(frameon=False)
ax.grid(ls=":", alpha=0.4, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()

out = args.out or os.path.join(OUTPUTS_DIR, f"{os.path.basename(run_dir)[:60]}_winrate.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=150)
print("saved", out)
