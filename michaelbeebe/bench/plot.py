# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Plot stock-NCCL vs MSCCL++ benchmark results.

Reads <run-dir>/results.json (produced by parse.py) and writes:
  - step_time_violin.png      Distribution of per-step training times
  - step_time_series.png      Per-step time vs step number (both runs overlaid)
  - throughput_bar.png        Median tokens/sec/GPU bar comparison
  - collectives_breakdown.png Stacked-bar count of native vs fallback per algo
  - loss_curves.png           Loss vs step (both runs) for sanity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {"nccl_baseline": "#888888", "mscclpp": "#0078D4"}
LABELS = {"nccl_baseline": "Stock NCCL", "mscclpp": "MSCCL++ (TorchComms)"}


def per_step_dts(run):
    return [r["dt_sec"] for r in run["step_metrics"]
            if r.get("dt_sec") is not None]


def per_step_dts_post_warmup(run, warmup):
    return [r["dt_sec"] for r in run["step_metrics"]
            if r.get("dt_sec") is not None and r["step"] > warmup]


def step_loss_pairs(run):
    return [(r["step"], r["loss"]) for r in run["step_metrics"]]


def plot_step_time_violin(run_dir: Path, results: dict):
    runs = results["runs"]
    warmup = results["warmup"]
    data, labels, colors = [], [], []
    for k in ("nccl_baseline", "mscclpp"):
        if k not in runs:
            continue
        dts_ms = [d * 1000 for d in per_step_dts_post_warmup(runs[k], warmup)]
        if not dts_ms:
            continue
        data.append(dts_ms)
        labels.append(LABELS[k])
        colors.append(PALETTE[k])
    if not data:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.6)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Per-step training time (ms)")
    ax.set_title(f"Step-time distribution (warmup={warmup} excluded)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "step_time_violin.png", dpi=150)
    plt.close(fig)


def plot_step_time_series(run_dir: Path, results: dict):
    runs = results["runs"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k in ("nccl_baseline", "mscclpp"):
        if k not in runs:
            continue
        rows = [r for r in runs[k]["step_metrics"] if r.get("dt_sec") is not None]
        if not rows:
            continue
        xs = [r["step"] for r in rows]
        ys = [r["dt_sec"] * 1000 for r in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=LABELS[k], color=PALETTE[k])
    ax.set_xlabel("Step")
    ax.set_ylabel("Step time (ms)")
    ax.set_title("Per-step training time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "step_time_series.png", dpi=150)
    plt.close(fig)


def plot_throughput_bar(run_dir: Path, results: dict):
    """Throughput bar: median step time → relative speedup."""
    runs = results["runs"]
    if "nccl_baseline" not in runs or "mscclpp" not in runs:
        return
    a = runs["nccl_baseline"]["timing_summary"]
    b = runs["mscclpp"]["timing_summary"]
    if not (a.get("n") and b.get("n")):
        return

    # Relative throughput (1.0 = baseline).
    rel = a["median_sec"] / b["median_sec"] if b["median_sec"] > 0 else 0
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(
        [LABELS["nccl_baseline"], LABELS["mscclpp"]],
        [a["median_sec"] * 1000, b["median_sec"] * 1000],
        color=[PALETTE["nccl_baseline"], PALETTE["mscclpp"]],
        yerr=[
            [(a["median_sec"] - a["p10_sec"]) * 1000, (b["median_sec"] - b["p10_sec"]) * 1000],
            [(a["p90_sec"] - a["median_sec"]) * 1000, (b["p90_sec"] - b["median_sec"]) * 1000],
        ],
        capsize=8,
    )
    for bar, val in zip(bars, [a["median_sec"] * 1000, b["median_sec"] * 1000]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.0f} ms",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Median step time (ms)")
    delta_pct = (b["median_sec"] - a["median_sec"]) / a["median_sec"] * 100
    ax.set_title(f"Median step time  (MSCCL++ vs NCCL: {delta_pct:+.1f}%, rel-speed={rel:.2f}x)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "throughput_bar.png", dpi=150)
    plt.close(fig)


def plot_collectives_breakdown(run_dir: Path, results: dict):
    """Stacked-bar: count of collective ops on rank 0 by category, per run."""
    runs = results["runs"]
    if not runs:
        return
    # Aggregate categories: native algos (named) + fallback ops (named).
    all_categories = set()
    per_run_counts = {}
    for k in ("nccl_baseline", "mscclpp"):
        if k not in runs:
            continue
        s = runs[k]["collectives"]["summary"]
        d = {}
        for algo_key, vals in s["by_algo"].items():
            # algo_key: "collective|algo_name|TYPE"
            parts = algo_key.split("|")
            label = f"{parts[0]} ({parts[1]})"
            d[label] = d.get(label, 0) + vals["count"]
        for op, vals in s["by_fallback_op"].items():
            label = f"{op} (NCCL fallback)"
            d[label] = d.get(label, 0) + vals["count"]
        per_run_counts[k] = d
        all_categories.update(d.keys())

    if not per_run_counts:
        return

    cats = sorted(all_categories)
    runs_present = list(per_run_counts.keys())
    x = np.arange(len(runs_present))
    width = 0.7

    # Color: blue shades for MSCCL++ native, orange for fallback.
    colors = []
    for c in cats:
        if "fallback" in c.lower():
            colors.append("#D83B01")
        elif "fullmesh2" in c.lower():
            colors.append("#0078D4")
        elif "fullmesh" in c.lower():
            colors.append("#106EBE")
        elif "packet" in c.lower():
            colors.append("#005A9E")
        else:
            colors.append("#666666")

    fig, ax = plt.subplots(figsize=(8, 5))
    bottoms = np.zeros(len(runs_present))
    for cat, color in zip(cats, colors):
        heights = np.array([per_run_counts[r].get(cat, 0) for r in runs_present])
        ax.bar(x, heights, width=width, bottom=bottoms, label=cat, color=color)
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[r] for r in runs_present])
    ax.set_ylabel("Collective op count (rank 0)")
    ax.set_title("Collective dispatch breakdown by algorithm")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "collectives_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(run_dir: Path, results: dict):
    runs = results["runs"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k in ("nccl_baseline", "mscclpp"):
        if k not in runs:
            continue
        pairs = step_loss_pairs(runs[k])
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=LABELS[k], color=PALETTE[k])
    ax.set_xlabel("Step")
    ax.set_ylabel("Training loss")
    ax.set_title("Loss curve (sanity check — should overlap closely)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curves.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    args = p.parse_args()
    results = json.loads((args.run_dir / "results.json").read_text())

    plot_step_time_violin(args.run_dir, results)
    plot_step_time_series(args.run_dir, results)
    plot_throughput_bar(args.run_dir, results)
    plot_collectives_breakdown(args.run_dir, results)
    plot_loss_curves(args.run_dir, results)
    print(f"  wrote PNGs under {args.run_dir}")


if __name__ == "__main__":
    main()
