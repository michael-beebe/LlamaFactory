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

# Default to 300dpi so plots stay readable when rendered by terminal image
# viewers (chafa, viu, timg) — at 150dpi the matplotlib labels collapse into
# illegible mush at any reasonable cell size. Overridable via --dpi.
DEFAULT_DPI = 300

PALETTE = {
    "nccl_baseline": "#888888",
    "nccl_torchcomms": "#D29922",  # amber — TorchComms control
    "mscclpp": "#0078D4",
}
LABELS = {
    "nccl_baseline": "Stock NCCL",
    "nccl_torchcomms": "NCCL (TorchComms)",
    "mscclpp": "MSCCL++ (TorchComms)",
}
RUN_ORDER = ("nccl_baseline", "nccl_torchcomms", "mscclpp")


def per_step_dts(run):
    return [r["dt_sec"] for r in run["step_metrics"]
            if r.get("dt_sec") is not None]


def per_step_dts_post_warmup(run, warmup):
    return [r["dt_sec"] for r in run["step_metrics"]
            if r.get("dt_sec") is not None and r["step"] > warmup]


def step_loss_pairs(run):
    return [(r["step"], r["loss"]) for r in run["step_metrics"]]


def plot_step_time_violin(run_dir: Path, results: dict, dpi: int):
    runs = results["runs"]
    warmup = results["warmup"]
    data, labels, colors = [], [], []
    for k in RUN_ORDER:
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
    fig.savefig(run_dir / "step_time_violin.png", dpi=dpi)
    plt.close(fig)


def plot_step_time_series(run_dir: Path, results: dict, dpi: int):
    runs = results["runs"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k in RUN_ORDER:
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
    fig.savefig(run_dir / "step_time_series.png", dpi=dpi)
    plt.close(fig)


def plot_throughput_bar(run_dir: Path, results: dict, dpi: int):
    """Throughput bar: median step time across all runs present + pairwise deltas."""
    runs = results["runs"]
    present = [k for k in RUN_ORDER if k in runs and runs[k]["timing_summary"].get("n")]
    if len(present) < 2:
        return

    medians_ms = [runs[k]["timing_summary"]["median_sec"] * 1000 for k in present]
    p10_err = [
        (runs[k]["timing_summary"]["median_sec"] - runs[k]["timing_summary"]["p10_sec"]) * 1000
        for k in present
    ]
    p90_err = [
        (runs[k]["timing_summary"]["p90_sec"] - runs[k]["timing_summary"]["median_sec"]) * 1000
        for k in present
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(
        [LABELS[k] for k in present],
        medians_ms,
        color=[PALETTE[k] for k in present],
        yerr=[p10_err, p90_err],
        capsize=8,
    )
    for bar, val in zip(bars, medians_ms):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.0f} ms",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Median step time (ms)")

    # Title: most informative pairwise delta = vs baseline if available.
    if "nccl_baseline" in present:
        a = runs["nccl_baseline"]["timing_summary"]
        deltas = []
        for k in present:
            if k == "nccl_baseline":
                continue
            b = runs[k]["timing_summary"]
            d = (b["median_sec"] - a["median_sec"]) / a["median_sec"] * 100
            rel = a["median_sec"] / b["median_sec"] if b["median_sec"] > 0 else 0
            deltas.append(f"{LABELS[k]}: {d:+.1f}% ({rel:.2f}x)")
        ax.set_title("Median step time vs stock NCCL — " + "  |  ".join(deltas))
    else:
        ax.set_title("Median step time")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "throughput_bar.png", dpi=dpi)
    plt.close(fig)


def plot_collectives_breakdown(run_dir: Path, results: dict, dpi: int):
    """Stacked-bar: per-rank-0 collective dispatch counts, broken out by
    (collective, algorithm) bucket, for every backend in the run.

    Three data sources feed this plot:

      1. ``collectives.summary.by_algo`` — MSCCL++ native algorithms
         (e.g. ``default_allgather_fullmesh``), extracted from the
         ``[MSCCLPP]`` trace lines our backend emits at TRACE>=1. Only
         the mscclpp run produces these.
      2. ``collectives.summary.by_fallback_op`` — calls that went through
         our NcclFallback dispatcher (e.g. ``reduce_scatter``). Only the
         mscclpp run produces these.
      3. ``nccl_algos.per_op`` — NCCL ring/tree algorithm + protocol
         choices captured from NCCL_DEBUG=INFO logs. Every backend that
         used libnccl produces this (nccl_baseline, nccl_torchcomms, and
         mscclpp's fallback path).

    Previously this plot only consumed #1+#2, so the NCCL bars showed
    as zero (no MSCCL++ trace lines emitted) even though those backends
    were doing 18000+ collectives. Now all three sources are merged,
    op names are normalized (e.g. ``AllGather`` <-> ``allgather``), and
    each (op, algo) bucket gets one stack segment so the bars are
    directly comparable across backends.
    """
    runs = results["runs"]
    if not runs:
        return

    def _normalize_op(name: str) -> str:
        """Make NCCL CamelCase and MSCCL++ lowercase op names line up."""
        return name.lower().replace("_", "")

    # Per-run map of (op, algo_label) -> count, with op names normalized so
    # mscclpp's 'allgather' lines up with NCCL's 'AllGather'.
    per_run_counts: dict[str, dict[tuple[str, str], int]] = {}
    all_buckets: set[tuple[str, str]] = set()

    for k in RUN_ORDER:
        if k not in runs:
            continue
        d: dict[tuple[str, str], int] = {}
        cs = runs[k]["collectives"]["summary"]

        # Source 1: MSCCL++ native algorithm dispatches.
        for algo_key, vals in cs.get("by_algo", {}).items():
            # algo_key format: "collective|algo_name|TYPE"
            op_raw, algo_name, _atype = algo_key.split("|")
            label = f"{_normalize_op(op_raw)} · MSCCL++ {algo_name.replace('default_', '')}"
            d[(_normalize_op(op_raw), label)] = d.get((_normalize_op(op_raw), label), 0) + vals["count"]

        # Source 2: MSCCL++ -> NCCL fallback dispatches (op-level only,
        # algo/proto comes from source 3 below).
        for op_raw, vals in cs.get("by_fallback_op", {}).items():
            label = f"{_normalize_op(op_raw)} · NCCL fallback"
            d[(_normalize_op(op_raw), label)] = d.get((_normalize_op(op_raw), label), 0) + vals["count"]

        # Source 3: NCCL algorithm+protocol choices (any backend that used
        # libnccl, including mscclpp's fallback path).
        #
        # IMPORTANT: NCCL only populates one of the NCCL_DEBUG_FILE files
        # in practice — the rank that the global NCCL state is owned by —
        # so the per-op counts in nccl_algos.per_op are already rank-0-only,
        # NOT all-ranks-aggregated. (Verified: 1 of 8 per-rank NCCL log
        # files contains real data, the other 7 only carry the version
        # banner.) So we use the counts as-is, no normalization.
        #
        # For nccl_baseline / nccl_torchcomms this is the only source.
        # For mscclpp these calls were ALREADY counted as fallbacks in
        # source 2 above (the [NcclFallback] trace lines our backend
        # emits), so we'd be double-counting them. Skip those ops for the
        # mscclpp backend to avoid the duplicate.
        nccl_algos = runs[k].get("nccl_algos", {})
        mscclpp_fallback_ops = {_normalize_op(o) for o in cs.get("by_fallback_op", {})}
        for op_raw, entries in nccl_algos.get("per_op", {}).items():
            op = _normalize_op(op_raw)
            if k == "mscclpp" and op in mscclpp_fallback_ops:
                continue  # already counted by source 2 above
            for entry in entries:
                label = f"{op} · NCCL {entry['algo']}/{entry['proto']}"
                d[(op, label)] = d.get((op, label), 0) + entry["count"]

        if d:
            per_run_counts[k] = d
            all_buckets.update(d.keys())

    if not per_run_counts:
        return

    # Sort buckets by (normalized op, label) so legend groups by operation.
    buckets = sorted(all_buckets)
    runs_present = [k for k in RUN_ORDER if k in per_run_counts]
    x = np.arange(len(runs_present))
    width = 0.6

    # Color scheme:
    #   - allgather   : blue family
    #   - allreduce   : green family
    #   - reducescatter: orange family
    #   - other       : gray
    # Within each family, MSCCL++ native = darker shade, NCCL = lighter,
    # NCCL fallback = warmest accent.
    family_palettes = {
        "allgather":      ("#0D47A1", "#1976D2", "#42A5F5", "#90CAF9"),
        "allreduce":      ("#1B5E20", "#388E3C", "#66BB6A", "#A5D6A7"),
        "reducescatter":  ("#E65100", "#F57C00", "#FB8C00", "#FFB74D"),
    }
    fallback_color = "#D83B01"

    bucket_colors: list[str] = []
    family_seen: dict[str, int] = {}
    for op, label in buckets:
        if "fallback" in label.lower():
            bucket_colors.append(fallback_color)
            continue
        palette = family_palettes.get(op, ("#666666", "#888888", "#AAAAAA", "#CCCCCC"))
        idx = family_seen.get(op, 0)
        bucket_colors.append(palette[min(idx, len(palette) - 1)])
        family_seen[op] = idx + 1

    fig, ax = plt.subplots(figsize=(10, 6))
    bottoms = np.zeros(len(runs_present))
    for (op, label), color in zip(buckets, bucket_colors):
        heights = np.array([per_run_counts[r].get((op, label), 0) for r in runs_present])
        if heights.sum() == 0:
            continue
        ax.bar(x, heights, width=width, bottom=bottoms, label=label, color=color, edgecolor="white", linewidth=0.3)
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[r] for r in runs_present])
    ax.set_ylabel("Collective op count (rank 0)")
    ax.set_title("Collective dispatch breakdown by algorithm")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=True)
    ax.grid(axis="y", alpha=0.3)
    ax.margins(x=0.15)  # extra horizontal padding so x-tick labels don't crowd bars
    fig.tight_layout()
    fig.savefig(run_dir / "collectives_breakdown.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(run_dir: Path, results: dict, dpi: int):
    runs = results["runs"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k in RUN_ORDER:
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
    fig.savefig(run_dir / "loss_curves.png", dpi=dpi)
    plt.close(fig)


def plot_nccl_algos(run_dir: Path, results: dict, dpi: int):
    """Stacked-bar of NCCL algorithm/protocol usage per backend.

    For each run that has NCCL log data (i.e. every run, since we leave
    NCCL_DEBUG on across the board), show how many dispatched collectives
    landed in each (Algo/Proto) bucket, faceted by op.
    """
    runs = results["runs"]
    # Gather every (run -> op -> algo_key -> count) datum so we can build a
    # consistent legend.
    data = {}  # run_label -> {op: {algo_key: count}}
    all_ops = set()
    all_algos = set()
    for label in RUN_ORDER:
        if label not in runs:
            continue
        algos = runs[label].get("nccl_algos", {})
        if not algos.get("total_calls"):
            continue
        d = {}
        for op, entries in algos["per_op"].items():
            d[op] = {f"{e['algo']}/{e['proto']}": e["count"] for e in entries}
            all_ops.add(op)
            all_algos.update(d[op].keys())
        data[label] = d
    if not data:
        return

    ops = sorted(all_ops)
    runs_present = [k for k in RUN_ORDER if k in data]
    algos = sorted(all_algos)

    # Color per algo/proto: derive a stable color from a categorical map.
    cmap = plt.get_cmap("tab20")
    algo_colors = {a: cmap(i % 20) for i, a in enumerate(algos)}

    fig, axes = plt.subplots(
        1, len(ops),
        figsize=(max(4 * len(ops), 6), 5),
        sharey=False,
        squeeze=False,
    )
    axes = axes[0]
    for ax, op in zip(axes, ops):
        x = np.arange(len(runs_present))
        bottoms = np.zeros(len(runs_present))
        for algo in algos:
            heights = np.array([data[r].get(op, {}).get(algo, 0) for r in runs_present])
            if heights.sum() == 0:
                continue
            ax.bar(x, heights, bottom=bottoms, label=algo, color=algo_colors[algo], width=0.7)
            bottoms += heights
        ax.set_title(op, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[r] for r in runs_present], rotation=20, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("Calls" if ax is axes[0] else "")

    # Single legend across all subplots, deduplicated.
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                labels.append(l)
                handles.append(h)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(labels), 5), fontsize=9, title="Algo/Proto")
    fig.suptitle("NCCL algorithm/protocol selection per backend", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(run_dir / "nccl_algos.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Output PNG DPI (default {DEFAULT_DPI}; bump higher if you need "
        "tiny axis labels to stay legible under terminal image renderers).",
    )
    args = p.parse_args()
    results = json.loads((args.run_dir / "results.json").read_text())

    plot_step_time_violin(args.run_dir, results, args.dpi)
    plot_step_time_series(args.run_dir, results, args.dpi)
    plot_throughput_bar(args.run_dir, results, args.dpi)
    plot_collectives_breakdown(args.run_dir, results, args.dpi)
    plot_loss_curves(args.run_dir, results, args.dpi)
    plot_nccl_algos(args.run_dir, results, args.dpi)
    print(f"  wrote PNGs under {args.run_dir} (dpi={args.dpi})")


if __name__ == "__main__":
    main()
