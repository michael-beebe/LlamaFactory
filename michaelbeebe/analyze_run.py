#!/usr/bin/env python
"""
Analyze training run outputs for NCCL collectives and Nsight Systems profiling.

Usage:
  python michaelbeebe/analyze_run.py --run-dir michaelbeebe/outputs/qwen3_full_sft/20260117_000922

Outputs:
  - collectives_counts.png / bytes.png / size_heatmap.png (when data available)
  - summary printed to stdout
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore
import numpy as np

# Optional import
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


COLLECTIVE_REGEX = re.compile(
    r"AllReduce|AllGather|ReduceScatter|Broadcast|AllToAll|\bReduce\b|Gather|Scatter",
    re.IGNORECASE,
)
SIZE_REGEX = re.compile(r"size\s+(\d+)", re.IGNORECASE)


def parse_nccl_logs(
    logs: List[Path],
) -> Tuple[Dict[str, int], Dict[str, int], List[Tuple[str, int]]]:
    counts: Dict[str, int] = {}
    bytes_per_coll: Dict[str, int] = {}
    sizes: List[Tuple[str, int]] = []
    for log in logs:
        try:
            with log.open() as f:
                for line in f:
                    m_coll = COLLECTIVE_REGEX.search(line)
                    m_size = SIZE_REGEX.search(line)
                    if m_coll:
                        coll = m_coll.group(0)
                        coll_norm = coll.capitalize() if coll.islower() else coll
                        counts[coll_norm] = counts.get(coll_norm, 0) + 1
                        if m_size:
                            sz = int(m_size.group(1))
                            bytes_per_coll[coll_norm] = (
                                bytes_per_coll.get(coll_norm, 0) + sz
                            )
                            sizes.append((coll_norm, sz))
        except FileNotFoundError:
            continue
    return counts, bytes_per_coll, sizes


def run_nsys_stats(nsys_rep: Path) -> str:
    cmd = ["nsys", "stats", "-r", "nvtx_sum", str(nsys_rep)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"nsys stats failed: {proc.stderr}\n{proc.stdout}")
    return proc.stdout


def parse_nsys_nvtx_sum(output: str) -> List[Dict[str, str]]:
    rows = []
    lines = output.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Time (%)") and "Range" in line:
            header_idx = i
            break
    if header_idx is None:
        return rows
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            continue
        if line.strip().startswith("**"):
            break
        parts = line.split()
        if len(parts) < 9:
            continue
        # last column is Range (name)
        name = parts[-1]
        try:
            time_pct = float(parts[0])
            total_ns = int(parts[1])
            instances = int(parts[2])
        except ValueError:
            continue
        rows.append(
            {
                "name": name,
                "time_pct": time_pct,
                "total_ns": total_ns,
                "instances": instances,
                "total_s": total_ns / 1e9,
            }
        )
    return rows


def plot_counts(counts: Dict[str, int], out: Path, title: str):
    if not counts or plt is None:
        return
    coll, cnt = zip(*sorted(counts.items(), key=lambda kv: kv[0]))
    plt.figure(figsize=(8, 4))
    plt.bar(coll, cnt)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_bytes(bytes_per_coll: Dict[str, int], out: Path, title: str):
    if not bytes_per_coll or plt is None:
        return
    coll, b = zip(*sorted(bytes_per_coll.items(), key=lambda kv: kv[0]))
    mb = [x / 1024 / 1024 for x in b]
    plt.figure(figsize=(8, 4))
    plt.bar(coll, mb)
    plt.ylabel("MB")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_size_heatmap(logs: List[Path], out: Path, title: str):
    if plt is None:
        return
    # Best-effort: collect (coll, size) pairs
    sizes = []
    colls = []
    for log in logs:
        try:
            with log.open() as f:
                for line in f:
                    m_coll = COLLECTIVE_REGEX.search(line)
                    m_size = SIZE_REGEX.search(line)
                    if m_coll and m_size:
                        colls.append(m_coll.group(0))
                        sizes.append(int(m_size.group(1)))
        except FileNotFoundError:
            continue
    if not sizes:
        return
    # bucket sizes log2
    buckets = np.log2(np.array(sizes) + 1)
    bucket_bins = np.arange(0, np.ceil(buckets.max()) + 1)
    coll_set = sorted(set(colls))
    coll_index = {c: i for i, c in enumerate(coll_set)}
    heat = np.zeros((len(coll_set), len(bucket_bins)), dtype=int)
    for c, b in zip(colls, buckets):
        bi = int(np.floor(b))
        heat[coll_index[c], bi] += 1
    plt.figure(figsize=(10, 4 + len(coll_set) * 0.2))
    plt.imshow(heat, aspect="auto", cmap="viridis")
    plt.yticks(range(len(coll_set)), coll_set)
    plt.xticks(
        range(len(bucket_bins)),
        [f"2^{i}" for i in range(len(bucket_bins))],
        rotation=45,
        ha="right",
    )
    plt.colorbar(label="count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True, help="Run output directory")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    # NCCL logs
    nccl_logs = list(run_dir.glob("nccl_*.log"))
    nccl_counts, nccl_bytes, nccl_sizes = parse_nccl_logs(nccl_logs)

    # Nsight report
    nsys_rep = next(iter(run_dir.glob("*.nsys-rep")), None)
    nsys_rows = []
    if nsys_rep:
        try:
            out = run_nsys_stats(nsys_rep)
            nsys_rows = parse_nsys_nvtx_sum(out)
        except Exception as e:
            print(f"Warning: nsys stats failed: {e}", file=sys.stderr)

    # Aggregate counts from nsys if present
    nsys_counts = {
        r["name"]: r["instances"] for r in nsys_rows if r["name"].startswith("NCCL")
    }

    # Print summary
    print("\n=== NCCL Log Summary ===")
    if nccl_counts:
        for k, v in sorted(nccl_counts.items()):
            print(f"{k}: {v}")
    else:
        print("No NCCL log collective lines parsed.")

    print("\n=== Nsight NVTX Summary (collectives) ===")
    if nsys_counts:
        for k, v in sorted(nsys_counts.items()):
            row = next(r for r in nsys_rows if r["name"] == k)
            print(f"{k}: {v} instances, {row['total_s']:.3f}s")
    else:
        print("No NCCL NVTX ranges found.")

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Plots
    plot_counts(
        nccl_counts or nsys_counts,
        figures_dir / "collectives_counts.png",
        "Collectives Counts",
    )
    plot_bytes(
        nccl_bytes,
        figures_dir / "collectives_bytes.png",
        "Collectives Bytes (MB)",
    )
    plot_size_heatmap(
        nccl_logs,
        figures_dir / "collectives_size_heatmap.png",
        "Collectives Size Heatmap (log2 bytes)",
    )

    if not nccl_sizes:
        print(
            "\n[!] No message sizes parsed from NCCL logs. Ensure NCCL_DEBUG=INFO and NCCL_DEBUG_SUBSYS=COLL, and that NCCL logs include 'size'."
        )

    print("\nPlots (if data available) saved under:", figures_dir)


if __name__ == "__main__":
    main()
