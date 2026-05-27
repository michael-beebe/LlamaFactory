# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Parse stock-NCCL vs NCCL-via-TorchComms vs MSCCL++ benchmark logs into a
single results.json.

For each run (nccl_baseline/, nccl_torchcomms/, mscclpp/) extracts:
  - Per-step elapsed time (derived from logging-callback timestamps).
  - Per-rank collective dispatch counts and bytes, broken out by:
      - native MSCCL++ algorithms (logged as "[MSCCLPP] rank=... algo=...")
      - NCCL fallback path        (logged as "[NcclFallback] op -> NCCL ...")
  - Loss values per step (sanity check that runs trained equivalently).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

LOG_LINE_RE = re.compile(
    r"\[INFO\|(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\][^>]*>>\s*"
    r"epoch:\s*\d+,\s*step:\s*(?P<step>\d+),\s*loss:\s*(?P<loss>[-\d.]+),"
    r"\s*grad_norm:\s*(?P<gn>[-\d.]+)"
)
MSCCLPP_RE = re.compile(
    r"\[MSCCLPP\]\s+rank=(?P<rank>\d+)\s+collective=(?P<col>\w+)\s+"
    r"bytes=(?P<bytes>\d+)\s+dtype=(?P<dt>\d+)\s+->\s+algo='(?P<algo>[^']+)'\s+"
    r"\((?P<atype>NATIVE|DSL)\)"
)
NCCL_RE = re.compile(
    r"\[NcclFallback\]\s+(?P<op>\w+)\s+->\s+NCCL\s+(?P<rest>.*)"
)
NCCL_FIELDS_RE = re.compile(r"(\w+)=(\d+)")


def parse_step_times(stdout_path: Path, jsonl_path: Path | None = None) -> list[dict]:
    """Extract per-step timing rows.

    Prefers ``step_timings.jsonl`` (perf_counter precision) when present;
    falls back to parsing 1-second-resolution timestamps from the LlamaFactory
    log line in ``stdout.log``.
    """
    rows: list[dict] = []
    if jsonl_path and jsonl_path.exists():
        for line in jsonl_path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        rows.sort(key=lambda r: r["step"])
        return rows

    if not stdout_path.exists():
        return rows
    for line in stdout_path.read_text(errors="ignore").splitlines():
        m = LOG_LINE_RE.search(line)
        if m:
            rows.append({
                "step": int(m["step"]),
                "loss": float(m["loss"]),
                "grad_norm": float(m["gn"]),
                "ts": datetime.strptime(m["ts"], "%Y-%m-%d %H:%M:%S"),
            })
    rows.sort(key=lambda r: r["step"])
    for prev, cur in zip(rows, rows[1:]):
        cur["dt_sec"] = (cur["ts"] - prev["ts"]).total_seconds()
    if rows:
        rows[0]["dt_sec"] = None
    for r in rows:
        if isinstance(r.get("ts"), datetime):
            r["ts"] = r["ts"].isoformat()
    return rows


def parse_collectives(stderr_dir: Path) -> dict:
    """Aggregate per-rank collective dispatch tallies across all rank logs."""
    per_rank = defaultdict(lambda: {
        "native": defaultdict(lambda: {"count": 0, "bytes": 0}),
        "fallback": defaultdict(lambda: {"count": 0, "bytes": 0}),
    })
    if not stderr_dir.exists():
        return {"per_rank": {}, "summary": {}}

    for log in sorted(stderr_dir.rglob("stderr.log")):
        # Rank dir is the parent of stderr.log (e.g. .../attempt_0/3/stderr.log).
        try:
            rank = int(log.parent.name)
        except ValueError:
            continue
        for line in log.read_text(errors="ignore").splitlines():
            mm = MSCCLPP_RE.search(line)
            if mm:
                key = f"{mm['col']}|{mm['algo']}|{mm['atype']}"
                bucket = per_rank[rank]["native"][key]
                bucket["count"] += 1
                bucket["bytes"] += int(mm["bytes"])
                continue
            mn = NCCL_RE.search(line)
            if mn:
                op = mn["op"]
                fields = dict(NCCL_FIELDS_RE.findall(mn["rest"]))
                # Prefer recvCount, fall back to count or sendCount.
                count_field = fields.get("recvCount") or fields.get("count") or fields.get("sendCount") or "0"
                # Bytes unknown without dtype mapping; keep 0 here, count is sufficient.
                bucket = per_rank[rank]["fallback"][op]
                bucket["count"] += 1
                bucket["bytes"] += int(count_field)  # element count (not bytes), good for ratios

    # Summary across rank 0 (representative).
    summary = {"native_total": 0, "fallback_total": 0, "by_algo": {}, "by_fallback_op": {}}
    if 0 in per_rank:
        for k, v in per_rank[0]["native"].items():
            summary["by_algo"][k] = v
            summary["native_total"] += v["count"]
        for k, v in per_rank[0]["fallback"].items():
            summary["by_fallback_op"][k] = v
            summary["fallback_total"] += v["count"]

    # Convert defaultdicts to plain dicts for JSON
    plain = {
        rank: {
            "native": dict(d["native"]),
            "fallback": dict(d["fallback"]),
        }
        for rank, d in per_rank.items()
    }
    return {"per_rank": plain, "summary": summary}


def parse_run(run_dir: Path) -> dict:
    return {
        "step_metrics": parse_step_times(
            run_dir / "stdout.log",
            run_dir / "step_timings.jsonl",
        ),
        "collectives": parse_collectives(run_dir / "per_rank"),
    }


def summarize_step_times(rows: list[dict], warmup: int) -> dict:
    dts = [r["dt_sec"] for r in rows if r.get("dt_sec") is not None and r["step"] > warmup]
    if not dts:
        return {"n": 0}
    dts_sorted = sorted(dts)
    n = len(dts_sorted)
    def pct(p):
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return dts_sorted[idx]
    return {
        "n": n,
        "warmup_excluded": warmup,
        "mean_sec": sum(dts) / n,
        "median_sec": pct(0.5),
        "p10_sec": pct(0.1),
        "p90_sec": pct(0.9),
        "min_sec": min(dts),
        "max_sec": max(dts),
        "total_sec": sum(dts),
    }


RUN_LABELS = ("nccl_baseline", "nccl_torchcomms", "mscclpp")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args()

    results = {"warmup": args.warmup, "runs": {}}
    for label in RUN_LABELS:
        run_dir = args.run_dir / label
        if not run_dir.exists():
            print(f"  SKIP {label}: missing")
            continue
        run_data = parse_run(run_dir)
        run_data["timing_summary"] = summarize_step_times(run_data["step_metrics"], args.warmup)
        results["runs"][label] = run_data
        ts = run_data["timing_summary"]
        cs = run_data["collectives"]["summary"]
        print(f"  {label}:")
        if ts.get("n"):
            print(f"    steps measured (post-warmup={args.warmup}): {ts['n']}")
            print(f"    step time : median={ts['median_sec']*1000:.1f}ms "
                  f"p10={ts['p10_sec']*1000:.1f}ms p90={ts['p90_sec']*1000:.1f}ms "
                  f"mean={ts['mean_sec']*1000:.1f}ms")
        print(f"    rank0 collectives: native={cs['native_total']} fallback={cs['fallback_total']}")

    out = args.run_dir / "results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  wrote {out}")

    # Pairwise step-time deltas across whichever runs are present.
    # baseline ↔ each other run (negative = faster than baseline).
    def _delta(a_label: str, b_label: str) -> None:
        runs = results["runs"]
        if a_label not in runs or b_label not in runs:
            return
        a = runs[a_label]["timing_summary"]
        b = runs[b_label]["timing_summary"]
        if not (a.get("n") and b.get("n")):
            return
        delta = (b["median_sec"] - a["median_sec"]) / a["median_sec"] * 100
        speedup = a["median_sec"] / b["median_sec"] if b["median_sec"] > 0 else float("nan")
        print(f"  median step time: {b_label} vs {a_label}: {delta:+.2f}%  (rel-speed={speedup:.3f}x)")

    print()
    _delta("nccl_baseline", "nccl_torchcomms")  # TorchComms shim cost
    _delta("nccl_torchcomms", "mscclpp")        # MSCCL++ algo benefit (apples-to-apples)
    _delta("nccl_baseline", "mscclpp")          # total impact


if __name__ == "__main__":
    main()
