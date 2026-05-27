# MSCCL++ vs NCCL Benchmark Harness

End-to-end A/B/C benchmark for the MSCCL++ TorchComms backend hook in
LlamaFactory v1's FSDP2 training path.

## What it measures

Three runs share the same FSDP2 training config and dataset; the only
difference is how device-mesh collectives are routed:

| Run                | `LLAMAFACTORY_TORCHCOMMS_BACKEND` | What it isolates                              |
|--------------------|-----------------------------------|------------------------------------------------|
| `nccl_baseline`    | (unset)                           | Stock `torch.distributed` NCCL — pure baseline |
| `nccl_torchcomms`  | `nccl`                            | NCCL routed through the TorchComms shim        |
| `mscclpp`          | `mscclpp`                         | MSCCL++ algorithms via TorchComms              |

Pairwise deltas:

- `nccl_baseline` ↔ `nccl_torchcomms`: pure TorchComms shim overhead
- `nccl_torchcomms` ↔ `mscclpp`: MSCCL++ algorithm benefit (apples-to-apples)
- `nccl_baseline` ↔ `mscclpp`: total real-world impact

For each run we collect:

- **Per-step training time** (median, p10, p90, full distribution)
- **Loss curve** per step (sanity that runs train equivalently)
- **Per-rank collective dispatch breakdown** — every MSCCL++ collective is
  logged via `MSCCLPP_TORCHCOMMS_TRACE=1` so we know exactly which algorithm
  handled each call, plus how many fell back to NCCL.

## Usage

```bash
# Defaults: 8 GPUs, 30 steps, 5 warmup, examples/v1/train_full/train_full_fsdp2.yaml
./michaelbeebe/bench/run.sh

# Override:
NPROC=2 STEPS=20 WARMUP=3 ./michaelbeebe/bench/run.sh

# Skip the NCCL-via-TorchComms control run (faster two-way A/B):
SKIP_NCCL_TORCHCOMMS=1 ./michaelbeebe/bench/run.sh

# Custom config:
CONFIG=examples/v1/train_full/train_full_fsdp2.yaml ./michaelbeebe/bench/run.sh
```

Each invocation creates a self-contained run directory:

```
michaelbeebe/bench/runs/<TIMESTAMP>_n<NPROC>_s<STEPS>/
├── nccl_baseline/
│   ├── stdout.log         # full training stdout (loss/grad_norm per step)
│   ├── stderr.log         # main-process stderr (warnings, errors)
│   └── per_rank/          # torchrun --redirects 3 per-rank logs
│       └── none_*/attempt_0/<rank>/{stdout,stderr}.log
├── nccl_torchcomms/       # same structure
├── mscclpp/               # same structure
├── results.json           # parsed metrics from all runs
├── step_time_violin.png   # per-step time distribution
├── step_time_series.png   # per-step time vs step number
├── throughput_bar.png     # median step time + pairwise deltas
├── collectives_breakdown.png  # native vs fallback per algorithm
└── loss_curves.png        # loss-vs-step overlay
```

## Required env

```bash
# Auto-detected from the torchcomms package if not set.
export TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP=/path/to/_comms_mscclpp.*.so
```

## Files

- `run.sh`      — orchestrator: runs all three configs, calls `parse.py` then `plot.py`
- `parse.py`    — extracts step times, loss values, and per-collective dispatch
                  tallies from the training logs into `results.json`
- `plot.py`     — generates PNG figures from `results.json`

## Notes

- MSCCL++ trace is always on during the bench (it's gated by env var, near-zero
  cost when off; with N=1 it just tags every dispatch with the algorithm name).
- Step-time measurement uses a `perf_counter` callback installed by
  `timing_runner.py`, which writes `step_timings.jsonl` per run. The parser
  falls back to LlamaFactory's 1-second-resolution `INFO|...step:` log lines
  if the JSONL is missing.
- The torchcomms hook only activates when `LLAMAFACTORY_TORCHCOMMS_BACKEND` is
  set to a non-empty value (`nccl` or `mscclpp`); the baseline run leaves it
  unset so it uses stock `torch.distributed` NCCL. The legacy
  `LLAMAFACTORY_USE_MSCCLPP=1` is preserved as an alias for
  `LLAMAFACTORY_TORCHCOMMS_BACKEND=mscclpp`.
