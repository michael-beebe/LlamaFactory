# MSCCL++ vs NCCL Benchmark Harness

End-to-end A/B benchmark for the MSCCL++ TorchComms backend hook in
LlamaFactory v1's FSDP2 training path.

## What it measures

For both `nccl_baseline` (stock PyTorch NCCL) and `mscclpp` (our hook
enabled via `LLAMAFACTORY_USE_MSCCLPP=1`):

- **Per-step training time** (median, p10, p90, full distribution)
- **Loss curve** per step (sanity that both runs train equivalently)
- **Per-rank collective dispatch breakdown** — every collective is
  logged via `MSCCLPP_TORCHCOMMS_TRACE=1` so we know exactly which
  algorithm handled each call, plus how many fell back to NCCL.

## Usage

```bash
# Defaults: 8 GPUs, 30 steps, 5 warmup, examples/v1/train_full/train_full_fsdp2.yaml
./michaelbeebe/bench/run.sh

# Override:
NPROC=2 STEPS=20 WARMUP=3 ./michaelbeebe/bench/run.sh

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
├── mscclpp/               # same structure
├── results.json           # parsed metrics from both runs
├── step_time_violin.png   # per-step time distribution
├── step_time_series.png   # per-step time vs step number
├── throughput_bar.png     # median step time + relative speedup
├── collectives_breakdown.png  # native vs fallback per algorithm
└── loss_curves.png        # loss-vs-step overlay
```

## Required env

```bash
# Auto-detected from the torchcomms package if not set.
export TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP=/path/to/_comms_mscclpp.*.so
```

## Files

- `run.sh`      — orchestrator: runs both configs, calls `parse.py` then `plot.py`
- `parse.py`    — extracts step times, loss values, and per-collective dispatch
                  tallies from the training logs into `results.json`
- `plot.py`     — generates PNG figures from `results.json`

## Notes

- MSCCL++ trace is always on during the bench (it's gated by env var, near-zero
  cost when off; with N=1 it just tags every dispatch with the algorithm name).
- Step-time measurement uses LlamaFactory's per-step `INFO|...step:` log
  timestamps (1-second resolution from the format string). For sub-second
  precision on faster steps, pass `--training.log_freq=1` and a higher step
  count to get more samples.
- The hook only activates when `LLAMAFACTORY_USE_MSCCLPP=1` is set; the
  baseline run leaves this unset so it uses stock `torch.distributed` NCCL.
