# MSCCL++ x LlamaFactory FSDP2 Benchmark Plan

## Goal

Measure how much MSCCL++ can speed up FSDP2 training in LlamaFactory compared
to stock NCCL. Produce a compelling before/after comparison on H100 GPUs.

## Key Insight: Where the Communication Is

FSDP2's heavy communication (gradient all-reduce, parameter all-gather) happens
**inside PyTorch's `fully_shard()` implementation** — it uses `torch.distributed`
with the NCCL backend. LlamaFactory doesn't call collectives directly for these;
PyTorch does.

This means we have **two approaches** to swap in MSCCL++:

### Approach A: LD_PRELOAD NCCL Shim (Easy, Existing)
- Use MSCCL++'s `libmscclpp_nccl.so` as a drop-in NCCL replacement
- Zero code changes to LlamaFactory
- Replaces ALL NCCL collectives (allreduce, allgather, etc.)
- Already works — just set `LD_PRELOAD`
- **Limitation**: Can't do mixed-backend (all-or-nothing)

### Approach B: TorchComms Integration (New, Requires PyTorch Changes)
- TorchComms operates at a higher level than `torch.distributed`
- FSDP2 uses `torch.distributed.ProcessGroup` internally, not TorchComms
- To use TorchComms for FSDP2's collectives, we'd need a custom
  `ProcessGroup` backend — this is a PyTorch-level change, not a
  LlamaFactory change
- **This is out of scope for this benchmark**

### Approach C: Explicit Collectives Only (Limited Impact)
- LlamaFactory has some explicit `dist.all_reduce()` calls for loss/metrics
- Swapping these to TorchComms/MSCCL++ is easy but they're a tiny fraction
  of total communication (~1%)
- Not worth the effort for benchmarking

## Recommended Plan: Approach A (LD_PRELOAD)

Since FSDP2's communication is inside PyTorch's NCCL backend, the LD_PRELOAD
shim is the right tool. This gives us a clean A/B comparison:

```
Run A: Stock NCCL         → torchrun ... train.py (baseline)
Run B: MSCCL++ via shim   → LD_PRELOAD=libmscclpp_nccl.so torchrun ... train.py
```

## Setup

### Model
- **Qwen3-4B-Instruct-2507** — large enough for meaningful FSDP2 sharding,
  small enough for 8 H100 GPUs

### Training Config
- FSDP2 with `reshard_after_forward=True`
- bf16 mixed precision, fp32 gradient reduction
- Full fine-tuning (not LoRA — we want maximum gradient communication)
- Dataset: `alpaca_en_demo` (built-in, no download needed)
- Batch size / seq length tuned for ~80% GPU memory utilization

### Hardware
- 8x H100 GPUs (single node, NVSwitch)
- This is where MSCCL++ NVLS algorithms shine

## Benchmark Steps

### Step 1: Environment Setup
- Install LlamaFactory in the conda env
- Verify FSDP2 training works with stock NCCL
- Download Qwen3-4B model weights

### Step 2: Baseline (NCCL)
- Run N training steps with stock NCCL
- Collect: step time, throughput (tokens/sec), GPU utilization
- Optional: nsys profile for communication breakdown

### Step 3: MSCCL++ (LD_PRELOAD)
- Same training config, same model, same data
- Add: `LD_PRELOAD=$MSCCLPP_BUILD/lib/libmscclpp_nccl.so`
- Collect same metrics

### Step 4: Analysis
- Compare step times, throughput
- Calculate speedup percentage
- Break down where MSCCL++ helps (allreduce vs allgather)
- Produce charts/tables for the PR

## File Structure

```
LlamaFactory/michaelbeebe/
├── plan.md              # This file
├── build_env.sh         # Environment setup (existing)
├── run.sh               # Training launcher (existing)
├── analyze_run.py       # Results analysis (existing)
├── configs/
│   ├── qwen3_4b_fsdp2_baseline.yaml   # NCCL baseline config
│   └── qwen3_4b_fsdp2_mscclpp.yaml    # MSCCL++ config (same + LD_PRELOAD)
├── results/
│   ├── baseline/        # NCCL training logs + profiles
│   └── mscclpp/         # MSCCL++ training logs + profiles
└── report.md            # Final comparison report
```

## Questions to Resolve

1. Does the existing `run.sh` support FSDP2? (Check `USE_FSDP` flag and config)
2. Can we control the number of training steps (for quick iteration)?
3. Do we need v0 or v1 training stack? (v1 has native FSDP2 without Accelerate)
4. Is Qwen3-4B already downloaded on this machine?
