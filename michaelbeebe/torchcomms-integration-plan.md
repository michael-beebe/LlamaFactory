# MSCCL++ TorchComms Integration — Recommended Improvements

This document captures findings from the LlamaFactory FSDP2 benchmark
(NCCL vs NCCL/TorchComms vs MSCCL++/TorchComms) and proposes concrete
changes to close the perf gap. Every claim cites file:line so it can be
verified independently.

## Bench baseline (8× H100, FSDP2, Qwen3-4B, 30 steps)

| Backend                | median step | step 1   | what handles each FSDP2 op |
|------------------------|-------------|----------|-----------------------------|
| stock NCCL             | 256 ms      | ~3.0 s   | NCCL `RING/SIMPLE` everywhere |
| NCCL via TorchComms    | 247 ms      | 2.5 s    | same NCCL `RING/SIMPLE` |
| MSCCL++ via TorchComms | 295 ms (**+19%**) | **15.85 s** | allgather: `fullmesh2` (97%) + `fullmesh` (3%); allreduce (tiny only): `allpair_packet`; reduce_scatter: **fallback → NCCL** |

The MSCCL++ slowdown decomposes into two distinct problems:

1. **Step-1 spike** (~13 s extra vs NCCL on the first iteration) — the
   integration's lazy-init story. Already mostly addressed by the
   LlamaFactory-side warmup in
   [`src/llamafactory/v1/accelerator/interface.py`](../src/llamafactory/v1/accelerator/interface.py)
   (commit `7b606ee1`).
2. **Steady-state +19% slowdown** (every step after warmup) — driven by
   the integration hard-coding `symmetricMemory=false` and the selector
   picking the cache-hostile `fullmesh2` algorithm anyway. **This is the
   real problem.**

## Root cause analysis (with evidence)

### A. Step-1 latency

`NativeAlgorithm::execute()` defers per-algorithm initialization to the
first call:

```cpp
// src/core/algorithm.cc:47
if (!initialized_) {
    initFunc_(comm);          // CUDA IPC handshakes for all GPU pairs
    initialized_ = true;
}
```

The `TorchCommMSCCLPP` backend constructs the `AlgorithmCollection` in
`init()` but never calls `execute()` on any of the registered
algorithms, so the first user collective during training pays the
full `InitFunc` cost (~13 s on 8 H100s for the three algorithms FSDP2
exercises: `fullmesh`, `fullmesh2`, `allpair_packet`).

NCCL avoids this because its equivalent setup happens during
`init_process_group` — *before* the StepTimingCallback starts ticking.

### B. Steady-state +19% slowdown

Two compounding problems:

**B1.** `AllgatherFullmesh2::generateAllgatherContextKey` returns a
unique key on every call when symmetric memory is off:

```cpp
// src/ext/collectives/allgather/allgather_fullmesh_2.cu:195-200
static int tag = 0;
symmetricMemory_ = symmetricMemory;
if (!symmetricMemory_) {
    // always return a new key if symmetric memory is not enabled.
    return mscclpp::AlgorithmCtxKey{nullptr, nullptr, 0, 0, tag++};
}
```

That means the context cache misses on *every* dispatch, triggering a
full `ContextInitFunc` which re-registers GPU memory across **all** peers
via `comm->registerMemory()` + `setupRemoteMemories()` (an exchange over
`TcpBootstrap`).

**B2.** Our backend hard-codes `symmetricMemory=false`:

```cpp
// python/mscclpp_torchcomms/csrc/TorchCommMSCCLPP.cpp:117
mscclpp::nccl::AlgorithmSelectorConfig config{
    .symmetricMemory = false,
    ...
};
```

So we always hit the slow path. And the selector picks `fullmesh2` for
*every* allgather ≤32 MiB regardless:

```cpp
// src/ext/nccl/algorithm_selector.cc:153-156
if (messageSize <= 32 * (1 << 20)) {
    return algoMap.at("default_allgather_fullmesh2");
}
```

Sister algorithm `default_allgather_fullmesh` would behave well —
`AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0}` (always the same key
→ cache always hits) — but is only picked for >32 MiB messages.

FSDP2 with CUDA graphs would *not* fix this — `tag++` ignores buffer
pointers entirely. The selector picking `fullmesh2` for sub-32 MiB
allgathers is the issue, full stop.

### C. `reduce_scatter` falls back to NCCL

No native MSCCL++ `reduce_scatter` algorithm is registered. FSDP2's
main gradient-reduction op (290 calls / 30 steps) goes through the
dlopen'd `libnccl.so.2`. We pay **NCCL cost + a fallback dispatch
layer** for every gradient reduction.

## Recommendations (in priority order)

### P0 — Steady-state perf (closes most of the +19% gap)

#### R1 — Make the selector prefer `fullmesh` over `fullmesh2` when symmetric memory isn't available

| Field | Value |
|---|---|
| Where | mscclpp repo — `src/ext/nccl/algorithm_selector.cc:148-167` |
| Effort | ~30 min |
| Estimated impact | -10 to -15% step time (eliminates the per-call IPC re-registration) |
| Risk | Low. `fullmesh` is the existing default for >32 MiB; we'd just extend its range when no symmetric memory. |

Proposed change:

```cpp
std::shared_ptr<Algorithm> selectSingleNodeAllgather(
    const std::unordered_map<std::string, std::shared_ptr<Algorithm>>& algoMap,
    const CollectiveRequest& request,
    const AlgorithmSelectorConfig& config) {
  const size_t messageSize = request.messageSize;

  if (messageSize <= 32 * (1 << 20)) {
    // fullmesh2 caches per buffer config ONLY when symmetric memory is
    // available. Without it, every call re-registers memory across all
    // peers — far slower than NCCL Ring for typical FSDP2 sizes. Prefer
    // fullmesh in that case, which has a constant context key and reuses
    // its registration across calls.
    if (!config.symmetricMemory && !config.isCuMemMapAllocated) {
      return algoMap.at("default_allgather_fullmesh");
    }
    return algoMap.at("default_allgather_fullmesh2");
  }
  // ... existing >32 MiB handling unchanged
}
```

#### R2 — Register a native MSCCL++ `reduce_scatter` algorithm

| Field | Value |
|---|---|
| Where | mscclpp repo — new files under `src/ext/collectives/reduce_scatter/`, registration in `src/ext/collectives/algorithm_collection_builder.cc` |
| Effort | ~1-2 weeks (new CUDA kernel) |
| Estimated impact | -5 to -10% (eliminates NCCL fallback dispatch + uses NVLink directly) |
| Risk | Medium — new kernel implementation, need correctness tests |

Pattern can mirror `AllgatherFullmesh` in reverse: each peer writes a
partial-reduced chunk to a per-destination scratch buffer via
`SmChannel`, a single kernel performs the final reduction into the
output. This is what NCCL does internally for Ring ReduceScatter; the
gain comes from skipping the libnccl.so dlopen marshalling layer and
using the cached connection state we already paid for in `InitFunc`.

### P1 — Step-1 latency (cleanup of the workaround)

#### R3 — Move the warmup into `TorchCommMSCCLPP::init()` (library-level, not app-level)

| Field | Value |
|---|---|
| Where | mscclpp repo — `python/mscclpp_torchcomms/csrc/TorchCommMSCCLPP.cpp::init()` (end of method) |
| Effort | ~1 hour |
| Estimated impact | Step 1 drops from ~16 s to ~3 s for every consumer, not just LlamaFactory |
| Risk | Low — `AlgorithmCollection` already exists by the end of init; we iterate it and run one dummy execute per algo |

The LlamaFactory-side warmup
([`_warmup_torchcomms_meshes`](../src/llamafactory/v1/accelerator/interface.py))
is a workaround for what should be backend behavior. Move it down a
layer so torchtitan, vLLM, and any future torchcomms consumer get the
same fix for free.

Sketch:
```cpp
void TorchCommMSCCLPP::init(...) {
  // ... existing init building algorithmCollection_ ...

  // Trigger lazy InitFunc on every registered native algorithm so the
  // first user collective doesn't pay the CUDA-IPC handshake cost.
  for (const auto& [collective, byName] : algorithmCollection_->algorithms()) {
    for (const auto& [algoName, algo] : byName) {
      if (algo->type() != AlgorithmType::Native) continue;
      // tiny stream-allocated dummy buffers; ignore output, only the
      // first-dispatch side effect matters.
      runOneWarmupCollective(algo, internal_stream_);
    }
  }
  cudaStreamSynchronize(internal_stream_);
}
```

#### R4 — Remove the LlamaFactory-side warmup once R3 lands

| Field | Value |
|---|---|
| Where | `src/llamafactory/v1/accelerator/interface.py::_warmup_torchcomms_meshes` |
| Effort | 5 min |
| Estimated impact | None on perf (R3 already covers it); just code hygiene |

### P2 — Observability + future-proofing

#### R5 — Per-call wall-time instrumentation

| Field | Value |
|---|---|
| Where | `python/mscclpp_torchcomms/csrc/TorchWorkMSCCLPP.cpp` |
| Effort | ~1 day |
| Estimated impact | Diagnostic only — lets us decompose step time into "registration / kernel / fallback dispatch / collective wait" |
| Risk | Low; gated behind `MSCCLPP_TORCHCOMMS_TRACE=2` so it's off by default |

Currently `MSCCLPP_TORCHCOMMS_TRACE=1` emits one line per dispatch with
algo + bytes + dtype but no timing. Adding wall-time fields would let us
see *exactly* which phase dominates the per-call overhead (we suspect
`registerMemory` from R1, but it would be nice to confirm). Useful for
finding the next bottleneck after R1+R2 land.

#### R6 — Plumb `symmetricMemory` from the tensor's storage

| Field | Value |
|---|---|
| Where | `python/mscclpp_torchcomms/csrc/TorchCommMSCCLPP.cpp:117` |
| Effort | ~1 day |
| Estimated impact | Future-proof: when PyTorch SymmetricMemory + FSDP2 integration lands upstream, our backend will automatically benefit (cache hits, faster `fullmesh2`) |
| Risk | Low; default-false fallback stays correct |

Replace the hard-coded `.symmetricMemory = false` with a runtime check:
inspect the tensor's storage allocator (or check
`at::cuda::SymmetricMemory::has_memory(tensor)`) and pass the real
value through to the selector.

## Out of scope

- **CUDA graphs**: FSDP2 doesn't use them today. Even if it did,
  `fullmesh2`'s `tag++` would still miss the cache regardless of buffer
  stability. R1 above is the correct fix; CUDA graphs are orthogonal.
- **Multi-node selection**: the existing selector only handles
  single-node NVLink. Adding IB-aware multi-node selection is future
  work and unrelated to the current FSDP2-on-one-node bottleneck.
- **NVLS** (NVLink Switch): not configured / not available on this
  hardware. Algorithm registry includes NVLS variants but the selector
  skips them when `nvlsSupported=false`.

## Verification plan

| Recommendation | How to verify | Pass criteria |
|---|---|---|
| R1 | Run bench; check `[MSCCLPP] rank=0 collective=allgather` trace lines | All ≤32 MiB allgathers show `algo='default_allgather_fullmesh'`. Median step time drops from ~295 ms toward ~255 ms (parity with NCCL/TorchComms). |
| R2 | Same bench, check `[NcclFallback]` trace lines | Zero `reduce_scatter -> NCCL` fallbacks. Median step time drops another 5-10%. |
| R3 | Disable R4 + set `LLAMAFACTORY_TORCHCOMMS_WARMUP=0` | First step ≤4 s (was 15.85 s with no warmup). |
| R5 | Set `MSCCLPP_TORCHCOMMS_TRACE=2`; run one training step | Per-call trace lines sum to ≥95% of the step's wall time; allows attribution. |
| R6 | Run with PyTorch SymmetricMemory enabled (when upstream lands) | Trace shows `symmetricMemory=true` and `fullmesh2` cache hits (no `tag++` churn). |

## Estimated overall outcome

After R1 + R2 + R3:

| Backend | median step (today) | median step (projected) |
|---|---|---|
| stock NCCL | 256 ms | 256 ms (baseline) |
| NCCL via TorchComms | 247 ms | 247 ms (baseline) |
| MSCCL++ via TorchComms | **295 ms (+19%)** | **~240 ms (-3%)** |

I.e. MSCCL++ should reach parity or modestly beat NCCL on this 8×H100
FSDP2 workload. To go faster than that on this hardware we'd need NVLS,
which isn't available here.

## Owner-side actions

1. Review this document, push back on anything that doesn't match your
   priorities.
2. Decide whether R2 (new kernel) is in scope or should wait — it's the
   biggest single-item effort but also has the second-biggest payoff.
3. Decide whether the mscclpp changes (R1, R2, R3, R5, R6) land on
   `michaelbeebe/torchcomms` (your fork branch) or go upstream as a PR
   to microsoft/mscclpp. R1 alone is a self-contained 10-line patch
   that would be uncontroversial to upstream.
