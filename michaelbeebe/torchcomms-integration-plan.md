# MSCCL++ TorchComms Integration — Recommended Improvements

**Scope:** changes confined to the torchcomms integration. That means
**only** these files are in scope:

* `python/mscclpp_torchcomms/csrc/` (the C++ backend)
* `python/mscclpp_torchcomms/CMakeLists.txt`
* `LlamaFactory/src/llamafactory/v1/accelerator/interface.py` (the consumer hook)

Everything under `src/core/`, `src/ext/collectives/`, and
`src/ext/nccl/` is explicitly **out of scope** — those are MSCCL++
library / NCCL-shim concerns, not torchcomms integration concerns.

## Bench baseline (8× H100, FSDP2, Qwen3-4B, 30 steps)

| Backend                | median step | step 1   | what handles each FSDP2 op |
|------------------------|-------------|----------|-----------------------------|
| stock NCCL             | 256 ms      | ~3.0 s   | NCCL `RING/SIMPLE` everywhere |
| NCCL via TorchComms    | 247 ms      | 2.5 s    | same NCCL `RING/SIMPLE` |
| MSCCL++ via TorchComms | 295 ms (**+19%**) | **15.85 s** | allgather: `fullmesh2` (97%) + `fullmesh` (3%); allreduce (tiny only): `allpair_packet`; reduce_scatter: **fallback → NCCL** |

## Root cause analysis

### A. Step-1 latency (15.85 s vs 2.5 s)

`NativeAlgorithm::execute()` (in the library) defers `InitFunc` to the
first call. Our `TorchCommMSCCLPP::init()` builds the algorithm
collection but never warms it — so the user's first training collective
pays the cost. Already mitigated app-side in LlamaFactory (commit
`7b606ee1`); we should pull this into our backend (see R5 below).

### B. Steady-state +19% slowdown

Decomposes into two parts:

**B1. `fullmesh2` keeps re-registering memory.** `fullmesh2`'s context
key generator returns `tag++` whenever `symmetricMemory == false`. We
set `symmetricMemory = false` in `TorchCommMSCCLPP.cpp:117`, AND the
NCCL-shim selector our `selectAlgorithm()` delegates to picks `fullmesh2`
for every allgather ≤32 MiB. Net effect: every allgather (1989 calls/run)
re-registers GPU memory across all peers via `TcpBootstrap`. That's the
bulk of the +19%.

**B2. Per-call wrapper overhead.** ~5-15 μs per dispatch from
`std::getenv("MSCCLPP_TORCHCOMMS_TRACE")` (every call!),
`selectAlgorithm()` doing map lookups + selector evaluation with no
caching, heap-alloc'ing a fresh `TorchWorkMSCCLPP` per call, and two
`cudaEventRecord`s. Aggregate is ~10-30 ms across 2100 calls (≈0.3-1%
of cumulative step time). Smaller contributor than B1 but free to fix.

### C. reduce_scatter falls back to NCCL

**Out of scope** per request — requires a new MSCCL++ kernel.

## Recommendations (in priority order)

### P0 — Steady-state perf (closes most of the +19% gap)

#### R0 — Replace the fullmesh2 preference in our selector

**Where:** `python/mscclpp_torchcomms/csrc/TorchCommMSCCLPP.cpp::selectAlgorithm()` (line 93-141)

**Effort:** ~2 hours
**Estimated impact:** **-10 to -15% step time** (this is the single biggest in-scope win)
**Risk:** Low — `fullmesh` is well-tested and already used as the fallback for >32 MiB allgathers.

`selectAlgorithm()` currently delegates allgather selection to
`mscclpp::nccl::selectSingleNodeAllgather(algoMap, request, config)`,
which picks `default_allgather_fullmesh2` for every allgather ≤32 MiB.
We do this delegation ourselves — we can intercept it.

Proposed change inside `selectAlgorithm()` around line 138:

```cpp
if (request.collective == "allgather") {
    // fullmesh2 caches per-buffer-config ONLY when symmetric memory is
    // available. Without it, every call re-registers IPC memory across
    // all peers (~ms-scale TcpBootstrap exchange), which dominates step
    // time on FSDP2-style workloads. Prefer fullmesh in that case, which
    // always reuses the same context.
    const size_t messageSize = request.messageSize;
    if (!config.symmetricMemory && !config.isCuMemMapAllocated && messageSize <= 32 * (1 << 20)) {
        auto it = algoMap.find("default_allgather_fullmesh");
        if (it != algoMap.end()) return it->second;
    }
    return mscclpp::nccl::selectSingleNodeAllgather(algoMap, request, config);
}
```

We do **not** modify the upstream `selectSingleNodeAllgather`; we just
short-circuit it in our delegating function.

#### R1 — Detect and propagate real `symmetricMemory` flag

**Where:** `python/mscclpp_torchcomms/csrc/TorchCommMSCCLPP.cpp:117`

**Effort:** ~1 day
**Estimated impact:** Future-proof. Today it changes nothing because
FSDP2 doesn't use SymmetricMemory yet. When upstream PyTorch enables it
for FSDP2 (work in progress in `torch._distributed._symmetric_memory`),
this flips on automatically and `fullmesh2` becomes cache-friendly.
**Risk:** Low — `false` default keeps current behavior correct.

Replace the hard-coded `.symmetricMemory = false` with a detection
helper: check whether the input/output tensor storage is backed by
`SymmetricMemory` (using PyTorch's `at::cuda::SymmetricMemory::has_memory(t)`
or equivalent storage-pointer check). Plumb through to the selector.

### P1 — Per-call wrapper overhead (small but free)

#### R2 — Cache the trace flag at init time

**Where:** `python/mscclpp_torchcomms/csrc/TorchCommMSCCLPP.{hpp,cpp}`

**Effort:** ~30 min
**Estimated impact:** Removes the `std::getenv` call from every dispatch.
~50-200 ns × 2100 calls ≈ 0.1-0.4 ms/run. Tiny but trivially free.
**Risk:** None — semantics unchanged, just cached.

In `init()`, read `MSCCLPP_TORCHCOMMS_TRACE` once and stash on a
`const bool trace_;` member. Replace the two `std::getenv` sites in
`executeCollective()` and `reduce_scatter_single` with `if (trace_)`.

#### R3 — Cache the (collective, message-size-bucket) → algorithm lookup

**Where:** `python/mscclpp_torchcomms/csrc/TorchCommMSCCLPP.{hpp,cpp}::selectAlgorithm()`

**Effort:** ~2 hours
**Estimated impact:** Removes ~1-5 μs/call from the selector walk. Modest
direct savings, but more importantly it lets us add other selector
heuristics in R0 without worrying about per-call cost.
**Risk:** Low — cache key (collective, log2(messageSize)) plus a few
config bits captures every meaningful selection input. Cache is rebuilt
on `init()`/`finalize()` boundaries.

Add a tiny per-comm cache:
```cpp
struct AlgoSelectionKey {
    std::string collective;
    int sizeBucket;     // = log2(messageSize), -1 for 0-byte
    bool symMem;
    bool cuMemMap;
};
std::unordered_map<AlgoSelectionKey, std::shared_ptr<mscclpp::Algorithm>> selectionCache_;
```

Look up first; on miss, run the full selector and cache the answer.

#### R4 — Pool TorchWorkMSCCLPP objects

**Where:** `python/mscclpp_torchcomms/csrc/TorchWorkMSCCLPP.{hpp,cpp}`

**Effort:** ~half day
**Estimated impact:** Removes the ~1 μs heap allocation per call. ~2 ms
across 2100 calls. **Probably not worth doing** unless we find that
allocator contention shows up under heavier workloads.
**Risk:** Medium — `c10::intrusive_ptr` lifetime is tricky to pool
correctly. Defer unless R6 instrumentation proves it matters.

### P2 — Step-1 latency cleanup

#### R5 — Move the LlamaFactory-side warmup into `TorchCommMSCCLPP::init()`

**Where:** `python/mscclpp_torchcomms/csrc/TorchCommMSCCLPP.cpp::init()`,
then delete `_warmup_torchcomms_meshes` in `interface.py`.

**Effort:** ~1 hour
**Estimated impact:** No additional perf gain over today's app-side
warmup, but the fix benefits every torchcomms consumer (torchtitan,
future apps) and lets us drop the workaround from LlamaFactory.
**Risk:** Low — the app-side version stays as a fallback until R5 lands.

At the end of `init()`, iterate the registered native algorithms and run
each one once with tiny dummy buffers on `internal_stream_`. This
triggers the lazy `InitFunc` (per-algorithm CUDA-IPC handshake) **before**
the user's first training collective.

### P3 — Observability

#### R6 — Per-call wall-time instrumentation (`MSCCLPP_TORCHCOMMS_TRACE=2`)

**Where:** `python/mscclpp_torchcomms/csrc/TorchCommMSCCLPP.cpp`, `TorchWorkMSCCLPP.cpp`

**Effort:** ~1 day
**Estimated impact:** Diagnostic only. Lets us decompose each
collective's wall time into: selector time, context init time, kernel
launch time, fallback dispatch time. Useful for finding the next
bottleneck after R0 lands.
**Risk:** None — gated behind `TRACE=2`, off by default.

## What was deferred or dropped

| Original idea | Status | Why |
|---|---|---|
| Tweak `src/ext/nccl/algorithm_selector.cc` to prefer `fullmesh` | **Replaced by R0** | Out of scope (library code); R0 achieves the same effect from inside our integration |
| Add a native MSCCL++ `reduce_scatter` algorithm | **Dropped per request** | Requires new CUDA kernel; out of torchcomms integration scope |
| Pool TorchWorkMSCCLPP objects (R4) | **Deferred** | Smallest expected impact; revisit only if profiling shows allocator pressure |
| CUDA-graph capture support | **Out of scope** | FSDP2 doesn't use CUDA graphs; would require upstream PyTorch FSDP2 changes |

## Verification plan

| Recommendation | How to verify | Pass criteria |
|---|---|---|
| R0 | `MSCCLPP_TORCHCOMMS_TRACE=1`; grep for `algo='default_allgather_*'` lines | All ≤32 MiB allgathers report `algo='default_allgather_fullmesh'`. Median step time drops from ~295 ms toward ~250 ms (parity with NCCL/TorchComms). |
| R1 | Run with PyTorch SymmetricMemory enabled once available | Trace shows `symmetricMemory=true`; `fullmesh2` cache hits visible in any added context-init counter. |
| R2 | `strace -e trace=getenv` on rank 0 during one step | Zero `getenv("MSCCLPP_TORCHCOMMS_TRACE")` calls after `init()` returns. |
| R3 | Add a hit/miss counter to the selection cache; print on `finalize()` | After warmup, miss count == small constant (≤ number of distinct collective-bucket pairs); hit count == total dispatches − miss count. |
| R5 | Set `LLAMAFACTORY_TORCHCOMMS_WARMUP=0`; rebuild backend with R5 | Step 1 ≤ 4 s (was 15.85 s without warmup, ~0.5 s with app-side warmup). |
| R6 | `TRACE=2`; sum per-call durations from one step | Sum ≥ 95% of step's wall-clock time; allows attribution. |

## Expected outcome

After R0 + (R2, R3, R5):

| Backend                | today               | projected           |
|------------------------|---------------------|---------------------|
| stock NCCL             | 256 ms              | 256 ms (baseline)   |
| NCCL via TorchComms    | 247 ms              | 247 ms (baseline)   |
| MSCCL++ via TorchComms | **295 ms (+19%)**   | **~250 ms (±2%)**   |

I.e., MSCCL++ should reach **parity** with NCCL/TorchComms on this
workload using only torchcomms-integration changes. To go faster on
this exact workload we'd need either (a) a native `reduce_scatter`
algorithm (out of scope per request), (b) PyTorch SymmetricMemory
+ FSDP2 (R1 future-proofs for it), or (c) NVLS support (different
hardware).

## Owner-side actions

1. Review priorities — flag anything you want re-ordered or dropped.
2. Confirm R0 is the right first move (highest ROI, smallest change, in scope).
3. Decide whether R5 lands now or later. The current app-side warmup
   works; R5 is mainly a "do it in the right place" cleanup.
4. All R# changes go on `michaelbeebe/torchcomms-llamafactory-testing`
   (the work branch we made earlier), **not** `michaelbeebe/torchcomms`
   (the PR branch).
