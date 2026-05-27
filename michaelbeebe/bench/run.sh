#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Benchmark runner: stock NCCL vs NCCL-via-TorchComms vs MSCCL++-via-TorchComms
# on the same FSDP2 training config. Produces a self-contained run directory
# with raw per-rank stderr logs, training stdout, parsed JSON metrics, and PNG
# figures.
#
# The three-way comparison isolates:
#   nccl_baseline   vs nccl_torchcomms : pure TorchComms shim overhead
#   nccl_torchcomms vs mscclpp         : MSCCL++ algorithm benefit
#   nccl_baseline   vs mscclpp         : total real-world impact
#
# Usage:
#   ./bench/run.sh                                   # 8 GPUs, default config
#   NPROC=2 ./bench/run.sh                           # 2 GPUs
#   STEPS=20 WARMUP=5 ./bench/run.sh                 # custom step counts
#   CONFIG=examples/v1/train_full/train_full_fsdp2.yaml ./bench/run.sh
#   SKIP_NCCL_TORCHCOMMS=1 ./bench/run.sh            # skip the middle control run
#
# Required env (auto-detected if possible):
#   TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP  path to _comms_mscclpp.*.so

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="${REPO_ROOT}/michaelbeebe/bench"
RUNS_ROOT="${BENCH_DIR}/runs"

NPROC="${NPROC:-8}"
STEPS="${STEPS:-30}"
WARMUP="${WARMUP:-5}"
CONFIG="${CONFIG:-examples/v1/train_full/train_full_fsdp2.yaml}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUNS_ROOT}/${TIMESTAMP}_n${NPROC}_s${STEPS}"

# Optional: path to libmscclpp_nccl.so (the LD_PRELOAD NCCL drop-in shim).
# When provided, a third 'mscclpp_shim' run is added to the comparison.
# Auto-detect from a sibling mscclpp build dir if not set.
if [[ -z "${MSCCLPP_NCCL_SHIM:-}" ]]; then
    for cand in \
        "${REPO_ROOT}/../build/cp310-cp310-linux_x86_64/lib/libmscclpp_nccl.so" \
        "${REPO_ROOT}/../build-torchcomm/lib/libmscclpp_nccl.so"; do
        if [[ -f "${cand}" ]]; then
            MSCCLPP_NCCL_SHIM="$(realpath "${cand}")"
            break
        fi
    done
fi

mkdir -p "${RUN_DIR}"

# Auto-detect MSCCL++ backend .so if not provided.
if [[ -z "${TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP:-}" ]]; then
    AUTOSO=$(python3 -c "import torchcomms,os,glob; \
        d=os.path.dirname(torchcomms.__file__); \
        m=glob.glob(os.path.join(d,'_comms_mscclpp*.so')); \
        print(m[0] if m else '')")
    if [[ -z "${AUTOSO}" ]]; then
        echo "ERROR: TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP not set and no _comms_mscclpp*.so found in torchcomms package dir" >&2
        exit 1
    fi
    export TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP="${AUTOSO}"
fi

echo "== Bench config =="
echo "  RUN_DIR    : ${RUN_DIR}"
echo "  NPROC      : ${NPROC}"
echo "  STEPS      : ${STEPS} (warmup ${WARMUP})"
echo "  CONFIG     : ${CONFIG}"
echo "  MSCCL++ .so: ${TORCHCOMMS_BACKEND_LIB_PATH_MSCCLPP}"
echo

run_one() {
    local label="$1"
    # Backend selector: "" → plain torch.distributed NCCL (no torchcomms);
    #                   "nccl"|"mscclpp" → torchcomms with that backend.
    local backend="$2"
    local out_dir="${RUN_DIR}/${label}"
    mkdir -p "${out_dir}/per_rank"

    echo "== Running: ${label} (backend=${backend:-stock-nccl}) =="
    # Override max_steps from CLI; trace is always on so we can parse algo selection.
    LLAMAFACTORY_TORCHCOMMS_BACKEND="${backend}" \
    MSCCLPP_TORCHCOMMS_TRACE=1 \
    BENCH_TIMING_PATH="${out_dir}/step_timings.jsonl" \
    PYTHONPATH="${BENCH_DIR}/..:${PYTHONPATH:-}" \
    torchrun \
        --nproc_per_node="${NPROC}" \
        --rdzv_backend c10d \
        --rdzv_endpoint="localhost:0" \
        --redirects 3 \
        --log-dir "${out_dir}/per_rank" \
        --tee 0 \
        "${BENCH_DIR}/timing_runner.py" \
        "${CONFIG}" \
        max_steps="${STEPS}" \
        > "${out_dir}/stdout.log" 2> "${out_dir}/stderr.log" || {
            echo "  FAILED — see ${out_dir}/stderr.log"; return 1;
        }
    echo "  OK"
}

cd "${REPO_ROOT}"

run_one nccl_baseline    ""
if [[ "${SKIP_NCCL_TORCHCOMMS:-0}" != "1" ]]; then
    run_one nccl_torchcomms  "nccl"
fi
run_one mscclpp          "mscclpp"

# Parse + plot
echo
echo "== Parsing logs =="
python3 "${BENCH_DIR}/parse.py" --run-dir "${RUN_DIR}" --warmup "${WARMUP}"
echo
echo "== Generating plots =="
python3 "${BENCH_DIR}/plot.py" --run-dir "${RUN_DIR}"

echo
echo "== DONE =="
echo "  Artifacts: ${RUN_DIR}"
ls -la "${RUN_DIR}"
