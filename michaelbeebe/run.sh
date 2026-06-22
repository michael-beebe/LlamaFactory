#!/usr/bin/env bash
# Copyright 2025 the LlamaFactory authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run: LLAMAFACTORY_FSDP2_UNSHARD_ASYNC_OP=1 NPROC=8 STEPS=200 WARMUP=20 ./run.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
CONFIG_PATH="${CONFIG_PATH:-}"
MODEL_KEY="${MODEL_KEY:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-}"
USE_FSDP="${USE_FSDP:-1}"
FSDP_CONFIG="${FSDP_CONFIG:-${PROJECT_ROOT}/examples/accelerate/fsdp_config.yaml}"
PROFILE_NSYS="${PROFILE_NSYS:-1}"
PROFILE_FORCE_DDP="${PROFILE_FORCE_DDP:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DEFAULT_EXTRA_ARGS="${DEFAULT_EXTRA_ARGS:-per_device_train_batch_size=8 gradient_accumulation_steps=1 cutoff_len=8192 gradient_checkpointing=false}"
ANALYZE_RUN="${ANALYZE_RUN:-1}"
RUN_METHOD="${RUN_METHOD:-auto}"
USE_RAY="${USE_RAY:-}"
FORCE_TORCHRUN="${FORCE_TORCHRUN:-}"

print_step() {
  echo -e "\n==> $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

print_step "Checking configuration"
print_usage() {
  cat <<EOF
Usage: $0 --model <name> [options]
  -m, --model NAME           Model key (e.g., qwen3_full_sft)
  -c, --config PATH          Path to training config YAML (overrides --model)
      --profile-nsys         Enable Nsight Systems profiling (default)
      --no-profile-nsys      Disable Nsight Systems profiling
      --force-ddp            Force DDP (disable FSDP path)
      --fsdp                 Use FSDP (default)
      --no-fsdp              Disable FSDP
      --extra-args "k1=v1 k2=v2"  Extra overrides passed to trainer
      --nproc-per-node N     Processes per node (GPUs)
      --master-addr ADDR     Master addr
      --master-port PORT     Master port
  -h, --help                 Show this help

Env vars still respected: CONFIG_PATH, MODEL_KEY, USE_FSDP, PROFILE_NSYS, PROFILE_FORCE_DDP, EXTRA_ARGS, DEFAULT_EXTRA_ARGS, NPROC_PER_NODE, PYTHON_BIN, ANALYZE_RUN, RUN_METHOD, USE_RAY, FORCE_TORCHRUN.
EOF
}

resolve_config_from_model() {
  local model_key="$1"
  declare -A MODEL_CONFIGS=(
    [qwen3_full_sft]="${PROJECT_ROOT}/examples/train_full/qwen3_full_sft.yaml"
    [qwen3_lora_sft]="${PROJECT_ROOT}/examples/train_lora/qwen3_lora_sft.yaml"
    [qwen3vl_lora_sft]="${PROJECT_ROOT}/examples/train_lora/qwen3vl_lora_sft.yaml"
  )

  if [[ -n "${MODEL_CONFIGS[${model_key}]:-}" ]]; then
    echo "${MODEL_CONFIGS[${model_key}]}"
    return 0
  fi

  local candidate="${PROJECT_ROOT}/examples/${model_key}.yaml"
  if [[ -f "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  return 1
}

# Parse CLI args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model|--model-name)
      MODEL_KEY="$2"; shift 2;;
    -c|--config|--config-path)
      CONFIG_PATH="$2"; shift 2;;
    --profile-nsys)
      PROFILE_NSYS=1; shift;;
    --no-profile-nsys)
      PROFILE_NSYS=0; shift;;
    --force-ddp|--profile-force-ddp)
      PROFILE_FORCE_DDP=1; USE_FSDP=0; shift;;
    --fsdp)
      USE_FSDP=1; shift;;
    --no-fsdp)
      USE_FSDP=0; shift;;
    --extra-args)
      EXTRA_ARGS="$2"; shift 2;;
    --nproc-per-node)
      NPROC_PER_NODE="$2"; shift 2;;
    --master-addr)
      MASTER_ADDR="$2"; shift 2;;
    --master-port)
      MASTER_PORT="$2"; shift 2;;
    -h|--help)
      print_usage; exit 0;;
    *)
      if [[ -z "${MODEL_KEY}" && -n "$1" && ! "$1" =~ ^- ]]; then
        MODEL_KEY="$1"; shift;
      elif [[ -z "${CONFIG_PATH}" && -f "$1" ]]; then
        CONFIG_PATH="$1"; shift;
      else
        echo "Unknown arg: $1" >&2; print_usage; exit 1;
      fi;
      ;;
  esac
done

if [[ -z "${CONFIG_PATH}" ]]; then
  if [[ -n "${MODEL_KEY}" ]]; then
    if ! CONFIG_PATH="$(resolve_config_from_model "${MODEL_KEY}")"; then
      echo "Unknown model key: ${MODEL_KEY}" >&2
      echo "Add mapping in run.sh or pass --config PATH." >&2
      exit 1
    fi
  else
    echo "Missing --model <name> or --config <config.yaml>" >&2
    print_usage
    exit 1
  fi
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    for candidate in python3.11 python3.12 python3.10 python3; do
      if command -v "${candidate}" >/dev/null 2>&1; then
        PYTHON_BIN="${candidate}"
        break
      fi
    done
  fi
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "No compatible Python found. Install Python 3.11 or 3.12." >&2
  exit 1
fi

require_cmd "${PYTHON_BIN}"
require_cmd "nsys"

if [[ -z "${MASTER_PORT}" || "${MASTER_PORT}" == "auto" ]]; then
  MASTER_PORT="$(${PYTHON_BIN} - <<'PY'
import socket
s = socket.socket()
s.bind(('', 0))
print(s.getsockname()[1])
s.close()
PY
)"
fi

if [[ "${USE_FSDP}" == "1" ]]; then
  require_cmd "accelerate"
  if [[ ! -f "${FSDP_CONFIG}" ]]; then
    echo "FSDP config not found: ${FSDP_CONFIG}" >&2
    exit 1
  fi
fi

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1; then
import torch
PY
  echo "Python at ${PYTHON_BIN} does not have torch installed." >&2
  echo "Activate the correct conda env (with torch) or set PYTHON_BIN explicitly." >&2
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-}"
if [[ -z "${NPROC_PER_NODE}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra _CUDA_DEVS <<< "${CUDA_VISIBLE_DEVICES}"
    NPROC_PER_NODE="${#_CUDA_DEVS[@]}"
  else
    NPROC_PER_NODE="$(${PYTHON_BIN} - <<'PY'
import torch
print(torch.cuda.device_count() or 1)
PY
)"
  fi
fi

if [[ -z "${MODEL_NAME}" ]]; then
  if [[ -n "${MODEL_KEY}" ]]; then
    MODEL_NAME="${MODEL_KEY}"
  fi
fi

if [[ -z "${MODEL_NAME}" ]]; then
  base_name="$(basename "${CONFIG_PATH}")"
  MODEL_NAME="${base_name%.*}"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
if [[ "${RUN_METHOD}" == "auto" ]]; then
  if [[ "${USE_RAY}" == "1" ]]; then
    RUN_METHOD="ray"
  elif [[ "${EXTRA_ARGS}" =~ (^|[[:space:]])deepspeed= ]] && [[ "${EXTRA_ARGS}" != *"deepspeed=null"* ]]; then
    RUN_METHOD="deepspeed"
  elif [[ "${USE_FSDP}" == "1" && "${PROFILE_FORCE_DDP}" != "1" ]]; then
    RUN_METHOD="fsdp"
  elif [[ "${USE_FSDP}" == "0" ]] || [[ "${FORCE_TORCHRUN}" == "1" ]]; then
    RUN_METHOD="ddp"
  else
    RUN_METHOD="custom"
  fi
fi
OUT_DIR="${PROJECT_ROOT}/michaelbeebe/outputs/${RUN_METHOD}/${MODEL_NAME}/${TIMESTAMP}"
LOG_PATH="${OUT_DIR}/train.log"
NSYS_OUTPUT_BASE="${OUT_DIR}/nsys_${MODEL_NAME}_${TIMESTAMP}"
FSDP_TEMP_CONFIG="${OUT_DIR}/fsdp_config.yaml"
NSYS_REP="${NSYS_OUTPUT_BASE}.nsys-rep"
NSYS_STATS_PATH="${OUT_DIR}/nsys_stats.txt"
NCCL_STATS_PATH="${OUT_DIR}/nccl_collectives.txt"
WORK_DIR="${PROJECT_ROOT}"

mkdir -p "${OUT_DIR}"

export NCCL_DEBUG="${NCCL_DEBUG:-TRACE}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-COLL,GRAPH,INIT}"
export NCCL_DEBUG_FILE="${NCCL_DEBUG_FILE:-${OUT_DIR}/nccl_%h_%p.log}"
if [[ "${PROFILE_NSYS}" == "1" ]]; then
  TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
  TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
fi
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-600}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"

# Parse extra args into array to pass to training script override fields
if [[ "${USE_FSDP}" == "1" ]]; then
  EXTRA_ARGS="deepspeed=null ${DEFAULT_EXTRA_ARGS} ${EXTRA_ARGS}"
else
  EXTRA_ARGS="${DEFAULT_EXTRA_ARGS} ${EXTRA_ARGS}"
fi
IFS=' ' read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_ARGS}"

print_step "Running training (Nsight Systems + NCCL logging enabled)"
print_step "NCCL_DEBUG=${NCCL_DEBUG} NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}"
print_step "Output directory: ${OUT_DIR}"

pushd "${WORK_DIR}" >/dev/null
trap 'popd >/dev/null || true' EXIT

if [[ "${USE_FSDP}" == "1" && "${PROFILE_FORCE_DDP}" != "1" ]]; then
  export FSDP_CONFIG FSDP_TEMP_CONFIG NPROC_PER_NODE CONFIG_PATH
  python - <<'PY'
import os, yaml

src = os.environ['FSDP_CONFIG']
dst = os.environ['FSDP_TEMP_CONFIG']
nproc = int(os.environ['NPROC_PER_NODE'])
train_cfg_path = os.environ.get('CONFIG_PATH')
train_cfg = {}
if train_cfg_path:
    try:
        with open(train_cfg_path, 'r') as f:
            train_cfg = yaml.safe_load(f) or {}
    except Exception:
        train_cfg = {}

finetuning_type = str(train_cfg.get('finetuning_type', '')).lower()

with open(src, 'r') as f:
    data = yaml.safe_load(f) or {}
data['num_processes'] = nproc
try:
    if 'fsdp_config' in data:
        data['fsdp_config']['fsdp_use_orig_params'] = True
        data['fsdp_config']['fsdp_backward_prefetch'] = data['fsdp_config'].get(
            'fsdp_backward_prefetch', 'BACKWARD_POST'
        )
        if finetuning_type in {'lora', 'qlora', 'dora', 'oft', 'ia3'}:
            data['fsdp_config']['fsdp_auto_wrap_policy'] = 'NO_WRAP'
            data['fsdp_config'].pop('fsdp_transformer_layer_cls_to_wrap', None)
        else:
            data['fsdp_config']['fsdp_transformer_layer_cls_to_wrap'] = data['fsdp_config'].get(
                'fsdp_transformer_layer_cls_to_wrap', 'Qwen3DecoderLayer'
            )
except Exception:
    pass

with open(dst, 'w') as f:
    yaml.safe_dump(data, f)
print(f"Wrote FSDP config to {dst} with num_processes={nproc}")
PY
  if [[ "${PROFILE_NSYS}" == "1" ]]; then
    nsys profile \
      -o "${NSYS_OUTPUT_BASE}" \
      --force-overwrite true \
      --trace=cuda,nvtx,osrt \
      --sample=none \
      --cpuctxsw=none \
      accelerate launch \
        --config_file "${FSDP_TEMP_CONFIG}" \
        "${PROJECT_ROOT}/src/train.py" \
        "${CONFIG_PATH}" "${EXTRA_ARGS_ARR[@]}" 2>&1 | tee "${LOG_PATH}"
  else
    accelerate launch \
      --config_file "${FSDP_TEMP_CONFIG}" \
        "${PROJECT_ROOT}/src/train.py" \
        "${CONFIG_PATH}" "${EXTRA_ARGS_ARR[@]}" 2>&1 | tee "${LOG_PATH}"
  fi
else
  if [[ "${PROFILE_NSYS}" == "1" ]]; then
    nsys profile \
      -o "${NSYS_OUTPUT_BASE}" \
      --force-overwrite true \
      --trace=cuda,nvtx,osrt \
      --sample=none \
      --cpuctxsw=none \
      "${PYTHON_BIN}" -m torch.distributed.run \
      --nnodes 1 \
      --node_rank 0 \
      --nproc_per_node "${NPROC_PER_NODE}" \
      --master_addr "${MASTER_ADDR}" \
      --master_port "${MASTER_PORT}" \
      "${PROJECT_ROOT}/src/llamafactory/launcher.py" \
      "${CONFIG_PATH}" "${EXTRA_ARGS_ARR[@]}" 2>&1 | tee "${LOG_PATH}"
  else
    "${PYTHON_BIN}" -m torch.distributed.run \
      --nnodes 1 \
      --node_rank 0 \
      --nproc_per_node "${NPROC_PER_NODE}" \
      --master_addr "${MASTER_ADDR}" \
      --master_port "${MASTER_PORT}" \
      "${PROJECT_ROOT}/src/llamafactory/launcher.py" \
      "${CONFIG_PATH}" "${EXTRA_ARGS_ARR[@]}" 2>&1 | tee "${LOG_PATH}"
  fi
fi

print_step "Training log saved to ${LOG_PATH}"

print_step "Running nsys stats"
if [[ "${PROFILE_NSYS}" == "1" ]]; then
  if [[ -f "${NSYS_REP}" ]]; then
    nsys stats "${NSYS_REP}" | tee "${NSYS_STATS_PATH}"
  else
    echo "Nsight report not found: ${NSYS_REP}" | tee "${NSYS_STATS_PATH}"
  fi
else
  echo "Nsight profiling disabled; skipping nsys stats." | tee "${NSYS_STATS_PATH}"
fi

print_step "Summarizing NCCL collectives"
NCCL_LOG_GLOB="${OUT_DIR}/nccl_*.log"
if compgen -G "${NCCL_LOG_GLOB}" > /dev/null; then
  awk -v ranks="${NPROC_PER_NODE}" '
    function update_size(kind, sz) {
      bytes[kind] += sz; counts[kind]++;
      size_hist[kind ":" sz]++;
      if (max_sz[kind] < sz) max_sz[kind] = sz;
    }
    /AllReduce/ { 
      c["AllReduce"]++; if (match($0, /size ([0-9]+)/, m)) update_size("AllReduce", m[1]); 
    }
    /AllGather/ { 
      c["AllGather"]++; if (match($0, /size ([0-9]+)/, m)) update_size("AllGather", m[1]); 
    }
    /ReduceScatter/ { 
      c["ReduceScatter"]++; if (match($0, /size ([0-9]+)/, m)) update_size("ReduceScatter", m[1]); 
    }
    /Broadcast/ { 
      c["Broadcast"]++; if (match($0, /size ([0-9]+)/, m)) update_size("Broadcast", m[1]); 
    }
    /AllToAll/ { 
      c["AllToAll"]++; if (match($0, /size ([0-9]+)/, m)) update_size("AllToAll", m[1]); 
    }
    /Reduce[^S]/ { 
      c["Reduce"]++; if (match($0, /size ([0-9]+)/, m)) update_size("Reduce", m[1]); 
    }
    /Gather/ { 
      c["Gather"]++; if (match($0, /size ([0-9]+)/, m)) update_size("Gather", m[1]); 
    }
    /Scatter/ { 
      c["Scatter"]++; if (match($0, /size ([0-9]+)/, m)) update_size("Scatter", m[1]); 
    }
    END {
      printf("Raw counts across logs (per-rank):\n");
      for (k in c) printf("  %s: %d\n", k, c[k]);
      if (ranks > 0) {
        printf("Per-rank average (raw/ranks=%d):\n", ranks);
        for (k in c) printf("  %s: %.2f\n", k, c[k] / ranks);
      }
      printf("\nBytes per collective (sum of per-rank sizes):\n");
      for (k in bytes) printf("  %s: %.2f MB\n", k, bytes[k]/1048576);
      printf("\nMax message size per collective:\n");
      for (k in max_sz) printf("  %s: %.2f MB\n", k, max_sz[k]/1048576);
      printf("\nTop 10 message sizes across all collectives (bytes):\n");
      n=0; for (h in size_hist) sizes[n++]=h;
      asort(sizes, sorted_sizes, "@val_num_desc");
      limit = (n<10)?n:10;
      for (i=1; i<=limit; i++) {
        split(sorted_sizes[i], parts, ":"); kind=parts[1]; sz=parts[2];
        printf("  %s size=%s: %d\n", kind, sz, size_hist[sorted_sizes[i]]);
      }
    }
  ' ${OUT_DIR}/nccl_*.log | tee "${NCCL_STATS_PATH}"
else
  echo "No NCCL logs found in ${OUT_DIR}." | tee "${NCCL_STATS_PATH}"
fi

if [[ "${ANALYZE_RUN}" == "1" ]]; then
  ANALYZE_SCRIPT="${PROJECT_ROOT}/michaelbeebe/analyze_run.py"
  if [[ -f "${ANALYZE_SCRIPT}" ]]; then
    print_step "Generating plots via analyze_run.py"
    "${PYTHON_BIN}" "${ANALYZE_SCRIPT}" --run-dir "${OUT_DIR}" || \
      echo "analyze_run.py failed; see logs above." >&2
  else
    echo "analyze_run.py not found at ${ANALYZE_SCRIPT}." >&2
  fi
fi

print_step "Done"
