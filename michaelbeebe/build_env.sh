#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-"${PROJECT_ROOT}/.venv"}"
PYTHON_BIN="${PYTHON_BIN:-}"
TORCH_CUDA_VERSION="${TORCH_CUDA_VERSION:-cu121}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/${TORCH_CUDA_VERSION}}"

print_step() {
  echo -e "\n==> $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

print_step "Checking prerequisites"
require_cmd "pip"

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

PYTHON_VERSION_FULL="$(${PYTHON_BIN} - <<'PY'
import sys
print("%d.%d" % (sys.version_info.major, sys.version_info.minor))
PY
)"

if [[ "${PYTHON_VERSION_FULL}" == "3.13" ]]; then
  echo "Python ${PYTHON_VERSION_FULL} detected." >&2
  echo "PyTorch CUDA wheels for torchaudio are not available on 3.13 yet." >&2
  echo "Install Python 3.11/3.12 and rerun with: PYTHON_BIN=python3.11 ./michaelbeebe/build_env.sh" >&2
  exit 1
fi

REQUIRED_CONDA_ENV="michaelbeebe-llamafactory"
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  if [[ "${CONDA_DEFAULT_ENV}" != "${REQUIRED_CONDA_ENV}" ]]; then
    echo "Active conda env is '${CONDA_DEFAULT_ENV}', expected '${REQUIRED_CONDA_ENV}'." >&2
    echo "Activate it first: conda activate ${REQUIRED_CONDA_ENV}" >&2
    exit 1
  fi
  print_step "Using conda environment: ${CONDA_DEFAULT_ENV}"
else
  echo "Conda environment not active. Activate: conda activate ${REQUIRED_CONDA_ENV}" >&2
  exit 1
fi

print_step "Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

print_step "Installing PyTorch with CUDA (${TORCH_CUDA_VERSION})"
python -m pip install --index-url "${PYTORCH_INDEX_URL}" torch torchvision torchaudio

print_step "Installing LLaMA Factory dependencies"
python -m pip install -e "${PROJECT_ROOT}"
python -m pip install -r "${PROJECT_ROOT}/requirements/dev.txt"

print_step "Sanity check: CUDA/NCCL availability"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("nccl available:", torch.distributed.is_nccl_available())
PY

print_step "Done"

echo "Next steps:"
echo "  1) Activate conda env: conda activate ${REQUIRED_CONDA_ENV}"
echo "  2) Start training: ./michaelbeebe/run.sh --model qwen3_lora_sft"
