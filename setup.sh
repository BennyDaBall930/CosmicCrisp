#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-tts311"

echo "[setup] Ensuring Python 3.11 via Homebrew..."
if ! command -v python3.11 >/dev/null 2>&1; then
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew not found. Install Homebrew first: https://brew.sh" >&2
    exit 1
  fi
  brew install python@3.11
  brew link python@3.11 --force --overwrite || true
fi

PY311="$(command -v python3.11)"
echo "[setup] Using Python at: ${PY311}"

if [ ! -d "${VENV_DIR}" ]; then
  echo "[setup] Creating venv at ${VENV_DIR}"
  "${PY311}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python --version
pip install --upgrade pip setuptools wheel

echo "[setup] Installing sidecar dependencies (TTS/Trainer + API libs)..."
# Core runtime for XTTS/VC
pip install --upgrade TTS trainer fastapi "uvicorn[standard]" soundfile

# Ensure torch is present; allow pip to pick the right wheel (macOS CPU/MPS, CUDA, etc.)
if ! python -c "import torch; import sys; print(torch.__version__)" >/dev/null 2>&1; then
  echo "[setup] Installing PyTorch..."
  pip install torch --index-url https://download.pytorch.org/whl/cpu || pip install torch
fi

echo "[setup] Done. To run the sidecar: ./run.sh"

