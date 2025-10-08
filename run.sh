#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-tts311"

if [ ! -d "${VENV_DIR}" ]; then
  echo "Sidecar venv missing. Run ./setup.sh first." >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"

HOST="${TTS_SIDECAR_HOST:-127.0.0.1}"
PORT="${TTS_SIDECAR_PORT:-7055}"
export COQUI_TOS_AGREED=1

echo "[sidecar] Starting on http://${HOST}:${PORT} (venv: ${VENV_DIR})"
exec python -m uvicorn sidecar.app:app --host "${HOST}" --port "${PORT}" --timeout-keep-alive 75
