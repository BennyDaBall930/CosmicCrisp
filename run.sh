#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OS_NAME="$(uname -s)"

if [[ "${OS_NAME}" == "Darwin" ]]; then
  exec "${ROOT_DIR}/dev/macos/run.sh" "$@"
fi

VENV_DIR="${ROOT_DIR}/venv"
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Virtual environment not found. Run ./setup.sh first." >&2
  exit 1
fi

cd "${ROOT_DIR}"
source "${VENV_DIR}/bin/activate"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/searxng:${PYTHONPATH:-}"
export PLAYWRIGHT_BROWSERS_PATH="${ROOT_DIR}/tmp/playwright"
export BROWSER_USE_CONFIG_DIR="${ROOT_DIR}/tmp/browseruse"
export XDG_CONFIG_HOME="${ROOT_DIR}/tmp/xdg"

mkdir -p logs tmp

echo "Starting Apple Zero runtime..."
exec python run_ui.py "$@"
