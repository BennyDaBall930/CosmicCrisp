#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OS_NAME="$(uname -s)"

echo "CosmicCrisp setup starting (OS=${OS_NAME})"

# On macOS delegate to the comprehensive native script.
if [[ "${OS_NAME}" == "Darwin" ]]; then
  exec "${ROOT_DIR}/dev/macos/setup.sh" "$@"
fi

# -------- Linux / generic Unix flow --------

cd "${ROOT_DIR}"

if [[ ! -d ".git" ]]; then
  echo "Please run this script from the repository root." >&2
  exit 1
fi

PY_BIN="${PYTHON:-python3.12}"
if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PY_BIN=python3
  else
    echo "Python 3.12+ is required. Install python3.12 or set PYTHON=/path/to/python before running setup." >&2
    exit 1
  fi
fi

VENV_DIR="${ROOT_DIR}/venv"
if [[ -d "${VENV_DIR}" ]]; then
  echo "Removing existing virtual environment at ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi

echo "Creating virtual environment with ${PY_BIN}"
"${PY_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "Upgrading pip and wheel"
pip install --upgrade pip wheel

echo "Installing project requirements"
pip install -r requirements.txt

if [[ -f "searxng/requirements.txt" ]]; then
  echo "Installing SearXNG dependencies"
  pip install -r searxng/requirements.txt -r searxng/requirements-server.txt
fi

# Install espeak for phonemizer if available via package manager.
if command -v apt-get >/dev/null 2>&1; then
  echo "Installing espeak via apt-get (requires sudo)"
  sudo apt-get update
  sudo apt-get install -y espeak-ng espeak libespeak-ng1 || true
elif command -v pacman >/dev/null 2>&1; then
  echo "Installing espeak via pacman (requires sudo)"
  sudo pacman -S --noconfirm espeak-ng || true
elif command -v yum >/dev/null 2>&1; then
  echo "Installing espeak via yum (requires sudo)"
  sudo yum install -y espeak-ng espeak || true
else
  echo "Install 'espeak-ng' manually for phonemizer support." >&2
fi

echo "Installing Playwright Chromium runtime"
export PLAYWRIGHT_BROWSERS_PATH="${ROOT_DIR}/tmp/playwright"
playwright install chromium
playwright install chromium --only-shell || true

deactivate

echo "Setup complete. Run './run.sh' to start the application."
