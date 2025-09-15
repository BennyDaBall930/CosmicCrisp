#!/usr/bin/env bash
# Basic smoke test for HoneyCrisp dev runtime
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f "${PROJECT_DIR}/run/ports.env" ]; then
  # shellcheck disable=SC1090
  source "${PROJECT_DIR}/run/ports.env"
else
  echo "run/ports.env missing. Start the runtime first." >&2
  exit 1
fi

fail=false

echo "Checking SearXNG on port ${SEARXNG_PORT}..."
if curl -fsS "http://127.0.0.1:${SEARXNG_PORT}/" >/dev/null; then
  echo "  OK"
else
  echo "  FAIL"; fail=true
fi

echo "Checking UI on port ${HONEYCRISP_UI_PORT}..."
if curl -fsS "http://127.0.0.1:${HONEYCRISP_UI_PORT}/health" >/dev/null; then
  echo "  OK"
else
  echo "  FAIL"; fail=true
fi

echo "Checking CDP port ${CHROME_CDP_PORT} listening..."
if lsof -nP -iTCP:"${CHROME_CDP_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "  OK"
else
  echo "  WARN: CDP not listening (Chrome may be unavailable)"
fi

if [ "$fail" = true ]; then
  exit 1
fi
echo "Smoke test passed."

