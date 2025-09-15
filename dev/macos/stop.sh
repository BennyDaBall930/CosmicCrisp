#!/usr/bin/env bash
# Stop HoneyCrisp macOS dev runtime (tmux or nohup)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common.sh"

stopped_any=false

if command -v tmux >/dev/null 2>&1 && tmux has-session -t honeycrisp 2>/dev/null; then
  echo "Killing tmux session 'honeycrisp'..."
  tmux kill-session -t honeycrisp || true
  stopped_any=true
fi

for name in searxng ui agent chrome; do
  pid_file="${RUN_DIR}/${name}.pid"
  if [ -f "$pid_file" ]; then
    pid=$(cat "$pid_file" 2>/dev/null || true)
    if [ -n "${pid:-}" ] && kill -0 "$pid" >/dev/null 2>&1; then
      echo "Stopping ${name} (pid ${pid})..."
      kill "$pid" 2>/dev/null || true
      sleep 1
      if kill -0 "$pid" >/dev/null 2>&1; then
        echo "Force killing ${name} (pid ${pid})..."
        kill -9 "$pid" 2>/dev/null || true
      fi
      stopped_any=true
    fi
    rm -f "$pid_file"
  fi
done

if [ "$stopped_any" = false ]; then
  echo "No running processes found."
else
  echo "Stopped. Logs remain in ${LOG_DIR}."
fi

