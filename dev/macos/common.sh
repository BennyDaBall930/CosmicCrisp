#!/usr/bin/env bash
# Common helpers for HoneyCrisp macOS dev runtime (no AppleScript)
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/run}"

mkdir -p "$LOG_DIR" "$RUN_DIR"

# Flag to ensure tmux warning is only shown once
TMUX_WARNING_SHOWN=false

port_in_use() {
  local port="$1"
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
}

find_free_port() {
  local p="$1"
  while port_in_use "$p"; do p=$((p+1)); done
  echo "$p"
}

is_listening() {
  local port="$1"
  port_in_use "$port"
}

wait_for_http() {
  local url="$1" timeout="${2:-60}"; local start end
  start=$(date +%s)
  while true; do
    if curl -fsS "$url" >/dev/null 2>&1; then return 0; fi
    end=$(date +%s)
    if [ $((end-start)) -ge "$timeout" ]; then return 1; fi
    sleep 1
  done
}

ensure_tmux_session() {
  local session="${1:-honeycrisp}"
  if command -v tmux >/dev/null 2>&1; then
    tmux has-session -t "$session" 2>/dev/null || tmux new-session -d -s "$session" -n bootstrap "sleep 3600"
  fi
}

# Start a process either in a tmux window or as background nohup.
# Usage: start_proc name command
start_proc() {
  local name="$1"; shift
  local cmd="$*"
  mkdir -p "$LOG_DIR" "$RUN_DIR"

  # Check for tmux and show a warning if it's not installed
  if ! $TMUX_WARNING_SHOWN && ! command -v tmux >/dev/null 2>&1; then
    if [ -t 1 ]; then # Only print colors if stdout is a terminal
        local YELLOW='\033[1;33m'; local NC='\033[0m'
        echo -e "${YELLOW}[Notice] tmux not found. Services are running in the background.${NC}" >&2
        echo -e "${YELLOW}To view service logs in separate terminals, install tmux (e.g., 'brew install tmux') and restart.${NC}" >&2
    else
        echo "[Notice] tmux not found. Services are running in the background." >&2
        echo "To view service logs in separate terminals, install tmux (e.g., 'brew install tmux') and restart." >&2
    fi
    TMUX_WARNING_SHOWN=true
  fi

  if command -v tmux >/dev/null 2>&1; then
    ensure_tmux_session honeycrisp
    # If a window with the same name exists, kill it first
    if tmux list-windows -t honeycrisp 2>/dev/null | awk '{print $2}' | sed 's/:$//' | grep -q "^${name}$"; then
      tmux kill-window -t "honeycrisp:${name}" || true
    fi
    # Write a tiny wrapper to avoid quoting issues with spaces in paths
    local wrapper="${RUN_DIR}/${name}.tmux.sh"
    cat > "$wrapper" <<WRAP
#!/usr/bin/env bash
set -e
cd "${PROJECT_DIR}"
eval exec ${cmd} 2>&1 | tee -a "${LOG_DIR}/${name}.log"
WRAP
    chmod +x "$wrapper"
    tmux new-window -t honeycrisp -n "$name" "/bin/bash \"$wrapper\""
  else
    # nohup fallback with PID file
    local runner="${RUN_DIR}/${name}.daemon.sh"
    cat > "$runner" <<RUN
#!/usr/bin/env bash
set -e
cd "${PROJECT_DIR}"
eval exec ${cmd} >> "${LOG_DIR}/${name}.log" 2>&1
RUN
    chmod +x "$runner"
    nohup "$runner" >/dev/null 2>&1 &
    echo $! > "${RUN_DIR}/${name}.pid"
  fi
}

write_ports_files() {
  local searx_port="$1" ui_port="$2" cdp_port="$3"
  mkdir -p "$RUN_DIR"
  cat > "${RUN_DIR}/ports.env" <<ENV
export SEARXNG_PORT=${searx_port}
export HONEYCRISP_UI_PORT=${ui_port}
export WEB_UI_PORT=${ui_port}
export CHROME_CDP_PORT=${cdp_port}
ENV
  cat > "${RUN_DIR}/ports.json" <<JSON
{ "searxng": ${searx_port}, "ui": ${ui_port}, "cdp": ${cdp_port} }
JSON
}

