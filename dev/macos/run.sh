#!/bin/bash

# Apple Zero macOS Run Script
# Native macOS execution without Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Apple Zero - macOS Native Runtime${NC}"
echo "======================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Change to project directory
cd "${PROJECT_DIR}"

echo "Project directory: ${PROJECT_DIR}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Please run setup.sh first.${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Check if required dependencies are installed
echo -e "${GREEN}Checking dependencies...${NC}"

# Check for pexpect
if ! python3 -c "import pexpect" 2>/dev/null; then
    echo -e "${RED}Error: pexpect not installed${NC}"
    echo "Please run: pip install pexpect ptyprocess"
    exit 1
fi

# Check for other critical dependencies
# Include playwright so browser tools load reliably. Avoid bare importing browser_use
# here since it may attempt to touch ~/.config during import; we do a safe check.
REQUIRED_MODULES=("flask" "litellm" "mcp" "chatterbox" "playwright" "aiohttp")
for module in "${REQUIRED_MODULES[@]}"; do
    if ! python3 -c "import $module" 2>/dev/null; then
        echo -e "${YELLOW}Warning: $module not found, installing requirements...${NC}"
        pip install -r requirements.txt
        break
    fi
done

# Safe check for browser_use with env sandbox to prevent touching ~/.config
if ! python3 - <<'PY' 2>/dev/null
import os
os.environ.setdefault('BROWSER_USE_CONFIG_DIR', 'tmp/browseruse')
os.environ.setdefault('XDG_CONFIG_HOME', 'tmp/xdg')
import browser_use  # noqa
print('OK')
PY
then
    echo -e "${YELLOW}Warning: browser_use import failed; reinstalling requirements...${NC}"
    pip install -r requirements.txt
fi

# Set environment variables (prefer isolated, project-local caches)
export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/searxng:${PYTHONPATH}"
export APPLE_ZERO_MODE="macos_native"
# Ensure FFmpeg@6 dylibs are visible to torchaudio/torio at process start.
FFMPEG6_PREFIX="/opt/homebrew/opt/ffmpeg@6"
if [ -d "${FFMPEG6_PREFIX}/lib" ]; then
    export COSMIC_FFMPEG_LIB_DIR="${FFMPEG6_PREFIX}/lib"
    export DYLD_LIBRARY_PATH="${COSMIC_FFMPEG_LIB_DIR}:${DYLD_LIBRARY_PATH}"
    export DYLD_FALLBACK_LIBRARY_PATH="${COSMIC_FFMPEG_LIB_DIR}:${DYLD_FALLBACK_LIBRARY_PATH}"
    export LDFLAGS="-L${COSMIC_FFMPEG_LIB_DIR} ${LDFLAGS}"
    export CPPFLAGS="-I${FFMPEG6_PREFIX}/include ${CPPFLAGS}"
    export PKG_CONFIG_PATH="${FFMPEG6_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}"
fi
# Prefer fast Metal kernels on Apple Silicon and avoid silent CPU fallbacks.
export PYTORCH_MPS_FAST_MATH=1
export PYTORCH_MPS_PREFER_METAL=1
export PYTORCH_ENABLE_MPS_FALLBACK=0
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9
# Keep Playwright assets under project tmp to avoid touching user Chrome
export PLAYWRIGHT_BROWSERS_PATH="${PROJECT_DIR}/tmp/playwright"
# Keep browser-use configs isolated to project tmp
export BROWSER_USE_CONFIG_DIR="${PROJECT_DIR}/tmp/browseruse"
export XDG_CONFIG_HOME="${PROJECT_DIR}/tmp/xdg"
export WEB_UI_HOST="127.0.0.1"

# Ensure SearXNG has a non-default secret to avoid startup exit
if [ -z "${SEARXNG_SECRET:-}" ]; then
  SEARXNG_SECRET=$(python3 - <<'PY'
import secrets
print(secrets.token_hex(16))
PY
  )
  export SEARXNG_SECRET
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found. Using default configuration.${NC}"
    echo "Create a .env file with your API keys for full functionality."
fi

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p tmp

echo -e "${GREEN}Starting Apple Zero UI Orchestrator...${NC}"
echo ""

SEARXNG_PID_FILE="${PROJECT_DIR}/tmp/searxng.pid"
SEARXNG_TERM_ID_FILE="${PROJECT_DIR}/tmp/searxng.terminal.id"
SEARXNG_PORT_DEFAULT=8888
mkdir -p "${PROJECT_DIR}/tmp"
rm -f "$SEARXNG_PID_FILE" "$SEARXNG_TERM_ID_FILE"

port_in_use() {
    local port=$1
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
}

find_free_port() {
    local preferred=${1:-$SEARXNG_PORT_DEFAULT}
    local port
    for port in $(seq "$preferred" $((preferred+10))); do
        if ! port_in_use "$port"; then
            echo "$port"
            return 0
        fi
    done
    echo "$preferred"
}

start_chrome_cdp() {
    # Launch user Chrome with remote debugging if not already running
    local port=${A0_CHROME_DEBUG_PORT:-9222}
    local profile_dir="${PROJECT_DIR}/tmp/chrome-debug-profile"
    mkdir -p "$profile_dir"

    export A0_CHROME_DEBUG_PORT="$port"
    export A0_CHROME_CDP_URL="http://127.0.0.1:${port}"

    # Try to find Chrome executable
    CHROME_BIN="${A0_CHROME_EXECUTABLE:-}"
    if [ -z "$CHROME_BIN" ]; then
        if [ -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]; then
            CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        elif command -v google-chrome >/dev/null 2>&1; then
            CHROME_BIN="$(command -v google-chrome)"
        elif command -v chromium >/dev/null 2>&1; then
            CHROME_BIN="$(command -v chromium)"
        fi
        export A0_CHROME_EXECUTABLE="$CHROME_BIN"
    fi

    # If port already in use, assume Chrome is already running with debugging enabled
    if port_in_use "$port"; then
        echo -e "${YELLOW}Chrome CDP port ${port} is already in use; reusing it.${NC}"
        return 0
    fi

    if [ -x "$CHROME_BIN" ]; then
        echo -e "${GREEN}Starting Chrome with remote debugging on port ${port}...${NC}"
        nohup "$CHROME_BIN" \
          --remote-debugging-port="$port" \
          --user-data-dir="$profile_dir" \
          --no-first-run --no-default-browser-check \
          about:blank \
          > "${PROJECT_DIR}/logs/chrome_debug.log" 2>&1 & echo $! > "${PROJECT_DIR}/tmp/chrome_debug.pid"
    else
        echo -e "${YELLOW}Chrome executable not found; skipping Chrome CDP launch.${NC}"
    fi
}

# Configure UI port now that helpers are available
UI_PORT_DEFAULT=8080
AGENT_TERM_ID_FILE="${PROJECT_DIR}/tmp/agent_terminal.terminal.id"

# Pick UI port (prefer default, fallback to free port)
if port_in_use "$UI_PORT_DEFAULT"; then
    WEB_UI_PORT=$(find_free_port "$UI_PORT_DEFAULT")
    echo -e "${YELLOW}Port $UI_PORT_DEFAULT in use; using $WEB_UI_PORT for UI.${NC}"
else
    WEB_UI_PORT=$UI_PORT_DEFAULT
fi
export WEB_UI_PORT

echo "Access the web interface at: http://localhost:${WEB_UI_PORT}"
echo "Press Ctrl+C to stop the server"
echo ""

start_searxng_terminal() {
    echo -e "${GREEN}Starting SearXNG in a new Terminal tab...${NC}"
    export SEARXNG_SETTINGS_PATH="${PROJECT_DIR}/searxng/settings.yml"

    # If a previous SearXNG from this project is running, stop it cleanly
    if [ -f "$SEARXNG_PID_FILE" ]; then
        oldpid=$(cat "$SEARXNG_PID_FILE" 2>/dev/null || true)
        if [ -n "$oldpid" ] && ps -p "$oldpid" >/dev/null 2>&1; then
            echo -e "${YELLOW}Stopping previous SearXNG (PID $oldpid)...${NC}"
            kill "$oldpid" 2>/dev/null || true
            wait "$oldpid" 2>/dev/null || true
        fi
        rm -f "$SEARXNG_PID_FILE"
    fi

    # Choose port: prefer default, otherwise first free
    if port_in_use "$SEARXNG_PORT_DEFAULT"; then
        SEARXNG_PORT=$(find_free_port "$SEARXNG_PORT_DEFAULT")
        echo -e "${YELLOW}Port $SEARXNG_PORT_DEFAULT in use; using $SEARXNG_PORT.${NC}"
    else
        SEARXNG_PORT=$SEARXNG_PORT_DEFAULT
    fi
    export SEARXNG_PORT
    export SEARXNG_BIND_ADDRESS=127.0.0.1
    export SEARXNG_URL="http://127.0.0.1:${SEARXNG_PORT}/search"
    # Persist URL to .env so helper modules reading dotenv get the correct port
    python3 - <<PY || true
from python.helpers import dotenv
dotenv.save_dotenv_value('SEARXNG_URL', 'http://127.0.0.1:'+str(${SEARXNG_PORT})+'/search')
PY

    # Ensure SearXNG dependencies are installed (check for httpx)
    if ! python3 -c "import httpx" 2>/dev/null; then
        echo -e "${YELLOW}Installing SearXNG dependencies...${NC}"
        pip install -r searxng/requirements.txt -r searxng/requirements-server.txt
    fi

    # Use osascript to open a new tab in the current Terminal window and run SearXNG.
    # We write the PID via exec. Fallback to in-process background if AppleScript fails.
    if command -v osascript >/dev/null 2>&1; then
        # Write a small runner script to avoid quoting issues with spaces
        RUNNER_SCRIPT="${PROJECT_DIR}/tmp/run_searxng.sh"
        cat > "$RUNNER_SCRIPT" <<'EOF'
#!/usr/bin/env bash
set -e
cd "__PROJECT_DIR__"
source venv/bin/activate
export PYTHONPATH="__PROJECT_DIR__":"__PROJECT_DIR__"/searxng:$PYTHONPATH
export SEARXNG_SETTINGS_PATH="__PROJECT_DIR__"/searxng/settings.yml
export SEARXNG_BIND_ADDRESS=127.0.0.1
export SEARXNG_PORT=__SEARXNG_PORT__
export SEARXNG_URL="http://127.0.0.1:${SEARXNG_PORT}/search"
export SEARXNG_SECRET="__SEARXNG_SECRET__"
if curl -sS -o /dev/null -m 1 "http://127.0.0.1:${SEARXNG_PORT}/healthz"; then
  echo "SearXNG already running on port ${SEARXNG_PORT}. Exiting runner."
  exit 0
fi
printf '\e]0;SearXNG\a'
echo $$ > "__PROJECT_DIR__"/tmp/searxng.pid
exec "__PROJECT_DIR__"/venv/bin/python3 -m searx.webapp >> "__PROJECT_DIR__"/logs/searxng.log 2>&1
EOF
        # Replace placeholders with actual values (portable sed on macOS)
        sed -i '' -e "s|__PROJECT_DIR__|${PROJECT_DIR//|/\|}|g" "$RUNNER_SCRIPT"
        sed -i '' -e "s|__SEARXNG_PORT__|${SEARXNG_PORT}|g" "$RUNNER_SCRIPT"
        sed -i '' -e "s|__SEARXNG_SECRET__|${SEARXNG_SECRET}|g" "$RUNNER_SCRIPT"
        chmod +x "$RUNNER_SCRIPT"

        # Launch the runner script in a NEW WINDOW
        if TERM_REF=$(osascript <<APPLESCRIPT
tell application "Terminal"
    activate
    set newTab to do script ("bash " & quoted form of POSIX path of "${RUNNER_SCRIPT}")
    delay 0.3
    set winId to id of front window
    return (winId as string)
end tell
APPLESCRIPT
        ); then
            echo "$TERM_REF" > "$SEARXNG_TERM_ID_FILE"
        else
            echo -e "${RED}AppleScript failed; could not launch SearXNG tab.${NC}"
            echo -e "${YELLOW}Fallback: starting SearXNG in this terminal...${NC}"
            PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/searxng:$PYTHONPATH" SEARXNG_SETTINGS_PATH="${PROJECT_DIR}/searxng/settings.yml" SEARXNG_BIND_ADDRESS=127.0.0.1 SEARXNG_PORT="${SEARXNG_PORT}" SEARXNG_URL="http://127.0.0.1:${SEARXNG_PORT}/search" SEARXNG_SECRET="${SEARXNG_SECRET}" python3 -m searx.webapp & echo $! > "$SEARXNG_PID_FILE"
        fi
    else
        echo -e "${YELLOW}osascript not found; starting SearXNG in this terminal...${NC}"
        PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/searxng:$PYTHONPATH" SEARXNG_SETTINGS_PATH="${PROJECT_DIR}/searxng/settings.yml" SEARXNG_BIND_ADDRESS=127.0.0.1 SEARXNG_PORT="${SEARXNG_PORT}" SEARXNG_URL="http://127.0.0.1:${SEARXNG_PORT}/search" SEARXNG_SECRET="${SEARXNG_SECRET}" python3 -m searx.webapp & echo $! > "$SEARXNG_PID_FILE"
    fi
}

start_searxng_background() {
    echo -e "${GREEN}Starting SearXNG in background (no Terminal)...${NC}"
    export SEARXNG_SETTINGS_PATH="${PROJECT_DIR}/searxng/settings.yml"
    # Choose port: prefer default, otherwise first free
    if port_in_use "$SEARXNG_PORT_DEFAULT"; then
        SEARXNG_PORT=$(find_free_port "$SEARXNG_PORT_DEFAULT")
        echo -e "${YELLOW}Port $SEARXNG_PORT_DEFAULT in use; using $SEARXNG_PORT.${NC}"
    else
        SEARXNG_PORT=$SEARXNG_PORT_DEFAULT
    fi
    export SEARXNG_PORT
    export SEARXNG_BIND_ADDRESS=127.0.0.1
    export SEARXNG_URL="http://127.0.0.1:${SEARXNG_PORT}/search"
    # ensure PYTHONPATH is set even if shell was not activated
    export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/searxng:${PYTHONPATH}"
    python3 - <<PY || true
from python.helpers import dotenv
dotenv.save_dotenv_value('SEARXNG_URL', 'http://127.0.0.1:'+str(${SEARXNG_PORT})+'/search')
PY
    nohup "${PROJECT_DIR}/venv/bin/python3" -m searx.webapp >> "${PROJECT_DIR}/logs/searxng.log" 2>&1 & echo $! > "$SEARXNG_PID_FILE"
}

start_agent_terminal() {
    echo -e "${GREEN}Starting Agent Terminal in a new Terminal tab...${NC}"

    # Write a runner that waits for UI and then attaches
    RUNNER_SCRIPT="${PROJECT_DIR}/tmp/run_agent_terminal.sh"
    cat > "$RUNNER_SCRIPT" <<'EOF'
#!/usr/bin/env bash
set -e
cd "__PROJECT_DIR__"
source venv/bin/activate
export PYTHONPATH="__PROJECT_DIR__":"__PROJECT_DIR__"/searxng:$PYTHONPATH
UI_PORT="${WEB_UI_PORT:-__WEB_UI_PORT__}"
export WEB_UI_PORT="$UI_PORT"

# Wait for UI to become ready
echo "[agent-terminal] Waiting for UI on port ${UI_PORT}..."
for i in {1..120}; do
  if curl -sS -o /dev/null -m 1 "http://127.0.0.1:${UI_PORT}/health"; then
    break
  fi
  sleep 0.5
done

printf '\e]0;Agent Terminal\a'
# Let the client compute base from WEB_UI_PORT env; pass only cwd
exec "__PROJECT_DIR__"/venv/bin/python3 -m python.cli.agent_terminal --cwd "__PROJECT_DIR__"
EOF
    sed -i '' -e "s|__PROJECT_DIR__|${PROJECT_DIR//|/\|}|g" "$RUNNER_SCRIPT"
    sed -i '' -e "s|__WEB_UI_PORT__|${WEB_UI_PORT}|g" "$RUNNER_SCRIPT"
    chmod +x "$RUNNER_SCRIPT"

    if command -v osascript >/dev/null 2>&1; then
        if TERM_REF=$(osascript <<APPLESCRIPT
tell application "Terminal"
    activate
    set newTab to do script ("bash " & quoted form of POSIX path of "${RUNNER_SCRIPT}")
    delay 0.3
    set winId to id of front window
    return (winId as string)
end tell
APPLESCRIPT
        ); then
            echo "$TERM_REF" > "$AGENT_TERM_ID_FILE"
        fi
    fi
}

close_agent_terminal() {
    if [ -f "$AGENT_TERM_ID_FILE" ] && command -v osascript >/dev/null 2>&1; then
        TERM_REF=$(cat "$AGENT_TERM_ID_FILE" 2>/dev/null || true)
        if [ -n "$TERM_REF" ]; then
            if [[ "$TERM_REF" == *":"* ]]; then
                TERM_WIN_ID="${TERM_REF%%:*}"
                TERM_TAB_INDEX="${TERM_REF##*:}"
                osascript <<APPLESCRIPT >/dev/null 2>&1 || true
tell application "Terminal"
    try
        close tab ${TERM_TAB_INDEX} of window id ${TERM_WIN_ID}
    end try
end tell
APPLESCRIPT
            else
                osascript <<APPLESCRIPT >/dev/null 2>&1 || true
tell application "Terminal"
    try
        close window id ${TERM_REF}
    end try
end tell
APPLESCRIPT
            fi
        fi
    fi
}

close_searxng_terminal() {
    if [ -f "$SEARXNG_TERM_ID_FILE" ] && command -v osascript >/dev/null 2>&1; then
        TERM_REF=$(cat "$SEARXNG_TERM_ID_FILE" 2>/dev/null || true)
        if [ -n "$TERM_REF" ]; then
            if [[ "$TERM_REF" == *":"* ]]; then
                TERM_WIN_ID="${TERM_REF%%:*}"
                TERM_TAB_INDEX="${TERM_REF##*:}"
                osascript <<APPLESCRIPT >/dev/null 2>&1 || true
tell application "Terminal"
    try
        close tab ${TERM_TAB_INDEX} of window id ${TERM_WIN_ID}
    end try
end tell
APPLESCRIPT
            else
                # Backward compatibility: stored only window id; close the window
                osascript <<APPLESCRIPT >/dev/null 2>&1 || true
tell application "Terminal"
    try
        close window id ${TERM_REF}
    end try
end tell
APPLESCRIPT
            fi
        fi
    fi
}

wait_for_searxng() {
    local port=${SEARXNG_PORT:-$SEARXNG_PORT_DEFAULT}
    echo -e "${GREEN}Waiting for SearXNG to become ready on 127.0.0.1:${port}...${NC}"
    local attempts=200
    local ok=0
    for i in $(seq 1 ${attempts}); do
        if curl -sS -o /dev/null -m 1 "http://127.0.0.1:${port}/healthz" || \
           curl -sS -o /dev/null -m 1 "http://127.0.0.1:${port}/"; then
            ok=1
            break
        fi
        sleep 0.5
    done
    if [ "$ok" = "1" ]; then
        echo -e "${GREEN}SearXNG is up.${NC}"
        return 0
    else
        echo -e "${YELLOW}SearXNG did not respond in time; continuing anyway.${NC}"
        return 1
    fi
}

start_ui_server() {
    echo -e "${GREEN}Starting Flask dev server on port ${WEB_UI_PORT} in this window...${NC}"
    # Start UI in background but keep logs in this terminal
    python3 run_ui.py --port "${WEB_UI_PORT}" & echo $! > "${PROJECT_DIR}/tmp/ui.pid"
}

wait_for_ui() {
    local port=${WEB_UI_PORT}
    echo -e "${GREEN}Waiting for UI to become ready on 127.0.0.1:${port}...${NC}"
    local attempts=200
    local ok=0
    for i in $(seq 1 ${attempts}); do
        if curl -sS -o /dev/null -m 1 "http://127.0.0.1:${port}/health"; then
            ok=1
            break
        fi
        sleep 0.5
    done
    if [ "$ok" = "1" ]; then
        echo -e "${GREEN}UI is up.${NC}"
        return 0
    else
        echo -e "${RED}UI did not respond in time on port ${port}.${NC}"
        return 1
    fi
}

# Orchestration: SearXNG -> UI -> Agent Terminal
start_searxng_terminal
if ! wait_for_searxng; then
  echo -e "${RED}SearXNG did not become ready in time. Check logs at ${PROJECT_DIR}/logs/searxng.log${NC}"
fi
# Prefer headless Playwright — do NOT attach to user's Chrome
# Set A0_ENABLE_CDP=1 if you explicitly want to attach to a running Chrome.
if [ "${A0_ENABLE_CDP:-}" = "1" ]; then
  start_chrome_cdp
else
  echo -e "${GREEN}Using isolated headless Playwright browser (no Chrome CDP).${NC}"
fi

# Clean shutdown trap: kill SearXNG process and close Terminal window
cleanup() {
    echo -e "\n${YELLOW}Shutting down Apple Zero...${NC}"
    # Stop UI if running (launched in this window)
    if [ -f "${PROJECT_DIR}/tmp/ui.pid" ]; then
        UI_PID=$(cat "${PROJECT_DIR}/tmp/ui.pid" 2>/dev/null || true)
        if [ -n "$UI_PID" ]; then
            kill "$UI_PID" 2>/dev/null || true
            wait "$UI_PID" 2>/dev/null || true
        fi
        rm -f "${PROJECT_DIR}/tmp/ui.pid"
    fi
    if [ -f "$SEARXNG_PID_FILE" ]; then
        SEARXNG_PID=$(cat "$SEARXNG_PID_FILE" 2>/dev/null || true)
        if [ -n "$SEARXNG_PID" ]; then
            kill "$SEARXNG_PID" 2>/dev/null || true
            wait "$SEARXNG_PID" 2>/dev/null || true
        fi
        rm -f "$SEARXNG_PID_FILE"
    fi
    close_searxng_terminal
    close_agent_terminal
    # Close UI tab/window if we opened one
    if [ -f "${PROJECT_DIR}/tmp/ui.terminal.id" ] && command -v osascript >/dev/null 2>&1; then
        TERM_REF=$(cat "${PROJECT_DIR}/tmp/ui.terminal.id" 2>/dev/null || true)
        if [ -n "$TERM_REF" ]; then
            if [[ "$TERM_REF" == *":"* ]]; then
                TERM_WIN_ID="${TERM_REF%%:*}"
                TERM_TAB_INDEX="${TERM_REF##*:}"
                osascript <<APPLESCRIPT >/dev/null 2>&1 || true
tell application "Terminal"
    try
        close tab ${TERM_TAB_INDEX} of window id ${TERM_WIN_ID}
    end try
end tell
APPLESCRIPT
            else
                osascript <<APPLESCRIPT >/dev/null 2>&1 || true
tell application "Terminal"
    try
        close window id ${TERM_REF}
    end try
end tell
APPLESCRIPT
            fi
        fi
        rm -f "${PROJECT_DIR}/tmp/ui.terminal.id"
    fi
    rm -f "$SEARXNG_TERM_ID_FILE"
    rm -f "$AGENT_TERM_ID_FILE"
    if [ -f "${PROJECT_DIR}/tmp/chrome_debug.pid" ]; then
        CH_PID=$(cat "${PROJECT_DIR}/tmp/chrome_debug.pid" 2>/dev/null || true)
        if [ -n "$CH_PID" ]; then
            kill "$CH_PID" 2>/dev/null || true
            wait "$CH_PID" 2>/dev/null || true
        fi
        rm -f "${PROJECT_DIR}/tmp/chrome_debug.pid"
    fi
    exit 0
}
trap cleanup INT TERM

# Start the UI server in this window and wait for it
start_ui_server
wait_for_ui || true

# Optionally launch Agent Terminal (set A0_START_AGENT_TERMINAL=1)
if [ "${A0_START_AGENT_TERMINAL:-}" = "1" ]; then
  start_agent_terminal
fi

# Keep this orchestrator running for cleanup (Ctrl+C to stop)
echo -e "${GREEN}All components launched. Press Ctrl+C here to stop and clean up.${NC}"
UI_PID=$(cat "${PROJECT_DIR}/tmp/ui.pid" 2>/dev/null || true)
if [ -n "$UI_PID" ]; then
  wait "$UI_PID"
else
  while true; do sleep 3600; done
fi
