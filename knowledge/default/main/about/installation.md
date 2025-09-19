# Install Apple Zero on macOS

Apple Zero ships with a native macOS toolchain—no Docker images or Linux VMs. Follow the steps below to bootstrap the runtime, install the web UI, and connect your preferred models.

## Prerequisites
- macOS 12 Monterey or newer
- Homebrew (auto-installed by the setup script if missing)
- Python 3.11 available on PATH (`brew install python@3.11`)
- 15–20 GB free disk space for Playwright browsers, logs, and embeddings
- API keys for your preferred providers (OpenAI, Anthropic, Gemini, LM Studio, etc.)

> Tip: Clone the repository somewhere under your user directory to avoid SIP permission prompts.

## Fast Path (recommended)

```bash
# from the repository root
./dev/macos/setup.sh
```

The script:
1. Ensures Homebrew exists, then installs native dependencies (FFmpeg, PortAudio, Poppler, Tesseract, etc.).
2. Creates a fresh `venv/` with Python 3.11 and installs `requirements.txt` plus SearXNG extras.
3. Downloads Playwright Chromium binaries into `./tmp/playwright` so browser tools can launch without touching system Chrome.

When setup completes, start the runtime and UI with:

```bash
./dev/macos/run.sh
```

`run.sh` activates the virtualenv, checks critical Python modules, isolates browser-use caches under `./tmp`, seeds a SearXNG instance, and serves the Apple Zero web client (defaults to `http://localhost:8080`). Use `Ctrl+C` to stop the orchestrator.

## Initial Configuration
1. Copy the example environment file: `cp .env.example .env`.
2. Populate API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, etc.).
3. Optional: create `runtime.toml` for repeatable defaults (embeddings provider, router policies, token budgets).
4. Run `./dev/macos/run.sh` and sign in to providers or LM Studio as needed.

### Verifying the Install
- Visit `http://localhost:8080` to confirm the UI loads and receives SSE timelines.
- From another terminal, test the FastAPI health check: `curl http://127.0.0.1:8080/runtime/health`.
- Run focused runtime tests (after activating the venv):
  ```bash
  source venv/bin/activate
  PYTHONPATH="$(pwd)" python -m pytest tests/runtime -q
  ```

## Manual Setup (for custom environments)
If you prefer to manage dependencies yourself:

```bash
# Ensure Python 3.11 is the active interpreter
python3.11 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -r searxng/requirements.txt -r searxng/requirements-server.txt  # optional search stack

# Install Playwright assets into the repo-local cache
export PLAYWRIGHT_BROWSERS_PATH="$(pwd)/tmp/playwright"
playwright install chromium
```

Then launch the runtime:

```bash
export PYTHONPATH="$(pwd):$(pwd)/searxng"
export PLAYWRIGHT_BROWSERS_PATH="$(pwd)/tmp/playwright"
export BROWSER_USE_CONFIG_DIR="$(pwd)/tmp/browseruse"
export XDG_CONFIG_HOME="$(pwd)/tmp/xdg"
python -m python.runtime.api.app
```

Use `open http://localhost:8080` to reach the UI when running `./dev/macos/run.sh`, or connect to the FastAPI server via the CLI scripts under `run/`.

## Optional Integrations
- **Mem0 Hybrid Memory:** set `MEM0_ENABLED=true`, `MEM0_API_KEY`, and `MEM0_BASE_URL` to merge remote recall with the local SQLite store. Rebuild local vectors with `python -m python.runtime.tools.reindex_memory` after large imports.
- **Chrome Remote Debugging:** export `A0_CHROME_EXECUTABLE` and `A0_CHROME_DEBUG_PORT` before calling `run.sh` if you need a specific Chrome build for the browser tool.
- **Observability:** set `HELICONE_ENABLED=true` plus `HELICONE_*` keys to forward telemetry. Logs land in `./logs/runtime_observability.jsonl`; Prometheus metrics are exposed at `/runtime/admin/metrics`.

## Troubleshooting
- Missing dependencies? Re-run `./dev/macos/setup.sh` after updating Homebrew or Python.
- Browser tool failing to launch? Ensure `playwright install chromium` completed and that `PLAYWRIGHT_BROWSERS_PATH` points to `./tmp/playwright`.
- SSE stream drops? Check `logs/` for runtime errors and verify your API quotas.

Apple Zero trades Docker for a lightweight macOS-native stack. Once the runtime and UI are up, you can iterate on agents, tools, prompts, and memory workflows directly from the host machine.
