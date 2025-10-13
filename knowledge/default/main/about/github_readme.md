![Apple Zero Runtime](res/header.png)
# Apple Zero Runtime
Apple Zero is the macOS-native evolution of the Cosmic Crisp / Agent Zero stack. It fuses a FastAPI gateway, async orchestrator, declarative tool registry, and composite memory into a single runtime that runs directly on macOS—no Docker layer required.

## Start Here
- **[Installation](installation.md):** Bootstrap the native toolchain, Python environment, and UI runner on macOS.
- **Runtime Overview (`README.md`):** Architectural deep dive, configuration tables, and testing workflows.
- **`python/runtime` package:** FastAPI app, dependency container, orchestrator, memory adapters, and token service.
- **`webui/`:** React client that consumes `/runtime` SSE timelines, manual browser takeover prompts, and configuration overrides.

## Runtime Highlights
- **Unified Gateway:** `python/runtime/api/app.py` exposes REST + SSE endpoints that power the CLI and web UI.
- **Async Orchestrator:** `python/runtime/agent/orchestrator.py` manages planner loops, subagents, tool dispatch, and memory persistence.
- **Composite Memory:** Local SQLite + FAISS embeddings (`python/runtime/memory`) with optional Mem0 hybrid retrieval.
- **Model Routing:** `python/runtime/models/router.py` selects providers (OpenAI, Anthropic, Gemini, LM Studio, etc.) using capability- and budget-aware policies.
- **Tooling Surface:** Browser, code, shell, search, and custom instruments registered via `python/runtime/tools` and surfaced in the UI timeline.

## Working Copies & Configuration
- **Environment:** `.env` or shell exports configure embeddings, router models, observability, and planner behaviour. Override defaults through `runtime.toml` for repeatable setups.
- **Memory Assets:** Runtime caches (`./tmp/`), memory database (`./data/runtime.sqlite`), and optional Mem0 credentials enable long-lived recall.
- **Logging & Metrics:** JSON logs stream to `./logs/runtime_observability.jsonl`; Prometheus metrics live at `/runtime/admin/metrics`; Helicone headers are emitted when `HELICONE_ENABLED=true`.

## Operational Checklist
1. Install prerequisites and create the virtual environment (see [Installation](installation.md)).
2. Populate `.env` with model API keys and router overrides.
3. Launch the runtime with `./run.sh` or `python -m python.runtime.api.app`.
4. Open `http://localhost:8080` for the UI, or interact through the CLI entrypoints in `run/`.
5. Rebuild embeddings or import legacy data using the helper CLIs in `python/runtime/tools` when migrating memory.

## Testing & Maintenance
- Run focused runtime tests with `python -m pytest tests/runtime -q` (ensure `PYTHONPATH=$(pwd)`).
- Use `python -m python.runtime.tools.reindex_memory` to keep embeddings fresh after large imports.
- Track regressions and telemetry via the observability stack before promoting configuration changes.

Apple Zero keeps the rapid iteration workflow of Agent Zero while embracing first-class macOS support. Keep these docs close when retuning models, extending tools, or migrating historical knowledge into the native runtime.
