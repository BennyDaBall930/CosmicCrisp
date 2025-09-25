# 🚀 Apple Zero Runtime

<div align="center">
<h2>🤖 The Future of AI Agents on macOS 🍎</h2>

<p><em>"The macOS-native evolution of Agent Zero - where AI meets Apple Silicon magic!"</em></p>

[![Stars](https://img.shields.io/github/stars/BennyDaBall930/cosmiccrisp?style=for-the-badge&logo=github)](https://github.com/BennyDaBall930/cosmiccrisp)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![macOS](https://img.shields.io/badge/platform-macOS-lightgrey?style=for-the-badge&logo=apple)](https://www.apple.com/macos/)

</div>

---

## 🤔 Why Apple Zero?

Ever wish your Mac could just... do stuff? Like actually be helpful beyond reminding you about your next meeting?

**Apple Zero** is here to change that! We're talking **AI-powered automation** that runs natively on your Mac with zero Docker headaches. Whether you need to research complex topics, write code, browse the web, or even plan your next vacation, Apple Zero has got you covered!

### ✨ What Makes It Special?

- 🍎 **Truly Native** - No containers, no VMs, just pure macOS goodness
- 🔧 **Modular & Extensible** - Mix and match AI models, tools, and capabilities
- 🧠 **Smart Memory** - Remembers your context across sessions
- 🗣️ **Voice Powered** - Make your Mac speak in your favorite voice (native voice cloning)
- 🚀 **Blazing Fast** - Apple Silicon optimized with proper MPS support
- 🎯 **Agent-Like** - Handles complex multi-step tasks autonomously

> "It's like having a team of AI interns who actually follow instructions." - A satisfied user (probably)

---

## 🏗️ Architecture at a Glance

```
User Request 🚀 → Web UI/CLI → FastAPI Gateway → Container Runtime
                                                  ↓
                                        ┌─────────────────┐
                                        │   Orchestrator   │  ← Goal → Plan → Execute Loop
                                        └─────────────────┘
                                                ↓
                        ┌─────────────────┬─────────────────┬─────────────────┐
                        │     Memory      │     Model       │     Tools       │
                        │   (SQLite +     │   Router        │   (Browser,     │
                        │    FAISS)       │                 │    Shell, etc)  │
                        └─────────────────┴─────────────────┴─────────────────┘
```

---

## 🎯 Key Features

### 🧠 Smart Memory System
Our hybrid memory combines local SQLite + vector search with optional Mem0 for that extra brain boost!

### 🗣️ Chatterbox TTS
Gone are the days of robotic voices! Apple Zero speaks like a human... most of the time. (We're working on improving latency...)

### 🛠️ Tool Ecosystem
Browser automation? ✓ Code execution? ✓ File operations? ✓ Email sending? ✓ Database queries? ✓

### 🔄 Model Router
Automatically picks the best AI model for your task. Efficient as hell, expensive as heaven.

### 📊 Live Observability
Track everything with metrics, logs, and shiny dashboards. Because who doesn't love graphs?

---

## 🚀 Quick Start (You're 2 Minutes Away from AI Glory!)

### Prerequisites
- macOS 12+ (sorry, Tim Cook's making it hard for older Macs)
- Python 3.10+
- A caffeine source (coffee/tea/energy drinks)

### 🚀 Launch Sequence

```bash
# 1. Bootstrap everything (this is the magic button)
./dev/macos/setup.sh

# 2. Blast off!
./dev/macos/run.sh

# 3. Your AI awaits...
open http://localhost:8080
```

That's it! 🤯

---

## ⚙️ Configuration (The Important Bits)

Apple Zero is highly configurable. Here's the beer-to-water ratio you need:

### Essential Environment Variables

```env
# AI Models (the brain)
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_other_key

# Memory (the long-term brain)
MEM0_ENABLED=true  # For that extra storage
MEM0_API_KEY=mem0_key

# TTS Features (make it speak!)
ENABLE_TTS=true
TTS_LANGUAGE_ID=en  # or 'es', 'fr', 'de', your call
TTS_EXAGGERATION=0.5  # 0=boring robot, 1=very excited robot
```

### Advanced Configuration (`runtime.toml`)

```toml
[router]
default_model = "gpt-4o"  # Or whatever you can afford

[[router.models]]
name = "gpt-4o"
priority = 1  # Higher = preferred
capabilities = ["general", "code", "research"]

[embeddings]
provider = "openai"  # Local options available too
model = "text-embedding-3-large"

[memory]
db_path = "./data/runtime.sqlite"
```

## 📋 What's Inside This Delicious README

Jump to your favorite section:

- [🎬 Voice & TTS Magic](#-voice--tts-magic)
- [🧠 Memory Systems (The Brain)](#-memory-systems-the-brain)
- [⚠️ Migration Guide](#-migration-guide-from-older-cosmic-crisp-versions)
- [🧪 Testing & Quality Assurance](#-testing--quality-assurance)
- [🍎 macOS-Specific Goodness](#-macos-specific-goodness)
- [🙏 Credits & Attribution](#-credits--attribution)

---

## 🎬 Voice & TTS Magic

Want your Mac to finally have something interesting to say? 🎙️

**Apple Zero** ditches the robotic monotone for **Chatterbox TTS** - our brand new voice system that's actually pleasant to listen to!

### 🚀 What Makes Our Voices So Dang Good?

- **Human-Like Speech**: Feels like a real person is talking to you
- **Emotional Range**: Make it whisper sweet nothings or yell enthusiastically
- **Multi-Language Party**: English, French, German, Spanish... your choice!
- **Voice Cloning**: Upload a reference audio and make it sound like your favorite celebrity (legal version, of course)
- **GPU Goodness**: Apple Silicon MPS acceleration for ultra-smooth performance

### ⚡ Quick Voice Configuration

```env
# Make your Mac chatty
ENABLE_TTS=true

# Voice personality (0=dull robot, 1=wild excitement)
TTS_EXAGGERATION=0.6

# Language (en/fr/de/es)
TTS_LANGUAGE_ID=en
```

*Pro tip: Set TTS_EXAGGERATION to 0.9 if you're feeling dramatic today!*

### 🔄 Migrating from "Robot Beeps"?

We ditched the old Kokoro TTS engine (rip) for Chatterbox. It's that good. Just update your config - we'll handle the magic!

## Voice & TTS Features

**BREAKING CHANGE**: The Kokoro TTS engine has been completely removed and replaced with Chatterbox TTS for enhanced voice quality and features.

### Key Improvements Over Kokoro
- **Enhanced Audio Quality**: Advanced neural voice synthesis with emotion control
- **Multi-Language Support**: Native multilingual capabilities with proper accent handling
- **Reference Voice Cloning**: Ability to style voice output based on audio samples
- **Fine-Grained Control**: Adjustable emotion intensity, CFG (classifier-free guidance), and device targeting
- **Performance**: Optimized for both CPU and GPU acceleration (MPS, CUDA, CPU fallback)

### Configuration Options

| Feature | Environment Variable | Default | Description |
| --- | --- | --- | --- |
| **System Enable/Disable** | `ENABLE_TTS` | `true` | Completely disable TTS if set to `false` |
| **Voice Config File** | `TTS_CONFIG_PATH` | (runtime lookup) | Path to JSON/TOML configuration file |
| **Multilingual Model** | `TTS_MULTILINGUAL` | `false` | Enable multilingual Chatterbox model |
| **Sample Rate** | `TTS_SAMPLE_RATE` | `24000` | Audio output sample rate (Hz) |
| **Emotion Intensity** | `TTS_EXAGGERATION` | `0.5` | Emotional expression level (0.0-1.0) |
| **Voice Consistency** | `TTS_CFG` | `0.35` | Classifier-free guidance strength |
| **Reference Audio** | `TTS_AUDIO_PROMPT_PATH` | (none) | WAV file for voice style reference |
| **Language** | `TTS_LANGUAGE_ID` | `"en"` | Language code for multilingual model |
| **Max Text Length** | `TTS_MAX_CHARS` | `600` | Maximum characters per synthesis chunk |
| **Silence Between Chunks** | `TTS_JOIN_SILENCE_MS` | `120` | Milliseconds of silence between text chunks |

### TTS Configuration Examples

**Basic Setup (.env):**
```env
# Simple English voice with moderate emotion
ENABLE_TTS=true
TTS_EXAGGERATION=0.6
TTS_CFG=0.4
TTS_LANGUAGE_ID=en
```

**Advanced Setup (runtime.toml):**
```toml
[tts.chatterbox]
multilingual = false  # Set to true for language support
sample_rate = 24000
exaggeration = 0.5   # Voice emotion intensity
cfg = 0.35          # Voice consistency
max_chars = 600     # Text chunk size
join_silence_ms = 120 # Gap between chunks

# Optional reference voice (copy-paste audio sample)
# audio_prompt_path = "./reference-voice.wav"

# Device targeting (auto-detects MPS/CUDA/CPU)
# device = "auto"  # or "mps", "cuda", "cpu"
```

**Multilingual Voices:**
```env
TTS_MULTILINGUAL=true
TTS_LANGUAGE_ID=de  # German, French, Spanish, etc.
```

### Fallback to Browser TTS

If Chatterbox fails to initialize or TTS is disabled, the system falls back to pyttsx-based browser TTS. This provides basic cross-platform compatibility but lacks the advanced features of Chatterbox.

### Migration from Kokoro

**❌ Breaking Changes:**
- `tts_kokoro` settings in configuration files are completely ignored
- Kokoro-dependent voice parameters must be migrated to Chatterbox equivalents
- Existing voice configuration files need format updates

**✅ Migration Path:**
1. Remove all `tts_kokoro` configuration blocks
2. Add Chatterbox configuration using the settings above
3. Adjust emotion levels (Kokoro values may not translate directly)
4. Test audio output quality after migration
5. Enable `ENABLE_TTS=false` temporarily to use browser fallback during transition

## Quick Start

### Prerequisites

- macOS 12+
- Python 3.10+
- Homebrew (installer in `dev/macos/setup.sh` will bootstrap if missing)

### Bootstrap

```bash
# 1. Install native dependencies, virtualenv, Playwright cache
./dev/macos/setup.sh

# 2. Launch the combined FastAPI + UI runtime
./dev/macos/run.sh

# 3. Open the web client
open http://localhost:8080
```

To run the FastAPI app directly without the helper script:

```bash
source venv/bin/activate
python -m python.runtime.api.app
```

## Configuration Essentials

Runtime configuration merges environment variables with an optional `runtime.toml`. The loader lives in `python/runtime/config.py` and exposes the following top-level sections: `embeddings`, `memory`, `tokens`, `prompts`, `observability`, `agent`, `tools`, and `router`.

| Key | Purpose | Example Overrides |
| --- | --- | --- |
| `EMBEDDINGS_PROVIDER`, `EMBEDDINGS_MODEL` | Configure embedding service (OpenAI, local LM Studio via LiteLLM) | `EMBEDDINGS_PROVIDER=openai`, `EMBEDDINGS_MODEL=text-embedding-3-large` |
| `MEM0_ENABLED`, `MEM0_API_KEY`, `MEM0_BASE_URL` | Enable hybrid Mem0 retrieval alongside SQLite | `MEM0_ENABLED=true` |
| `SUMMARIZER_MODEL`, `TOKEN_BUDGET_*` | Override default token budgets and summarizer model | `TOKEN_BUDGET_GPT_4O=196000` |
| `ROUTER_DEFAULT_MODEL`, `ROUTER_MODELS` | Declare routing policies and capabilities | `ROUTER_MODELS='[{"name":"gpt-4o-mini","priority":1}]'` |
| `HELICONE_ENABLED`, `HELICONE_BASE_URL`, `HELICONE_API_KEY` | Toggle Helicone proxy & observability headers | `HELICONE_ENABLED=true` |

### Sample `runtime.toml`

```toml
[embeddings]
provider = "openai"
model = "text-embedding-3-large"

[memory]
db_path = "./data/runtime.sqlite"
mem0_enabled = false

[tokens]
summarizer_model = "gpt-4.1-mini"

[router]
default_model = "gpt-4o"
default_strategy = "balanced"
[[router.models]]
name = "gpt-4o"
priority = 1
max_context = 128000
capabilities = ["general", "code"]

[[router.models]]
name = "lm-studio:gpt4all"
provider = "openai"
priority = 2
max_context = 32000
is_local = true
metadata.api_base = "http://localhost:1234/v1"
```

## Memory & Context Management

Apple Zero uses a composite memory stack (`python/runtime/memory`) combining a local SQLite+FAISS store with optional Mem0 hybrid retrieval.

| Provider | Model / Endpoint | Config Keys | Notes |
| --- | --- | --- | --- |
| OpenAI | `text-embedding-3-large` (default) | `EMBEDDINGS_PROVIDER=openai`, `EMBEDDINGS_MODEL=text-embedding-3-large` | High-quality embeddings, cached in `./tmp/embeddings.sqlite`. |
| LiteLLM → LM Studio | Example `text-embedding-3-small` served by LM Studio | `EMBEDDINGS_PROVIDER=local_mlx`, `EMBEDDINGS_MODEL=lm-studio`, set `OPENAI_BASE_URL` to LM Studio server | Uses LiteLLM-compatible REST API. |
| Null | Deterministic stub for tests | `EMBEDDINGS_PROVIDER=null` | Returns fixed vectors for offline testing. |

### Mem0 Hybrid Retrieval

Set the following to enable Mem0 alongside the local store:

```env
MEM0_ENABLED=true
MEM0_API_KEY=your_mem0_token
MEM0_BASE_URL=https://your-mem0-host
```

When enabled, the adapter merges remote results with the local SQLite store and defers reindex operations to the fallback store when possible.

### CLI Tools

Two helper CLIs ship with the runtime:

```bash
# Rebuild vectors for all (or a specific) sessions
python -m python.runtime.tools.reindex_memory --verbose
python -m python.runtime.tools.reindex_memory --session research-2024

# Import legacy JSON/NDJSON exports into the new store
python -m python.runtime.tools.migrate_memory ./memory-exports --session backlog --kind fact
```

Both utilities respect your environment configuration and will load embeddings/memory using the same runtime container settings.

## Token Budgets & Context Fitting

`python/runtime/tokenizer/token_service.py` manages window fitting. Default budgets ship with the runtime and can be overridden via `runtime.toml` or environment variables.

| Model (normalized) | Default Window |
| --- | --- |
| `gpt-4o` | 128,000 |
| `gpt-4.1-mini` | 64,000 |
| `claude-3-5` | 200,000 |
| `gemini-2.5-pro` | 1,000,000 |
| `local-mlx` | 8,192 |

Override options:

- Inline JSON: `TOKEN_BUDGETS='{"gpt-4o":196000, "claude-3-5":180000}'`
- Per-model env: `TOKEN_BUDGET_GPT_4O=196000`
- Default summarizer: `SUMMARIZER_MODEL=claude-3-5`

The service preserves the system prompt, injects summarised memory snippets, and trims conversation history. Summaries are emitted as synthetic `system` messages annotated with "Conversation summary".

## Subagents & Autonomy

The orchestrator (`python/runtime/agent/orchestrator.py`):

- Runs a goal → analyze → execute → summarize loop.
- Seeds a `TaskPlanner` priority queue and persists task outcomes to memory.
- Spawns subagents for tool classes such as `browser` and `code`, respecting `AGENT_SUBAGENT_MAX_DEPTH` and `AGENT_SUBAGENT_TIMEOUT` from `AgentConfig`.
- Streams events via the `EventBus` for the UI timeline (`task_started`, `tool_start`, `browser_hil_required`, etc.).

Tune behaviour via environment variables:

```env
AGENT_MAX_LOOPS=25
AGENT_SUBAGENT_MAX_DEPTH=2
AGENT_SUBAGENT_TIMEOUT=90
AGENT_PERSONA=concise
```

## UI & Human-in-the-Loop Browsing

The React web UI consumes `/runtime` SSE streams to render:

- Planner timeline (tasks, analysis, tool invocations)
- Memory recalls and summaries per run
- Token stream output with model attribution
- Manual browser takeover prompts (`/runtime/browser/continue`) when the `browser` tool encounters a CAPTCHA or block after two failed attempts
- Configuration widgets for persona, autonomy level, and model selection (ties into router overrides)

Manual intervention flow:
1. UI receives `browser_hil_required` event.
2. Operator completes the task manually (e.g., solves CAPTCHA).
3. Submit context back via `/runtime/browser/continue` to resume execution.

## Observability & Logging

`python/runtime/observability.py` unifies metrics, JSON logging, and Helicone proxy headers.

- JSON logs: `./logs/runtime_observability.jsonl`
- Prometheus endpoint: `GET /runtime/admin/metrics` (enable admin auth if required)
- Helicone: set `HELICONE_ENABLED=true`, `HELICONE_BASE_URL=https://helicone.yourhost`, `HELICONE_API_KEY=...`

Each model call is tagged with session/run/token metadata, and counters expose run/task/tool/memory/token statistics. When Helicone is enabled the router hands back the appropriate `Helicone-*` headers for downstream logging.

## Migration & Compatibility Notes

### Importing Legacy Memory

```bash
# Convert legacy exports (memory/*.json, memory/*.ndjson)
python -m python.runtime.tools.migrate_memory ./memory --session archive-2023 --kind fact

# Rebuild vectors after import
python -m python.runtime.tools.reindex_memory --verbose
```

### Configuring Models

- **OpenAI / Azure / Anthropic:** supply API keys via `.env` (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) and add entries under `router.models` for routing policies (cost, latency, capabilities).
- **Google Gemini:** configure LiteLLM provider keys (`GEMINI_API_KEY`) and register the Gemini model in `router.models` with `provider="google"`.
- **Local LM Studio / Ollama:** expose the server as an OpenAI-compatible endpoint (e.g., http://localhost:1234/v1) and set `metadata.api_base` for that router entry. Mark `is_local=true` to allow the router to prefer remote models unless `allow_local=True` is specified.

### Environment Variables Cheat Sheet

| Purpose | Variable(s) |
| --- | --- |
| Memory location | `MEMORY_DB_PATH=./data/runtime.sqlite` |
| Mem0 hybrid | `MEM0_ENABLED=true`, `MEM0_API_KEY=...`, `MEM0_BASE_URL=...` |
| Router overrides | `ROUTER_MODELS`, `ROUTER_DEFAULT_MODEL`, `ROUTER_DEFAULT_STRATEGY` |
| Token budgets | `TOKEN_BUDGETS`, `TOKEN_BUDGET_<MODEL>`, `SUMMARIZER_MODEL` |
| UI flags | `AGENT_PERSONA`, `AGENT_MAX_LOOPS`, `AGENT_PLANNER_PROFILE` |
| Observability | `HELICONE_*`, `OBSERVABILITY_LOG_PATH`, `OBSERVABILITY_METRICS_NS` |

## Testing

A focused runtime test suite lives under `tests/runtime/`:

- `test_embeddings_cache.py` – cache hit/miss and batching behaviour
- `test_memory_recall.py` – FAISS similarity and Mem0 fallback
- `test_token_fit.py` – budget trimming and summarisation
- `test_loop_integration.py` – orchestration loop integration
- `test_tool_browser.py` – browser-use retries and human-in-loop escalation
- `test_prompt_manager.py` – overrides, personas, adaptive hints
- `test_model_router.py` – routing policies, telemetry headers

Run all runtime tests with:

```bash
PATH="$(realpath ../.pytest-venv/bin):$HOME/bin:$PATH" \
PYTHONPATH="$(pwd)" \
python -m pytest tests/runtime -q
```

## macOS Platform Notes

- **Native PTY Support**: No Docker required - terminal functionality lives under `python/adapters/terminal/macos_pty.py`
- **Web Terminal**: Full pseudo-terminal support for interactive shell sessions in the web UI
- **Playwright Browsers**: Automatically installed into `./tmp/playwright` by the setup script; override with `PLAYWRIGHT_BROWSERS_PATH` if you need a shared cache
- **FFmpeg Integration**: Voice synthesis leverages system FFmpeg installation for audio processing
- **MPS Acceleration**: Automatic detection and utilization of Apple Silicon GPU acceleration for TTS and other compute-intensive operations

For deeper customization, inspect `python/runtime/container.py` to see how singletons are bootstrapped and cached across the runtime.

---

## 🙏 Credits & Attribution

This project is the macOS evolution of the amazing **Agent Zero** framework! Apple Zero wouldn't be possible without the incredible work of the Agent Zero team and contributors.

### 🌟 The Agent Zero Team

A huge thank you to the original Agent Zero contributors:
- **[Agent Zero Core Team](https://github.com/frdel/agent_zero)** - For creating the foundation that made all of this possible!

### 🤝 Special Thanks to

- **Homebrew Community** - Making macOS package management actually bearable
- **LiteLLM Contributors** - For the unified LLM API that makes connecting to dozens of models painless
- **FastAPI Team** - Because async APIs should be this easy
- **Playwright** - Web automation that's actually reliable (mostly!)
- **Faiss Contributors** - Vector search that doesn't make your brain hurt

### 📜 License & Acknowledgments

This project inherits the wonderful MIT license from the original Cosmic Crisp / Agent Zero projects. We're proud to build upon such solid foundations!

---

<div align="center">

<h3>🤖 Ready to Unleash AI on Your Mac? 🚀</h3>

<p><strong>The future is here, and it's delightfully user-friendly!</strong></p>

[![Get Started](https://img.shields.io/badge/Get%20Started-Today-green?style=for-the-badge&logo=apple)](./dev/macos/setup.sh)

<small>"Because who needs boring, when you can have brilliant?" - Apple Zero Team</small>

</div>
