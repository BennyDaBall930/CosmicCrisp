# CosmicCrisp Plan

## Overview
Port core patterns of AgentGPT into a minimal, local-first backend named **CosmicCrisp**.  The service exposes streaming FastAPI endpoints, an extensible memory layer, token budgeting, a tool registry, and an asynchronous agent loop.  Tests cover token limits, tool parsing, memory fallback, streaming, offline search behaviour, and prompt snapshots.

## File Tree
```
cosmiccrisp/
  PLAN.md
  __init__.py
  api/
    __init__.py
    app.py               # FastAPI app with /run and /chat endpoints
  agent/
    __init__.py
    service.py           # agent loop: start→analyze→execute→create→summarize→chat
    prompts.py           # static prompt strings
    task_parser.py       # typed function-call parsing via Pydantic
    helpers.py           # utility functions
  memory/
    __init__.py
    interface.py         # AgentMemory protocol
    null.py              # NullMemory implementation
    sqlite_faiss.py      # SQLite-backed FAISS-like memory (simplified)
    fallback.py          # MemoryWithFallback wrapper
  tools/
    __init__.py
    base.py              # Tool base class
    registry.py          # registry for tools
    search.py            # search tool with graceful fallback
    code.py              # code generation tool using LM Studio/OpenAI
    image.py             # image generation placeholder
    browser.py           # browser-use wrapper (read-only)
  streaming/
    __init__.py
    stream.py            # helpers for newline-delimited async streaming
  tokenizer/
    __init__.py
    token_service.py     # TokenService enforcing token budgets

tests/
  cosmiccrisp/
    __init__.py
    test_token_service.py       # token budget edge cases
    test_task_parser.py         # invalid tool schema rejected
    test_memory_fallback.py     # fallback memory when primary fails
    test_streaming.py           # end-to-end streaming happy path
    test_offline_search.py      # search failure falls back to reasoning
    golden_prompts.py           # frozen prompt strings
    test_prompts_snapshot.py    # golden prompt snapshot test
```

## Interfaces
- **AgentMemory**
  - `async enter(session_id: str)`
  - `async add(item: dict) -> str`
  - `async similar(query: str, k: int = 5) -> list[dict]`
  - `async reset()`
- **TokenService**
  - `count(text: str) -> int`
  - `trim(prompt: str, requested: int) -> int` – adjust `max_tokens` to fit within total budget.
- **Tool system**
  - `Tool` base with `name: str` and async `run(**kwargs)`.
  - `ToolCall` Pydantic model validating `tool` and `args`.
  - `ToolRegistry` for registration and lookup.
- **Agent Loop**
  - Async coroutines for each stage emitting text chunks via a generator.

## Test List
1. **TokenService budget edges** – prompts near budget are auto-trimmed.
2. **Parser rejects invalid tool** – Pydantic validation error on unknown tool.
3. **Memory fallback** – primary memory raising exception triggers secondary.
4. **Streaming happy path** – `/run` streams incremental chunks.
5. **Offline search fallback** – search tool error returns reasoning-only message.
6. **Golden prompt snapshots** – prompts match frozen reference.
```
