"""FastAPI application exposing the Apple Zero runtime."""
from __future__ import annotations

import base64
import os
from typing import Any, AsyncIterator, Dict

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .schemas import (
    BrowserContinueRequest,
    CancelRequest,
    ChatRequest,
    MemoryQuery,
    ResumeRequest,
    RunRequest,
    ToolInvokeRequest,
)
from .sse import sse_iter
from ..config import load_runtime_config
from ..container import event_bus, get_orchestrator, memory, tool_registry

TAGS_METADATA = [
    {"name": "Chat", "description": "Interactive chat endpoints with token streaming."},
    {"name": "Run", "description": "Autonomous execution endpoints."},
    {"name": "Tools", "description": "Tool discovery and invocation."},
    {"name": "Events", "description": "Server-sent events and notifications."},
    {"name": "Admin", "description": "Administrative APIs."},
]

app = FastAPI(title="Apple Zero Runtime API", version="0.1.0", openapi_tags=TAGS_METADATA)

_config = load_runtime_config()
_origins = [origin.strip() for origin in os.getenv("RUNTIME_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_admin_auth(request: Request) -> None:
    mode = os.getenv("RUNTIME_AUTH_MODE", "none").lower()
    if mode == "none":
        return
    if mode == "basic":
        header = request.headers.get("Authorization")
        if not header or not header.startswith("Basic "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        encoded = header.split(" ", 1)[1]
        try:
            decoded = base64.b64decode(encoded).decode()
        except Exception as exc:  # pragma: no cover
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED) from exc
        username, _, password = decoded.partition(":")
        expected_user = os.getenv("RUNTIME_AUTH_USER", "")
        expected_pass = os.getenv("RUNTIME_AUTH_PASS", "")
        if username != expected_user or password != expected_pass:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        return
    if mode == "bearer":
        header = request.headers.get("Authorization", "")
        token = header.split("Bearer ")[-1].strip()
        expected = os.getenv("RUNTIME_AUTH_TOKEN")
        if not expected or token != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)


def _registry_snapshot() -> Dict[str, Any]:
    registry = tool_registry()
    return {
        name: {
            "enabled": entry.enabled,
            "description": entry.description,
            "tags": entry.tags,
        }
        for name, entry in registry.items()
    }


router = APIRouter(prefix="/runtime")


@router.post("/chat", tags=["Chat"])
async def chat_endpoint(payload: ChatRequest) -> StreamingResponse:
    orchestrator = get_orchestrator()

    async def source() -> AsyncIterator[Dict[str, Any]]:
        async for event in orchestrator.stream_chat(payload):
            yield event

    headers = {"Cache-Control": "no-store", "Connection": "keep-alive"}
    return StreamingResponse(sse_iter(source()), media_type="text/event-stream", headers=headers)


@router.post("/run", tags=["Run"])
async def run_endpoint(payload: RunRequest) -> StreamingResponse:
    orchestrator = get_orchestrator()

    async def source() -> AsyncIterator[Dict[str, Any]]:
        async for event in orchestrator.stream_run(payload):
            yield event

    headers = {"Cache-Control": "no-store", "Connection": "keep-alive"}
    return StreamingResponse(sse_iter(source()), media_type="text/event-stream", headers=headers)


@router.post("/run/cancel", tags=["Run"])
async def cancel_endpoint(payload: CancelRequest) -> Dict[str, Any]:
    orchestrator = get_orchestrator()
    cancelled = await orchestrator.cancel_run(payload.run_id)
    if not cancelled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    return {"ok": True, "run_id": payload.run_id}


@router.post("/run/resume", tags=["Run"])
async def resume_endpoint(payload: ResumeRequest) -> Dict[str, Any]:
    orchestrator = get_orchestrator()
    resumed = await orchestrator.resume_run(payload.run_id, model_override=payload.model_override)
    if not resumed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    return {"ok": True, "run_id": payload.run_id}


@router.post("/browser/continue", tags=["Run"])
async def browser_continue_endpoint(payload: BrowserContinueRequest) -> Dict[str, Any]:
    orchestrator = get_orchestrator()
    acknowledged = await orchestrator.browser_continue(payload.session_id, payload.context_id, payload.payload)
    if not acknowledged:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="context not found")
    return {"ok": True}


@router.get("/events", tags=["Events"])
async def events_endpoint() -> StreamingResponse:
    bus = event_bus()

    async def source() -> AsyncIterator[Dict[str, Any]]:
        async for event in bus.subscribe():
            yield {"event": "event", "data": event}

    headers = {"Cache-Control": "no-store", "Connection": "keep-alive"}
    return StreamingResponse(sse_iter(source()), media_type="text/event-stream", headers=headers)


@router.get("/tools", tags=["Tools"])
async def tools_endpoint() -> Dict[str, Any]:
    return {"tools": _registry_snapshot()}


@router.post("/tools/invoke", tags=["Tools"])
async def tools_invoke_endpoint(payload: ToolInvokeRequest) -> Dict[str, Any]:
    registry = tool_registry()
    tool = registry.get(payload.name)
    if tool is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="tool not found")
    result = await tool.run(**payload.inputs)
    return {"result": result}


@router.get("/admin/memory", tags=["Admin"], dependencies=[Depends(require_admin_auth)])
async def admin_memory_endpoint(query: MemoryQuery) -> Dict[str, Any]:
    store = memory()
    items = []
    if query.query:
        items = await store.similar(query.query, k=query.top_k)
    if query.recent:
        recent_items = await store.recent(query.recent)
        items.extend(recent_items)
    return {"items": items}


@router.post("/admin/memory/reindex", tags=["Admin"], dependencies=[Depends(require_admin_auth)])
async def admin_memory_reindex_endpoint() -> Dict[str, Any]:
    store = memory()
    reindex = getattr(store, "reindex", None)
    if reindex is None:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED)
    await reindex()  # type: ignore[operator]
    return {"ok": True}


@router.delete("/admin/memory/{memory_id}", tags=["Admin"], dependencies=[Depends(require_admin_auth)])
async def admin_memory_delete_endpoint(memory_id: str) -> Dict[str, Any]:
    store = memory()
    delete = getattr(store, "delete", None)
    if delete is None:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED)
    await delete(memory_id)  # type: ignore[operator]
    return {"ok": True}


@router.get("/admin/health", tags=["Admin"], dependencies=[Depends(require_admin_auth)])
async def admin_health_endpoint() -> Dict[str, Any]:
    return {
        "config": {
            "embeddings_provider": _config.embeddings.provider,
            "memory_db": str(_config.memory.db_path),
            "token_models": list(_config.tokens.budgets.keys()),
            "tools_auto_modules": list(_config.tools.auto_modules),
        }
    }


@router.get("/admin/runs/{run_id}/artifacts", tags=["Admin"], dependencies=[Depends(require_admin_auth)])
async def admin_list_artifacts(run_id: str) -> Dict[str, Any]:
    return {"run_id": run_id, "artifacts": []}


@router.get("/admin/artifacts/{artifact_id}", tags=["Admin"], dependencies=[Depends(require_admin_auth)])
async def admin_download_artifact(artifact_id: str):
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="artifact not available")


app.include_router(router)
