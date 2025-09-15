"""FastAPI application exposing CosmicCrisp endpoints."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from ..agent.service import AgentService
from ..memory.sqlite_faiss import SQLiteFAISSMemory
from ..streaming.stream import stream_generator

app = FastAPI()

memory = SQLiteFAISSMemory()
agent = AgentService(memory=memory)


@app.post("/run")
async def run_endpoint(request: Request) -> StreamingResponse:
    data = await request.json()
    goal: str = data.get("goal", "")
    gen = agent.run(goal)
    return StreamingResponse(stream_generator(gen), media_type="text/plain")


@app.post("/chat")
async def chat_endpoint(request: Request) -> StreamingResponse:
    data = await request.json()
    session = data.get("session_id", "default")
    message = data.get("message", "")
    gen = agent.chat(session, message)
    return StreamingResponse(stream_generator(gen), media_type="text/plain")
