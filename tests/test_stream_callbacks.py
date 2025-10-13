import json

import pytest

from agent import LoopData, Agent


class DummyAgent:
    def __init__(self):
        self.loop_data = LoopData()
        self.events: list[str] = []
        self.last_kwargs: dict[str, dict] = {}

    async def call_extensions(self, extension_point: str, **kwargs):
        self.events.append(extension_point)
        self.last_kwargs[extension_point] = kwargs
        stream_data = kwargs.get("stream_data")
        if extension_point == "reasoning_stream_chunk" and stream_data:
            stream_data["chunk"] = "[masked]"
            stream_data["full"] = "[masked-full]"
        if extension_point == "response_stream_chunk" and stream_data:
            stream_data["chunk"] = stream_data["chunk"].replace("bad", "good")
            stream_data["full"] = stream_data["full"].replace("bad", "good")

    async def handle_intervention(self, progress: str = ""):
        return


@pytest.mark.asyncio
async def test_reasoning_stream_callbacks_order_and_masking():
    agent = DummyAgent()

    stream_data = {"chunk": "raw", "full": "accumulated"}

    await agent.call_extensions(
        "reasoning_stream_chunk", loop_data=agent.loop_data, stream_data=stream_data
    )
    assert stream_data["chunk"] == "[masked]"

    await Agent.handle_reasoning_stream(agent, stream_data["full"])
    await agent.call_extensions("reasoning_stream_end", loop_data=agent.loop_data)

    assert agent.events == [
        "reasoning_stream_chunk",
        "reasoning_stream",
        "reasoning_stream_end",
    ]


@pytest.mark.asyncio
async def test_response_stream_callbacks_order_and_redaction():
    agent = DummyAgent()

    payload = json.dumps(
        {
            "tool_name": "response",
            "tool_args": {"text": "Hello from callback test with adequate length."},
        }
    )

    stream_data = {"chunk": "bad chunk", "full": payload}

    await agent.call_extensions(
        "response_stream_chunk", loop_data=agent.loop_data, stream_data=stream_data
    )
    assert stream_data["chunk"] == "good chunk"

    await Agent.handle_response_stream(agent, stream_data["full"])
    await agent.call_extensions("response_stream_end", loop_data=agent.loop_data)

    assert agent.events == [
        "response_stream_chunk",
        "response_stream",
        "response_stream_end",
    ]
