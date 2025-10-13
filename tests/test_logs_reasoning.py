import pytest

from agent import LoopData
from python.helpers.log import Log
from python.extensions.message_loop_end._20_finish_generating_log import FinishGeneratingLog
from python.extensions.message_loop_start._20_reasoning_log import InitializeReasoningLog
from python.extensions.reasoning_stream._10_log_from_stream import LogFromStream
from python.extensions.response_stream._10_log_from_stream import LogFromStream as ResponseStreamLog


class StubAgent:
    def __init__(self):
        self.agent_name = "A0"
        self.context = type("ctx", (), {"log": Log()})()


@pytest.mark.asyncio
async def test_reasoning_log_lifecycle_completes_and_finishes():
    agent = StubAgent()
    loop_data = LoopData()

    await InitializeReasoningLog(agent=agent).execute(loop_data=loop_data)
    # Lazy init: card not created here
    assert "log_item_reasoning" not in loop_data.params_temporary

    await LogFromStream(agent=agent).execute(loop_data=loop_data, text="draft step")
    log_item = loop_data.params_temporary["log_item_reasoning"]
    assert log_item.temp is False
    assert log_item.content == "draft step"

    loop_data.params_temporary["log_item_generating"] = agent.context.log.log(
        type="agent", heading="Generating...", content=""
    )

    await FinishGeneratingLog(agent=agent).execute(loop_data=loop_data)

    reasoning_log = loop_data.params_temporary["log_item_reasoning"]
    assert reasoning_log.content == "draft step"
    assert reasoning_log.temp is False

    generating_log = loop_data.params_temporary["log_item_generating"]
    # no explicit finished kvp for agent logs


@pytest.mark.asyncio
async def test_reasoning_log_safety_net_creates_fallback():
    agent = StubAgent()
    loop_data = LoopData()
    loop_data.params_temporary["reasoning_text"] = "final chain"

    await FinishGeneratingLog(agent=agent).execute(loop_data=loop_data)

    reasoning_log = loop_data.params_temporary["log_item_reasoning"]
    assert reasoning_log.content == "final chain"
    assert reasoning_log.temp is False


@pytest.mark.asyncio
async def test_response_stream_bridge_populates_content():
    agent = StubAgent()
    loop_data = LoopData()

    payload = {
        "thoughts": [
            "alpha reasoning",
            "beta planning",
        ],
        "headline": "Responding",
        "tool_name": "response",
        "tool_args": {"text": "Hello"},
    }

    await ResponseStreamLog(agent=agent).execute(
        loop_data=loop_data,
        text="",
        parsed=payload,
    )

    gen_log = loop_data.params_temporary["log_item_generating"]
    assert "alpha reasoning" in gen_log.content
    assert "beta planning" in gen_log.content

    # reasoning text mirrored
    assert loop_data.params_temporary["reasoning_text"] == "alpha reasoning\nbeta planning"
