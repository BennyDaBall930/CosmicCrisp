import pytest

from models import ChatGenerationResult, ChatChunk, _parse_chunk


def test_parse_chunk_detects_reasoning_fields():
    chunk = {
        "choices": [
            {
                "delta": {
                    "content": [{"type": "text", "text": "partial"}],
                    "reasoning": [{"text": "alpha"}, {"text": "beta"}],
                    "x_gpt_thinking": "gamma",
                },
                "message": {},
                "model_extra": {
                    "message": {},
                    "internal_thoughts": "delta",
                },
            }
        ]
    }

    parsed = _parse_chunk(chunk)

    assert parsed["response_delta"] == "partial"
    assert parsed["reasoning_delta"]
    compact = parsed["reasoning_delta"].replace(" ", "")
    assert compact == "alphabetagammadelta"


def test_chat_generation_result_native_reasoning_delta_diff():
    result = ChatGenerationResult()

    first = result.add_chunk(ChatChunk(response_delta="", reasoning_delta="Plan"))
    assert first["reasoning_delta"] == "Plan"
    assert result.reasoning == "Plan"

    second = result.add_chunk(
        ChatChunk(response_delta="", reasoning_delta="Plan step")
    )
    assert second["reasoning_delta"] == " step"
    assert result.reasoning == "Plan step"


def test_chat_generation_result_tag_reasoning_split_across_chunks():
    result = ChatGenerationResult()

    result.add_chunk(ChatChunk(response_delta="<think>Do", reasoning_delta=""))
    result.add_chunk(ChatChunk(response_delta=" this</th", reasoning_delta=""))
    result.add_chunk(ChatChunk(response_delta="ink>Answer", reasoning_delta=""))

    assert result.reasoning == "Do this"
    assert result.response == "Answer"


def test_chat_generation_result_no_reasoning():
    result = ChatGenerationResult()

    result.add_chunk(ChatChunk(response_delta="Plain text", reasoning_delta=""))

    assert result.reasoning == ""
    assert result.response == "Plain text"
