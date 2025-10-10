from models import ChatGenerationResult, ChatChunk


def test_thinking_tags_are_parsed_into_reasoning():
    result = ChatGenerationResult()

    first = result.add_chunk(ChatChunk(response_delta="<think>reason", reasoning_delta=""))
    assert first["response_delta"] == ""
    assert first["reasoning_delta"] == "reason"

    second = result.add_chunk(
        ChatChunk(response_delta="ing</think>answer", reasoning_delta="")
    )
    assert second["reasoning_delta"] == "ing"
    assert second["response_delta"] == "answer"

    final = result.output()
    assert final["reasoning_delta"] == "reasoning"
    assert final["response_delta"] == "answer"


def test_native_reasoning_passthrough():
    result = ChatGenerationResult()
    chunk = result.add_chunk(
        ChatChunk(response_delta="answer", reasoning_delta="thought process")
    )
    assert result.native_reasoning is True
    assert chunk["reasoning_delta"] == "thought process"
    assert chunk["response_delta"] == "answer"


def test_unclosed_thinking_tag_flushes_to_reasoning():
    result = ChatGenerationResult()
    result.add_chunk(ChatChunk(response_delta="<think>partial", reasoning_delta=""))
    output = result.output()
    assert output["reasoning_delta"] == "partial"
