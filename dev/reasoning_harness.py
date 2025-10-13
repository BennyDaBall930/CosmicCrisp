#!/usr/bin/env python3
"""Local harness for exercising reasoning/response chunk parsing without provider calls."""

from __future__ import annotations

from models import ChatGenerationResult, _parse_chunk


def _native_reasoning_samples():
    return [
        {
            "choices": [
                {
                    "delta": {
                        "content": {"type": "text", "text": ""},
                        "reasoning_content": "Scout task",
                    },
                    "message": {},
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "content": {"type": "text", "text": ""},
                        "reasoning_content": " -> draft reply",
                    },
                    "message": {},
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"content": {"type": "text", "text": ""}},
                    "message": {
                        "content": "{\"tool_name\":\"response\",\"tool_args\":{\"text\":\"Final answer.\"}}"
                    },
                }
            ]
        },
    ]


def _tag_reasoning_samples():
    return [
        {
            "choices": [
                {
                    "delta": {"content": "<think>Plan"},
                    "message": {},
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"content": " the work</th"},
                    "message": {},
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {"content": "ink>Answer"},
                    "message": {},
                }
            ]
        },
    ]


def _run_labelled(label: str, raw_chunks: list[dict]):
    print(f"\n=== {label} ===")
    result = ChatGenerationResult()
    for raw in raw_chunks:
        parsed = _parse_chunk(raw)
        processed = result.add_chunk(parsed)
        if processed["reasoning_delta"]:
            print(f"Reasoning Δ: {processed['reasoning_delta']}")
        if processed["response_delta"]:
            print(f"Response Δ: {processed['response_delta']}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Response: {result.response}")


def main():
    _run_labelled("Native reasoning", _native_reasoning_samples())
    _run_labelled("Tag reasoning", _tag_reasoning_samples())


if __name__ == "__main__":
    main()
