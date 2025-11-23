from __future__ import annotations

import re

import numpy as np
import torch

from python.third_party.neuttsair.neutts import NeuTTSAir


class _NoopWatermarker:
    def apply_watermark(self, audio, sample_rate: int = 24_000):
        return audio


def test_infer_stream_ggml_handles_partial_tokens() -> None:
    engine = NeuTTSAir.__new__(NeuTTSAir)
    engine.sample_rate = 24_000
    engine.streaming_frames_per_chunk = 3
    engine.streaming_lookforward = 1
    engine.streaming_lookback = 2
    engine.streaming_overlap_frames = 1
    engine.hop_length = 2
    engine.streaming_stride_samples = engine.streaming_frames_per_chunk * engine.hop_length
    engine.max_context = 512
    engine.watermarker = _NoopWatermarker()

    decode_calls: list[list[int]] = []

    def _fake_decode(self, codes: str) -> np.ndarray:
        token_ids = [int(m) for m in re.findall(r"<\|speech_(\d+)\|>", codes)]
        decode_calls.append(token_ids)

        values: list[float] = []
        for token in token_ids:
            values.extend([float(token)] * self.hop_length)
        return np.array(values, dtype=np.float32)

    engine._decode = _fake_decode.__get__(engine, NeuTTSAir)

    class _DummyBackbone:
        def __call__(self, *_args, **kwargs):
            assert kwargs.get("stream") is True
            outputs = [
                {"choices": [{"text": "<|speech_200|><|speech_201|><|speech_202|><|speech_203|>"}]},
                {"choices": [{"text": "<|speech_204|><|speech_205|><|speech_206"}]},
                {"choices": [{"text": "|><|speech_207|>"}]},
                {"choices": [{"text": "<|speech_208|><|speech_209|><|speech_210|>"}]},
            ]
            for item in outputs:
                yield item

    engine.backbone = _DummyBackbone()

    ref_codes = torch.tensor([10, 20], dtype=torch.long)
    chunks = list(engine._infer_stream_ggml(ref_codes, "reference text", "input to speak"))

    assert len(chunks) >= 2
    first_chunk = chunks[0]
    assert isinstance(first_chunk, np.ndarray)
    assert first_chunk.size > 0
    # Voice sample should be gone – we should immediately start with generated speech tokens.
    assert np.isclose(first_chunk[0], 200.0)

    generated_tokens = []
    for call in decode_calls:
        generated_tokens.extend(token for token in call if token >= 200)

    unique_tokens = []
    for token in generated_tokens:
        if not unique_tokens or token != unique_tokens[-1]:
            unique_tokens.append(token)

    assert unique_tokens[:6] == [200, 201, 202, 203, 204, 205]
    assert unique_tokens[-1] == 210
