"""Lightweight fallback implementation for the NeuCodec ONNX decoder.

This avoids pulling in the full `neucodec` Python package (which depends on a
bleeding-edge PyTorch stack) while still allowing us to stream inference using
the officially published ONNX graph.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

try:  # pragma: no cover - heavy optional dependency
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "onnxruntime is required for NeuTTSAir streaming. Install it with `pip install onnxruntime`."
    ) from exc

try:
    from huggingface_hub import hf_hub_download
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "huggingface-hub is required to download NeuCodec assets. Install it with `pip install huggingface-hub`."
    ) from exc


class _UnsupportedCodec:
    def __init__(self, *_, **__):
        raise RuntimeError(
            "The full `neucodec` package is not available. Only the ONNX decoder "
            "is supported in this runtime."
        )


NeuCodec = _UnsupportedCodec  # type: ignore
DistillNeuCodec = _UnsupportedCodec  # type: ignore


@dataclass
class NeuCodecOnnxDecoder:
    """Minimal ONNX runtime wrapper that mirrors the behaviour expected by NeuTTSAir."""

    session: ort.InferenceSession
    sample_rate: int = 24_000

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        proxies: Optional[Dict[str, str]] = None,
        resume_download: bool = False,
        local_files_only: bool = False,
        token: Optional[str] = None,
        **_: Any,
    ) -> "NeuCodecOnnxDecoder":
        """Download the decoder graph from Hugging Face Hub and initialise an ONNX session."""
        onnx_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.onnx",
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            token=token,
        )
        # Touch meta.yaml to keep parity with upstream downloader (best-effort).
        try:  # pragma: no cover - optional bookkeeping
            hf_hub_download(
                repo_id=repo_id,
                filename="meta.yaml",
                revision=revision,
                cache_dir=cache_dir,
                force_download=False,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
            )
        except Exception:
            pass

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(onnx_path, sess_options=so)
        return cls(session=session)

    def encode_code(self, *_: Any, **__: Any) -> np.ndarray:  # pragma: no cover - API parity
        raise NotImplementedError("The ONNX decoder only supports decode_code().")

    def decode_code(self, codes: np.ndarray) -> np.ndarray:
        if not isinstance(codes, np.ndarray):
            raise ValueError("`codes` must be a numpy array of shape [B, 1, F].")
        if codes.ndim != 3 or codes.shape[1] != 1:
            raise ValueError(f"Unexpected code shape {codes.shape}; expected [B, 1, F].")
        recon = self.session.run(None, {"codes": codes})[0]
        return recon.astype(np.float32)
