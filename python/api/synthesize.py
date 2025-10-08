# api/synthesize.py

import base64
import os
import subprocess
import tempfile
import json
import urllib.request
import urllib.error

from python.helpers.api import ApiHandler, Request, Response

from python.helpers import settings

try:
    from TTS.api import TTS as CoquiTTS
except Exception:
    CoquiTTS = None


def _tts_settings() -> dict:
    all_settings = settings.get_settings()
    tts_settings = all_settings.get("tts")
    if not isinstance(tts_settings, dict):
        tts_settings = settings.get_default_settings()["tts"]
    return tts_settings


def _chatterbox_settings(tts_settings: dict):
    from python.helpers.chatterbox_tts import config_from_dict as _cfg
    chatterbox = tts_settings.get("chatterbox")
    if not isinstance(chatterbox, dict):
        chatterbox = settings.get_default_settings()["tts"]["chatterbox"]
    return _cfg(chatterbox)


def _xtts_settings(tts_settings: dict):
    from python.helpers.xtts_tts import config_from_dict as _cfg
    xtts = tts_settings.get("xtts")
    if not isinstance(xtts, dict):
        xtts = settings.get_default_settings()["tts"]["xtts"]
    return _cfg(xtts)


def _piper_vc_settings(tts_settings: dict):
    p = tts_settings.get("piper_vc")
    if not isinstance(p, dict):
        p = {}
    return {
        "piper_bin": p.get("piper_bin", "piper"),
        "piper_model": p.get("piper_model", os.environ.get("PIPER_MODEL", "")),
        "sample_rate": int(p.get("sample_rate", 22050)),
        "chunk_chars": int(p.get("max_chars", 280)),
        "join_silence_ms": int(p.get("join_silence_ms", 80)),
        "target_voice_wav": p.get("target_voice_wav") or "",
    }


class Synthesize(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        text = input.get("text", "")
        tts_settings = _tts_settings()
        engine = input.get("engine", str(tts_settings.get("engine", "chatterbox"))).lower()
        style = input.get("style", {}) or {}
        filtered_style = {k: v for k, v in style.items() if v is not None}
        if "voice_wav_path" in filtered_style and "speaker_wav_path" not in filtered_style:
            filtered_style["speaker_wav_path"] = filtered_style["voice_wav_path"]
        try:
            if engine == "xtts":
                try:
                    from python.helpers.xtts_tts import synthesize_base64 as xtts_synthesize_base64
                except Exception as e:
                    return {"error": f"XTTS unavailable: {e}", "success": False, "engine": engine}
                audio = await xtts_synthesize_base64(text, _xtts_settings(tts_settings), style=filtered_style)
                return {"audio": audio, "success": True, "engine": engine}

            if engine == "kokoro":
                try:
                    from python.helpers import kokoro_tts
                except Exception as e:
                    return {"error": f"Kokoro unavailable: {e}", "success": False, "engine": engine}
                try:
                    kokoro_cfg = tts_settings.get("kokoro", {}) if isinstance(tts_settings, dict) else {}
                    voice = (
                        filtered_style.get("speaker")
                        or filtered_style.get("voice")
                        or kokoro_cfg.get("voice")
                        or "am_puck,am_onyx"
                    )
                    speed_raw = filtered_style.get("speed", kokoro_cfg.get("speed", 1.1))
                    try:
                        speed = float(speed_raw)
                    except Exception:
                        speed = 1.1
                    sr_raw = filtered_style.get("sample_rate", kokoro_cfg.get("sample_rate", 24_000))
                    try:
                        sample_rate = int(sr_raw)
                    except Exception:
                        sample_rate = 24_000
                    audio = await kokoro_tts.synthesize_sentences([text], voice=voice, speed=speed, sample_rate=sample_rate)
                    return {"audio": audio, "success": True, "engine": engine}
                except Exception as e:
                    return {"error": str(e), "success": False, "engine": engine}

            if engine == "piper_vc":
                try:
                    cfg = _piper_vc_settings(tts_settings)
                    piper_bin, piper_model = cfg["piper_bin"], cfg["piper_model"]
                    if not piper_model:
                        return {"error": "Piper model path is not configured (tts.piper_vc.piper_model)", "success": False, "engine": engine}
                    target_voice_wav = (
                        filtered_style.get("speaker_wav_path")
                        or filtered_style.get("voice_wav_path")
                        or cfg["target_voice_wav"]
                    )
                    if not target_voice_wav:
                        return {"error": "Target voice WAV is required (style.speaker_wav_path or tts.piper_vc.target_voice_wav)", "success": False, "engine": engine}
                    sidecar_url = os.environ.get("TTS_SIDECAR_URL", "http://127.0.0.1:7055")

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
                        in_path = tmp_in.name
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                        out_path = tmp_out.name
                    try:
                        subprocess.run(
                            [piper_bin, "--model", piper_model, "--output_file", in_path, "--sentence_silence", "0.0"],
                            input=text.encode("utf-8"),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            check=True,
                        )
                        if CoquiTTS is not None:
                            try:
                                import torch
                                device = "cpu"
                                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                                    device = "mps"
                                elif torch.cuda.is_available():
                                    device = "cuda"
                            except Exception:
                                device = "cpu"
                            vc = CoquiTTS("voice_conversion_models/multilingual/multi-dataset/openvoice_v2").to(device)
                            vc.voice_conversion_to_file(source_wav=in_path, target_wav=target_voice_wav, file_path=out_path)
                            with open(out_path, "rb") as f:
                                audio_b64 = base64.b64encode(f.read()).decode("ascii")
                            return {"audio": audio_b64, "success": True, "engine": engine}
                        else:
                            # Fallback to sidecar VC
                            payload = {
                                "source_wav_path": in_path,
                                "target_wav_path": target_voice_wav,
                                "model_id": "voice_conversion_models/multilingual/multi-dataset/openvoice_v2",
                            }
                            data = json.dumps(payload).encode("utf-8")
                            req = urllib.request.Request(
                                f"{sidecar_url.rstrip('/')}/api/vc/convert",
                                data=data,
                                headers={"Content-Type": "application/json"},
                            )
                            try:
                                with urllib.request.urlopen(req, timeout=180.0) as resp:
                                    if resp.status != 200:
                                        raise RuntimeError(f"sidecar HTTP {resp.status}")
                                    raw = resp.read()
                                body = json.loads(raw.decode("utf-8"))
                                if not isinstance(body.get("audio_b64"), str):
                                    raise RuntimeError("sidecar VC missing audio_b64")
                                return {"audio": body["audio_b64"], "success": True, "engine": engine}
                            except Exception as sc_err:
                                return {"error": f"VC sidecar failed: {sc_err}", "success": False, "engine": engine}
                    except subprocess.CalledProcessError as cpe:
                        return {"error": f"Piper failed: {cpe.stderr.decode('utf-8', 'ignore')}", "success": False, "engine": engine}
                    except Exception as exc:
                        return {"error": str(exc), "success": False, "engine": engine}
                    finally:
                        for path in (in_path, out_path):
                            try:
                                if path and os.path.exists(path):
                                    os.remove(path)
                            except Exception:
                                pass
                except Exception as e:
                    return {"error": str(e), "success": False, "engine": engine}

            if engine == "chatterbox":
                try:
                    from python.helpers.chatterbox_tts import synthesize_base64 as chatterbox_synthesize_base64
                except Exception as e:
                    return {"error": f"Chatterbox unavailable: {e}", "success": False, "engine": engine}
                audio = await chatterbox_synthesize_base64(text, _chatterbox_settings(tts_settings), style=filtered_style)
                return {"audio": audio, "success": True, "engine": engine}

            if engine == "browser":
                return {"audio": None, "success": True, "engine": engine}

            raise ValueError(f"Unsupported TTS engine: {engine}")
        except Exception as e:
            return {"error": str(e), "success": False, "engine": engine}
    
    # def _clean_text(self, text: str) -> str:
    #     """Clean text by removing markdown, tables, code blocks, and other formatting"""
    #     # Remove code blocks
    #     text = re.sub(r'```[\s\S]*?```', '', text)
    #     text = re.sub(r'`[^`]*`', '', text)
        
    #     # Remove markdown links
    #     text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
    #     # Remove markdown formatting
    #     text = re.sub(r'[*_#]+', '', text)
        
    #     # Remove tables (basic cleanup)
    #     text = re.sub(r'\|[^\n]*\|', '', text)
        
    #     # Remove extra whitespace and newlines
    #     text = re.sub(r'\n+', ' ', text)
    #     text = re.sub(r'\s+', ' ', text)
        
    #     # Remove URLs
    #     text = re.sub(r'https?://[^\s]+', '', text)
        
    #     # Remove email addresses
    #     text = re.sub(r'\S+@\S+', '', text)
        
    #     return text.strip()
    
    # def _chunk_text(self, text: str) -> list[str]:
    #     """Split text into manageable chunks for TTS"""
    #     # If text is short enough, return as single chunk
    #     if len(text) <= 300:
    #         return [text]
        
    #     # Split into sentences first
    #     sentences = re.split(r'(?<=[.!?])\s+', text)
        
    #     chunks = []
    #     current_chunk = ""
        
    #     for sentence in sentences:
    #         sentence = sentence.strip()
    #         if not sentence:
    #             continue
                
    #         # If adding this sentence would make chunk too long, start new chunk
    #         if current_chunk and len(current_chunk + " " + sentence) > 300:
    #             chunks.append(current_chunk.strip())
    #             current_chunk = sentence
    #         else:
    #             current_chunk += (" " if current_chunk else "") + sentence
        
    #     # Add the last chunk if it has content
    #     if current_chunk.strip():
    #         chunks.append(current_chunk.strip())
        
    #     return chunks if chunks else [text]
