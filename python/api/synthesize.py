# api/synthesize.py

from python.helpers.api import ApiHandler, Request, Response

from python.helpers import settings
from python.helpers.chatterbox_tts import (
    ChatterboxConfig,
    config_from_dict as chatterbox_config_from_dict,
    synthesize_base64 as chatterbox_synthesize_base64,
)
from python.helpers.xtts_tts import (
    XTTSConfig,
    config_from_dict as xtts_config_from_dict,
    synthesize_base64 as xtts_synthesize_base64,
)


def _tts_settings() -> dict:
    all_settings = settings.get_settings()
    tts_settings = all_settings.get("tts")
    if not isinstance(tts_settings, dict):
        tts_settings = settings.get_default_settings()["tts"]
    return tts_settings


def _chatterbox_settings(tts_settings: dict) -> ChatterboxConfig:
    chatterbox = tts_settings.get("chatterbox")
    if not isinstance(chatterbox, dict):
        chatterbox = settings.get_default_settings()["tts"]["chatterbox"]
    return chatterbox_config_from_dict(chatterbox)


def _xtts_settings(tts_settings: dict) -> XTTSConfig:
    xtts = tts_settings.get("xtts")
    if not isinstance(xtts, dict):
        xtts = settings.get_default_settings()["tts"]["xtts"]
    return xtts_config_from_dict(xtts)


class Synthesize(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        text = input.get("text", "")
        tts_settings = _tts_settings()
        engine = str(tts_settings.get("engine", "chatterbox")).lower()
        style = input.get("style", {}) or {}
        filtered_style = {k: v for k, v in style.items() if v is not None}
        try:
            if engine == "xtts":
                audio = await xtts_synthesize_base64(
                    text,
                    _xtts_settings(tts_settings),
                    style=filtered_style,
                )
                return {"audio": audio, "success": True, "engine": engine}

            if engine == "chatterbox":
                audio = await chatterbox_synthesize_base64(
                    text,
                    _chatterbox_settings(tts_settings),
                    style=filtered_style,
                )
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
