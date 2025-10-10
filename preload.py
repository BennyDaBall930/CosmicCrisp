import asyncio
import subprocess
import os
import logging
from python.helpers import runtime, whisper, settings
from python.helpers.xtts_tts import config_from_dict as xtts_config_from_dict, get_backend as get_xtts_backend
from python.helpers.print_style import PrintStyle
import models


async def preload():
    try:
        current = settings.get_settings()

        # preload embedding model
        async def preload_embedding():
            try:
                if current["embed_model_provider"].lower() == "huggingface":
                    # Use the new LiteLLM-based model system
                    emb_mod = models.get_embedding_model(
                        "huggingface", current["embed_model_name"]
                    )
                    emb_txt = await emb_mod.aembed_query("test")
                    return emb_txt
            except Exception as e:
                PrintStyle().error(f"Error in preload_embedding: {e}")

        async def preload_xtts():
            try:
                tts_settings = current.get("tts", {}) if isinstance(current, dict) else {}
                if not isinstance(tts_settings, dict):
                    return
                if tts_settings.get("engine") != "xtts":
                    return
                cfg_map = tts_settings.get("xtts")
                cfg = xtts_config_from_dict(cfg_map if isinstance(cfg_map, dict) else {})
                await asyncio.to_thread(get_xtts_backend, cfg)
                PrintStyle(level=logging.DEBUG).print("XTTS backend warmed up")
            except Exception as e:
                PrintStyle().error(f"Error in preload_xtts: {e}")

        async def preload_kokoro():
            try:
                tts_settings = current.get("tts", {}) if isinstance(current, dict) else {}
                if not isinstance(tts_settings, dict):
                    return
                if tts_settings.get("engine") != "kokoro":
                    return
                from python.helpers import kokoro_tts
                await kokoro_tts.preload()
                PrintStyle(level=logging.DEBUG).print("Kokoro backend warmed up")
            except Exception as e:
                PrintStyle().error(f"Error in preload_kokoro: {e}")

        # async tasks to preload
        tasks = [
            preload_embedding(),
            preload_xtts(),
            preload_kokoro(),
        ]

        await asyncio.gather(*tasks, return_exceptions=True)
        PrintStyle().print("Preload completed")
    except Exception as e:
        PrintStyle().error(f"Error in preload: {e}")


# preload transcription model
if __name__ == "__main__":
    PrintStyle().print("Running preload...")
    runtime.initialize()
    asyncio.run(preload())
