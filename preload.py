import asyncio
import subprocess
import os
import logging
from python.helpers import runtime, whisper, settings
from python.helpers.print_style import PrintStyle
import models


async def preload():
    try:
        current = settings.get_settings()

        # preload whisper model
        async def preload_whisper():
            try:
                return await whisper.preload(current["stt_model_size"])
            except Exception as e:
                PrintStyle().error(f"Error in preload_whisper: {e}")

        # preload embedding model
        async def preload_embedding():
            if current["embed_model_provider"].lower() == "huggingface":
                try:
                    # Use the new LiteLLM-based model system
                    emb_mod = models.get_embedding_model(
                        "huggingface", current["embed_model_name"]
                    )
                    emb_txt = await emb_mod.aembed_query("test")
                    return emb_txt
                except Exception as e:
                    PrintStyle().error(f"Error in preload_embedding: {e}")

        async def preload_chatterbox():
            try:
                tts_settings = current.get("tts", {}) if isinstance(current, dict) else {}
                if not isinstance(tts_settings, dict):
                    return
                if tts_settings.get("engine") != "chatterbox":
                    return
                from python.helpers.chatterbox_tts import config_from_dict as _cfg, get_backend as _get
                cfg_map = tts_settings.get("chatterbox")
                cfg = _cfg(cfg_map if isinstance(cfg_map, dict) else {})
                await asyncio.to_thread(_get, cfg)
                PrintStyle(level=logging.DEBUG).print("Chatterbox backend warmed up")
            except Exception as e:
                PrintStyle().error(f"Error in preload_chatterbox: {e}")

        async def preload_xtts():
            try:
                tts_settings = current.get("tts", {}) if isinstance(current, dict) else {}
                if not isinstance(tts_settings, dict):
                    return
                if tts_settings.get("engine") != "xtts":
                    return
                from python.helpers.xtts_tts import config_from_dict as _cfg, get_backend as _get
                cfg_map = tts_settings.get("xtts")
                cfg = _cfg(cfg_map if isinstance(cfg_map, dict) else {})
                await asyncio.to_thread(_get, cfg)
                PrintStyle(level=logging.DEBUG).print("XTTS backend warmed up")
            except Exception as e:
                PrintStyle().error(f"Error in preload_xtts: {e}")

        async def preload_mlx():
            try:
                # Check if MLX server is enabled in settings
                if not current.get("mlx_server_enabled", False):
                    return

                PrintStyle(level=logging.DEBUG).print("Checking MLX server status...")

                # Import MLXServerManager here to avoid circular imports
                from python.helpers.mlx_server import MLXServerManager

                # Get server manager instance and check status
                manager = MLXServerManager.get_instance()
                status = manager.get_status()

                if status["status"] == "running":
                    PrintStyle(level=logging.DEBUG).print("MLX server is running and healthy.")
                else:
                    PrintStyle(level=logging.DEBUG).print(
                        f"MLX server is not ready yet (status: {status['status']})."
                    )

            except Exception as e:
                PrintStyle().error(f"Error checking MLX server status: {e}")
                import traceback
                PrintStyle().error(f"Traceback: {traceback.format_exc()}")

        # async tasks to preload
        tasks = [
            preload_embedding(),
            # preload_whisper(),
            preload_chatterbox(),
            preload_xtts(),
            preload_mlx(),
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
