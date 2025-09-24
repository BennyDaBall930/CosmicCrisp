import asyncio
import subprocess
import os
from python.helpers import runtime, whisper, settings
from python.helpers.chatterbox_tts import config_from_dict, get_backend
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
                cfg_map = tts_settings.get("chatterbox")
                cfg = config_from_dict(cfg_map if isinstance(cfg_map, dict) else {})
                await asyncio.to_thread(get_backend, cfg)
                PrintStyle().print("Chatterbox backend warmed up")
            except Exception as e:
                PrintStyle().error(f"Error in preload_chatterbox: {e}")

        async def preload_mlx():
            try:
                mlx_settings = current.get("apple_mlx", {})
                if not mlx_settings.get("enabled", False):
                    return
                
                PrintStyle().print("Preloading Apple MLX model...")
                # Get the provider instance, which will cache it
                provider = models.get_chat_model("apple_mlx", "")
                # Aload the model to warm it up
                if provider and hasattr(provider, 'aload'):
                    await provider.aload()
                    PrintStyle().print("Apple MLX model preloaded successfully.")
                else:
                    PrintStyle().error("Could not get a valid MLX provider to preload.")

            except Exception as e:
                PrintStyle().error(f"Error in preload_mlx: {e}")

        # async tasks to preload
        tasks = [
            preload_embedding(),
            # preload_whisper(),
            preload_chatterbox(),
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
