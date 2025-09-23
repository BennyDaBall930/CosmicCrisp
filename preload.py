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
                # Launch MLX server as subprocess
                settings_path = os.path.join(runtime.get_application_root(), "tmp", "settings.json")
                if not os.path.exists(settings_path):
                    PrintStyle().error("MLX settings file not found")
                    return
                proc = subprocess.Popen([
                    "python", "-m", "python.models.apple_mlx_provider", settings_path
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=runtime.get_application_root())
                PrintStyle().print(f"MLX server launched (PID: {proc.pid})")
                # Store PID for tracking
                with open(os.path.join(runtime.get_application_root(), "logs", "mlx_pid.txt"), "w") as f:
                    f.write(str(proc.pid))
                # Don't wait, let it run
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
