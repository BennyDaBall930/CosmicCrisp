import asyncio
import json
import logging
import shutil
import textwrap
import time
import urllib.request

from python.helpers import runtime, settings, whisper
from python.helpers.print_style import PrintStyle

import models
from python.runtime.audio.neutts_provider import VoiceMetadata


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

        async def preload_neutts():
            try:
                from python.runtime.container import get_tts_provider

                provider = get_tts_provider()

                current_codec = (current.get("tts", {}).get("neutts", {}) or {}).get("codec_repo")
                if current_codec == "neuphonic/neucodec":
                    try:
                        settings.set_settings_delta(
                            {"tts": {"neutts": {"codec_repo": "neuphonic/neucodec-onnx-decoder"}}},
                            apply=True,
                        )
                        provider.codec_repo = "neuphonic/neucodec-onnx-decoder"
                        log = PrintStyle()
                        log.print("Migrated NeuTTS codec_repo to neuphonic/neucodec-onnx-decoder for streaming support.")
                    except Exception as exc:  # pragma: no cover
                        PrintStyle().error(f"Failed to migrate NeuTTS codec_repo: {exc}")
                # Clean out orphaned voice directories (leftover partial registrations)
                for candidate in provider.voices_dir.glob("*"):
                    if candidate.is_dir() and not (candidate / "meta.json").exists():
                        shutil.rmtree(candidate, ignore_errors=True)

                existing = provider.list_voices()

                # Seed reference voices from the NeuTTS-Air samples repo if none exist.
                # https://github.com/neuphonic/neutts-air/tree/main/samples
                if not existing:
                    PrintStyle().print("Seeding NeuTTS sample voices (Jo, Dave)...")
                    base_url = "https://raw.githubusercontent.com/neuphonic/neutts-air/main/samples"
                    samples = (
                        {
                            "id": "jo",
                            "name": "Jo",
                            "codes": f"{base_url}/jo.pt",
                            "txt": f"{base_url}/jo.txt",
                        },
                        {
                            "id": "dave",
                            "name": "Dave",
                            "codes": f"{base_url}/dave.pt",
                            "txt": f"{base_url}/dave.txt",
                        },
                    )

                    created_ids: list[str] = []
                    for sample in samples:
                        voice_id = sample["id"]
                        voice_dir = provider.voices_dir / voice_id
                        try:
                            if voice_dir.exists():
                                shutil.rmtree(voice_dir, ignore_errors=True)
                            voice_dir.mkdir(parents=True, exist_ok=True)

                            codes_path = voice_dir / "ref.codes.pt"
                            text_path = voice_dir / "ref.txt"
                            meta_path = voice_dir / "meta.json"

                            # Download pre-encoded codes
                            with urllib.request.urlopen(sample["codes"], timeout=30) as response:
                                codes_path.write_bytes(response.read())

                            # Download reference transcript
                            with urllib.request.urlopen(sample["txt"], timeout=30) as response:
                                ref_text_raw = response.read().decode("utf-8", errors="ignore")
                            ref_text = textwrap.shorten(ref_text_raw.strip(), width=4000, placeholder=" …")
                            text_path.write_text(ref_text, encoding="utf-8")

                            # Write metadata and update in-memory cache
                            meta = VoiceMetadata(
                                id=voice_id,
                                name=sample["name"],
                                created_at=time.time(),
                                ref_text=ref_text,
                                sample_rate=provider.sample_rate,
                                quality=provider.quality_default,
                            )
                            meta_path.write_text(json.dumps(meta.to_json(), indent=2), encoding="utf-8")
                            with provider._metadata_lock:
                                provider._voices[voice_id] = meta

                            created_ids.append(voice_id)
                        except Exception as exc:
                            PrintStyle().error(f"Failed to seed voice {sample['name']}: {exc}")
                            shutil.rmtree(voice_dir, ignore_errors=True)

                    if created_ids:
                        default_id = created_ids[0]
                        provider.set_default_voice(default_id)
                        try:
                            settings.set_settings_delta(
                                {
                                    "tts": {
                                        "neutts": {
                                            "default_voice_id": default_id,
                                        }
                                    }
                                },
                                apply=True,
                            )
                        except Exception as exc:
                            PrintStyle().error(f"Failed to persist NeuTTS default voice: {exc}")

                PrintStyle(level=logging.DEBUG).print("NeuTTS provider warmed up")
            except Exception as e:
                PrintStyle().error(f"Error in preload_neutts: {e}")

        # async tasks to preload
        tasks = [
            preload_embedding(),
            preload_neutts(),
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
