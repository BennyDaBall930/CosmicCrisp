import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from python.helpers.chatterbox_tts import ChatterboxBackend, ChatterboxConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Chatterbox TTS smoke test")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--voice", default=None, help="Optional reference voice WAV path")
    parser.add_argument(
        "--device",
        default=None,
        choices=[None, "auto", "mps", "cuda", "cpu"],
        help="Device override (default auto selection)",
    )
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Emotion intensity (0-1)")
    parser.add_argument("--cfg", type=float, default=0.5, help="Classifier-free guidance weight")
    parser.add_argument("--multilingual", action="store_true", help="Use multilingual checkpoint")
    parser.add_argument("--language_id", default="en", help="Language code when multilingual")
    parser.add_argument("--out", default="out.wav", help="Output WAV path")
    args = parser.parse_args()

    cfg = ChatterboxConfig(
        device=None if args.device in (None, "auto") else args.device,
        multilingual=args.multilingual,
        exaggeration=args.exaggeration,
        cfg=args.cfg,
        audio_prompt_path=args.voice,
        language_id=args.language_id,
    )
    backend = ChatterboxBackend(cfg)
    wav_bytes = backend.synthesize(args.text)
    output_path = pathlib.Path(args.out)
    output_path.write_bytes(wav_bytes)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
