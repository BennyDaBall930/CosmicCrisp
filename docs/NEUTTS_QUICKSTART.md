# NeuTTS Voice Cloning Quick Start

NeuTTS now boots with fully functional voice cloning using the official pre-encoded
references from the `neuphonic/neutts-air` project. No local encoding or legacy
`neucodec` environments are required.

---

## Prerequisites
- Cosmic Crisp dependencies installed via `./setup.sh`
- Internet access on first launch (downloads the official `jo.pt` and `dave.pt`)
- macOS 13+ recommended for MPS acceleration (falls back to CPU if unavailable)

---

## Step 1 — Standard Setup

```bash
./setup.sh
```

This prepares all runtime dependencies. There is no separate encoding environment to
bootstrap anymore.

**Time:** ~5–10 minutes on a clean machine

---

## Step 2 — Launch the App

```bash
./run.sh
```

During preload the runtime will:
- download `jo.pt`, `jo.txt`, `dave.pt`, and `dave.txt` from the official repository,
- populate `data/tts/neutts/voices/jo/` and `data/tts/neutts/voices/dave/`,
- generate the corresponding `meta.json` files, and
- select Jo as the default voice (config persisted to `conf/tts.toml`).

When the UI is ready you should see **Jo** and **Dave** in the NeuTTS voice picker. Both
voices work immediately because the ONNX decoder only needs the pre-encoded `.pt` files.

---

## Step 3 — Try Voice Cloning

1. Open the Voice Lab tab in the UI.
2. Choose **Jo** or **Dave** from the list.
3. Type any prompt and generate speech.

Playback is watermarked automatically. The ONNX decoder keeps inference fast even on CPU,
while the GGUF backbone runs on MPS/GPU when available.

---

## Anatomy of a Voice Folder

Each voice lives in `data/tts/neutts/voices/<voice_id>/` and contains:

| File | Purpose |
|------|---------|
| `ref.codes.pt` | Pre-encoded latent codes (Torch tensor) |
| `ref.txt` | Reference transcript used during encoding |
| `meta.json` | Metadata matching `python/runtime/audio/neutts_provider.py::VoiceMetadata` |

Example `meta.json`:

```json
{
  "id": "jo",
  "name": "Jo",
  "created_at": 1725955200.123,
  "ref_text": "Hey there, I'm Jo!",
  "sample_rate": 24000,
  "quality": "q4",
  "watermarked": true
}
```

---

## Adding Custom Voices

The ONNX codec cannot encode audio locally. To add new voices you must provide your own
pre-encoded `ref.codes.pt`. Two common options:

1. **Use the official NeuTTS-Air tooling**
   - Clone `https://github.com/neuphonic/neutts-air`
   - Follow their `README` to run `scripts/encode_reference.py`
   - Copy the generated `.pt` file and transcript into
     `data/tts/neutts/voices/<your-id>/`
2. **Download third-party pre-encoded voices** that ship compatible `.pt` files.

After copying the files:

```bash
mkdir -p data/tts/neutts/voices/my-voice
cp /path/to/ref.codes.pt data/tts/neutts/voices/my-voice/
cp /path/to/ref.txt data/tts/neutts/voices/my-voice/
```

Create a matching `meta.json` (use the template above), then restart the app. The new
voice will appear automatically.

> **Note:** The “Register Voice” UI action now displays an error when the ONNX decoder is
> active. This is expected—encoding is intentionally disabled to avoid the broken
> `neucodec` dependency chain.

---

## Validation & Troubleshooting

- `python scripts/validate_voices.py` — sanity-checks the voice folders and ensures the
  `.pt` files load.
- Voice directories missing? Delete `data/tts/neutts/voices/` and rerun `./run.sh` to
  trigger automatic seeding.
- Seeing “ONNX decoder cannot encode reference audio”? You attempted to register a voice
  through the UI while still on the ONNX codec. Provide pre-encoded files instead.
- Empty audio output? Verify the `.pt` file matches the official format (Torch tensor of
  shape `[N]` with `int32` codes).

---

## Reference Scripts

| Script | Purpose |
|--------|---------|
| `scripts/docker_encode_voices_fixed.sh` | Optional helper to run the official encoder in Docker if you need custom voices |
| `scripts/validate_voices.py` | Confirms voice metadata + codes are usable |
| `scripts/setup_encoding_env.sh` | Legacy helper for manual encoding (only needed if you choose to manage encoding yourself) |

---

NeuTTS voice cloning should now “just work” out of the box. Enjoy experimenting with the
voices and share high-quality pre-encoded references for others to try! 🎉
