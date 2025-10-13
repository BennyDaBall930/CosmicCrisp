# NeuTTS Voice Registration Guide

NeuTTS now operates in a decoder-only configuration using the official ONNX codec.
Because the ONNX decoder cannot encode audio, **all voices must be provided as
pre-encoded `ref.codes.pt` files**. This guide explains how the runtime seeds the
default voices and how you can add additional voices manually.

---

## Default Voice Seeding (Jo & Dave)

On first launch `preload.py` downloads `jo.pt`, `jo.txt`, `dave.pt`, and `dave.txt`
directly from `https://github.com/neuphonic/neutts-air/tree/main/samples`. It writes the
following directories:

```
data/tts/neutts/voices/
├── jo/
│   ├── ref.codes.pt
│   ├── ref.txt
│   └── meta.json
└── dave/
    ├── ref.codes.pt
    ├── ref.txt
    └── meta.json
```

The metadata JSON matches the `VoiceMetadata` dataclass in
`python/runtime/audio/neutts_provider.py`. Jo is set as the default voice and the choice
is persisted to `conf/tts.toml`.

---

## Why Local Encoding Is Disabled

The original workflow attempted to install the full `neucodec` package so the runtime
could encode `.wav` files on demand. That path requires `transformers==4.30.0` and
`HubertModel`, creating an unsolvable conflict with the current transformer stack.

The updated runtime:

- Loads only the ONNX decoder (decoder-only, no encoder).
- Sets `NeuTTSAir.encoder = None` in ONNX mode.
- Raises an explicit error in `NeuTTSAir.encode_reference()` and
  `NeuttsProvider.register_voice()` when encoding is requested.

As a result, the Voice Lab “Register Voice” action reports:

> `NeuTTS is configured with the ONNX decoder, which cannot encode reference audio. Use
> pre-encoded reference files instead.`

This protects the runtime from broken dependency chains while keeping synthesis fast.

---

## Adding Custom Voices

1. **Obtain a pre-encoded `.pt` file**
   - Use the official NeuTTS-Air tooling (`scripts/encode_reference.py`) in a separate
     environment or Docker container.
   - Download a community-provided `.pt` file that matches the NeuTTS-Air format.
2. **Create a new voice directory**

```bash
VOICE_ID=my-voice
TARGET=data/tts/neutts/voices/${VOICE_ID}
mkdir -p "${TARGET}"
cp /path/to/ref.codes.pt "${TARGET}/"
cp /path/to/ref.txt "${TARGET}/"
```

3. **Write metadata**

`meta.json` must include the fields below. Adjust `name`, `ref_text`, and timestamps as
needed.

```json
{
  "id": "my-voice",
  "name": "My Voice",
  "created_at": 1725955200.0,
  "ref_text": "Text spoken in the reference audio.",
  "sample_rate": 24000,
  "quality": "q4",
  "watermarked": true
}
```

4. **Restart the application**

The provider scans `data/tts/neutts/voices/` at startup and loads every folder with a
`meta.json`. No additional registration step is necessary.

---

## Validation

- `python scripts/validate_voices.py` — verifies metadata and ensures the `.pt` tensor
  loads successfully.
- `python -c "from python.runtime.container import get_tts_provider; print(get_tts_provider().list_voices())"`
  — lists the voices recognized by the runtime.
- Logs during preload include any download or parsing errors.

If validation fails, remove the incomplete directory and rerun `./run.sh` to trigger
seeding again.

---

## Tips for Creating `ref.codes.pt`

- The tensor should be a 1D sequence of `int32` codes saved with `torch.save`.
- Reference audio should be 3–15 seconds and recorded cleanly at or resampled to 16 kHz.
- Include the exact transcript used during encoding in `ref.txt` for best results.
- Watermarking is handled at inference time; no extra steps required.

For a fully reproducible pipeline, run the official encoder in Docker using
`scripts/docker_encode_voices_fixed.sh`, then copy the outputs into the voice directory.

---

## Troubleshooting

- **UI says registration is disabled** — expected when ONNX decoder is active. Provide a
  `.pt` file instead of a `.wav`.
- **Voice missing from list** — confirm `meta.json` exists and is valid JSON. The provider
  ignores directories without metadata.
- **Audio sounds wrong** — ensure the `.pt` file came from NeuTTS-Air and not an older
  format (`.npy`). The runtime still supports legacy `.npy` files, but pre-encoded `.pt`
  offers higher fidelity.
- **Need to reset** — delete `data/tts/neutts/voices/` and rerun `./run.sh` to restore the
  default voices.

---

By relying on official pre-encoded assets, NeuTTS remains stable while keeping the door
open for power users to import their own high-quality voices.
