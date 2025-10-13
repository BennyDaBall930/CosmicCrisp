# NeuTTS Voice Cloning Fix – Summary

## Executive Overview

NeuTTS voice cloning previously failed because the preload step attempted to install and
use the full `neucodec` encoder. The package now depends on `transformers>=4.44.2` while
still importing `HubertModel`, which was removed from the transformers API. Every startup
therefore crashed with `ModuleNotFoundError: neucodec`, leaving the runtime without any
voices.

The new implementation bypasses the broken dependency chain entirely by using the
official **pre-encoded** voices published in the `neuphonic/neutts-air` repository. Voice
registration is now a simple file copy operation and no longer requires local encoding.

---

## Key Changes

### 1. Codec Loading Simplified (`python/third_party/neuttsair/neutts.py`)
- The ONNX decoder path no longer tries to import the full `neucodec` package.
- `self.encoder` is explicitly set to `None` when ONNX is active.
- `encode_reference()` raises `NotImplementedError` if encoding is requested while using
  the ONNX codec, clarifying that pre-encoded assets are mandatory.

### 2. Provider Guard Rails (`python/runtime/audio/neutts_provider.py`)
- `register_voice()` now rejects registration attempts when the ONNX decoder is
  configured, directing users to the pre-encoded workflow.
- Existing validation logic for the legacy codec remains intact for future compatibility.

### 3. Preload Voice Seeding (`preload.py`)
- Downloads `jo.pt`, `jo.txt`, `dave.pt`, and `dave.txt` from the official repo on first
  launch.
- Creates `data/tts/neutts/voices/<voice_id>/ref.codes.pt`, `ref.txt`, and `meta.json`
  directly—no temporary `.wav` files or encoding steps.
- Updates in-memory metadata and persists the default voice selection.

### 4. Documentation Refresh
- `docs/NEUTTS_QUICKSTART.md` now describes the zero-config experience and the updated
  custom voice workflow.
- `docs/neutts_voice_registration_guide.md` explains how to import pre-encoded voices and
  why local encoding is disabled.

---

## Operational Impact

- **Startup succeeds** without attempting to resolve impossible `neucodec` dependencies.
- **Voices load instantly**, ensuring the UI always has Jo and Dave available.
- **Users receive clear guidance** when trying to register new voices through the UI.
- **Future extensibility** remains: should a full encoder become viable again, the guard
  checks can be relaxed without touching the new seeding logic.

---

## Testing & Validation

1. Run `./run.sh` – verify the preload logs show successful downloads and no encoding
   attempts.
2. Ensure `data/tts/neutts/voices/jo/` and `/dave/` contain `ref.codes.pt`, `ref.txt`,
   and `meta.json`.
3. Confirm `python scripts/validate_voices.py` passes.
4. Exercise synthesis in the UI for both voices and check audio quality.
5. Attempt to register a new voice to validate the guard rails trigger with the new
   message.

---

## Follow-Up Opportunities

- Provide a convenience script that fetches community pre-encoded voices.
- Auto-detect and clean up legacy `.npy` assets once the migration is complete.
- Add telemetry/logging around voice usage to surface missing assets earlier.

This release restores NeuTTS voice cloning to a reliable, zero-maintenance state while
keeping the door open for advanced users to add their own voices using the official
tooling.
