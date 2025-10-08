# Changelog

## Local Memory Parity with Agent Zero
- Default embeddings switched to Hugging Face `sentence-transformers/all-MiniLM-L6-v2` with a dedicated provider.
- Memory store now records real provider/model metadata and prefers local sentence-transformers fallbacks.
- Memory dashboard opens as a modal and accurately reflects the active embedding backend.
- README and knowledge packs updated to document the new architecture and usage.

## Streaming TTS (XTTS + Piper VC) on Python 3.12
- Added selectable XTTS and Piper+OpenVoice engines with streaming and base64 APIs.
- Expanded settings UI and preload checks for resilient local synthesis on modern Python.
- Updated dependencies to Python 3.12-compatible speech stack (torch, TTS, phonemizer, soundfile).
