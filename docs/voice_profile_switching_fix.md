# Voice Profile Switching Fix

## Issue Summary

The Voice Lab "Set as default" button was not persisting voice selection across page reloads and runtime restarts. Additionally, users reported streaming audio cutting off after the first sentence.

## Root Causes

### Issue 1: Voice Switching Failure

**Problem:** The `voice-lab-store.js` `setDefaultVoice()` function had a critical bug on line 193:

```javascript
await persistSettings({ stream_default: settings.tts?.stream_default ?? true, neutts });
```

This was **overwriting the entire `tts` object** with only `stream_default` and `neutts`, which **deleted** all other TTS settings like `provider` and `sample_rate`.

**Impact:** When the backend's `settings.py` processed this malformed structure, it couldn't properly persist the nested `neutts.default_voice_id` field because the TTS structure was incomplete. The settings would appear to save but would immediately revert to the default (Jo) on the next settings load.

### Issue 2: Settings Endpoint Format Mismatch

**Problem:** The `/settings_set` endpoint only accepted UI format (sections/fields structure), but `voice-lab-store.js` was sending raw settings format (flat dictionary).

**Impact:** While the previous fix added the endpoint, it couldn't handle programmatic settings updates from JavaScript code.

### Issue 3: Streaming Audio "Cutoff"

**Problem:** This was actually a perception issue caused by Issue 1.

**Analysis:** The logs show streaming is working correctly (chunks 1-16+ being received and played). The "cutoff" was likely because:
1. The voice was still Jo (not Dave as expected)
2. The audio was completing normally - chunk counts vary by text length

## Solutions Implemented

### 1. Fixed `/settings_set` Endpoint (python/runtime/api/app.py)

Added format detection and deep merge support:

```python
@router.post("/settings_set", tags=["Admin"])
async def settings_set_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Update settings from the web UI or programmatic calls."""
    from python.helpers import settings as settings_helper
    try:
        # Detect format: UI format has "sections" key, raw format doesn't
        if "sections" in payload:
            # UI format: convert from sections/fields structure
            normalized = settings_helper.normalize_settings(settings_helper.convert_in(payload))
        else:
            # Raw format: merge directly with current settings
            current = settings_helper.get_settings()
            # Deep merge the payload into current settings
            merged = _deep_merge_settings(dict(current), payload)
            normalized = settings_helper.normalize_settings(merged)  # type: ignore[arg-type]
        
        settings_helper.set_settings(normalized)
        return {"settings": settings_helper.convert_out(settings_helper.get_settings())}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to update settings: {exc}"
        ) from exc


def _deep_merge_settings(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge update dict into base dict, handling nested structures properly."""
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = _deep_merge_settings(result[key], value)
        else:
            # Overwrite with new value
            result[key] = value
    return result
```

### 2. Fixed `voice-lab-store.js` setDefaultVoice()

Simplified the function to properly pass nested settings:

```javascript
async setDefaultVoice(voiceId) {
  this.loading = true;
  this.error = null;
  try {
    const settings = await loadSettings();
    const neutts = { ...(settings.tts?.neutts || {}), default_voice_id: voiceId };
    // Just pass the nested neutts object - persistSettings will handle merging
    await persistSettings({ neutts });
    await fetchApi("/runtime/tts/default", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ voice_id: voiceId }),
    });
    this.defaultVoiceId = voiceId;
    speechStore.tts_active_voice_id = voiceId;
  } catch (err) {
    console.error("[voice-lab] Failed to set default voice", err);
    this.error = err?.message || String(err);
  } finally {
    this.loading = false;
  }
}
```

## Verification Steps

1. **Restart the runtime** to load the updated code
2. **Open Voice Lab** in Settings → Speech
3. **Click "Set as default"** on Dave voice
4. **Verify in Network tab:**
   - POST `/settings_set` returns 200
   - POST `/runtime/tts/default` returns 200
5. **Check `tmp/settings.json`:**
   - Should show `"default_voice_id": "dave"` under `tts.neutts`
6. **Reload the page:**
   - Voice Lab should still show Dave as default
   - Console should log `Setting this.tts_active_voice_id = dave`
7. **Send a chat message:**
   - Audio should stream with Dave's voice
   - Console should log `voice_id: 'dave'` in TTS requests
8. **Restart `./run.sh`:**
   - Dave should remain the default voice after full restart

## Technical Notes

### Settings Flow

1. **Voice Lab** calls `setDefaultVoice()`
2. **Frontend** sends raw settings format to `/settings_set`
3. **Backend** detects format and deep merges with current settings
4. **Backend** normalizes and persists to `tmp/settings.json`
5. **Backend** also updates `/runtime/tts/default` endpoint
6. **Provider** updates in-memory default voice
7. **Frontend** updates local state and speech store

### Deep Merge Behavior

The deep merge preserves all existing TTS settings while only updating the `default_voice_id`:

```
Current: { tts: { provider: "neutts", sample_rate: 24000, neutts: { quality: "q4", default_voice_id: "jo" } } }
Update:  { tts: { neutts: { default_voice_id: "dave" } } }
Result:  { tts: { provider: "neutts", sample_rate: 24000, neutts: { quality: "q4", default_voice_id: "dave" } } }
```

## Future Improvements

1. Add client-side caching to prevent instant reversion on network delays
2. Add server-side logging for settings updates to aid debugging
3. Consider toast notifications for successful voice changes
4. Add validation to ensure voice ID exists before setting as default

## Related Files

- `python/runtime/api/app.py` - FastAPI endpoints
- `webui/components/settings/speech/voice-lab-store.js` - Voice Lab UI store
- `python/helpers/settings.py` - Settings normalization and persistence
- `tmp/settings.json` - Persisted settings file
- `docs/neutts_voice_default_fix.md` - Original fix documentation
