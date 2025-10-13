# NeuTTS Voice Default Persistence Fix

## Problem Summary

The Voice Lab "Set as default" button was not persisting the selected voice across page reloads or runtime restarts. The issue occurred because:

1. **Missing API endpoints**: The frontend called `/settings_get` and `/settings_set` endpoints that no longer existed after migration from Flask to FastAPI
2. **No persistence in `/runtime/tts/default`**: The endpoint only updated the in-memory provider state but didn't save to `tmp/settings.json`
3. **Silent failures**: The frontend didn't check response status, so errors went unnoticed

## Changes Made

### 1. Added FastAPI Settings Endpoints (`python/runtime/api/app.py`)

```python
@router.post("/settings_get", tags=["Admin"])
async def settings_get_endpoint() -> Dict[str, Any]:
    """Get current settings for the web UI."""
    from python.helpers import settings as settings_helper
    return {"settings": settings_helper.convert_out(settings_helper.get_settings())}


@router.post("/settings_set", tags=["Admin"])
async def settings_set_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Update settings from the web UI."""
    from python.helpers import settings as settings_helper
    try:
        normalized = settings_helper.normalize_settings(settings_helper.convert_in(payload))
        settings_helper.set_settings(normalized)
        return {"settings": settings_helper.convert_out(settings_helper.get_settings())}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to update settings: {exc}"
        ) from exc
```

### 2. Updated `/runtime/tts/default` to Persist Settings (`python/runtime/api/app.py`)

```python
@router.post("/tts/default", tags=["TTS"])
async def tts_set_default(payload: TTSDefaultRequest) -> Dict[str, Any]:
    provider = get_tts_provider()
    provider.set_default_voice(payload.voice_id)
    
    # Persist the default voice to settings
    from python.helpers import settings as settings_helper
    settings_helper.set_settings_delta({
        "tts": {
            "neutts": {
                "default_voice_id": payload.voice_id
            }
        }
    })
    
    return {"default_voice_id": payload.voice_id}
```

### 3. Added Error Handling to Frontend (`webui/components/settings/speech/voice-lab-store.js`)

```javascript
async function persistSettings(patch) {
  const settings = await loadSettings();
  const next = {
    ...settings,
    tts: {
      ...(settings.tts || {}),
      ...patch,
    },
  };
  const response = await fetchApi("/settings_set", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(next),
  });
  
  if (!response.ok) {
    const errorText = await response.text();
    let errorMessage = `HTTP ${response.status}`;
    try {
      const errorJson = JSON.parse(errorText);
      errorMessage = errorJson.detail || errorJson.error || errorMessage;
    } catch (_) {
      errorMessage = errorText || errorMessage;
    }
    throw new Error(`Failed to save settings: ${errorMessage}`);
  }
  
  return next;
}
```

## Verification Steps

### 1. Restart the Runtime

```bash
./run.sh
```

### 2. Test Voice Default Persistence

1. Open the web UI and navigate to Settings → Speech
2. In the Voice Lab section, select a voice (e.g., "Dave") and click "Set as default"
3. Verify in browser console:
   - Network tab shows `POST /settings_set` returning 200 OK
   - Network tab shows `POST /runtime/tts/default` returning 200 OK
   - Console log shows: `Setting this.tts_active_voice_id = dave`

### 3. Verify Settings File Updated

```bash
cat tmp/settings.json | grep default_voice_id
```

Should show:
```json
"default_voice_id": "dave"
```

### 4. Test Page Reload Persistence

1. Refresh the page (Cmd+R / Ctrl+R)
2. Navigate back to Settings → Speech
3. Verify Dave is still shown as the default voice
4. Check browser console for: `Setting this.tts_active_voice_id = dave`

### 5. Test Runtime Restart Persistence

1. Stop the runtime (Ctrl+C)
2. Restart: `./run.sh`
3. Open the web UI and check Settings → Speech
4. Verify Dave is still the default voice
5. Test synthesis to confirm Dave's voice is used by default

### 6. Test Synthesis

1. In the chat, send a message and enable TTS playback
2. Verify the synthesized speech uses Dave's voice (not Jo)
3. Check network tab: `POST /runtime/tts/speak` should show `X-NeuTTS-Voice-Id: dave` in response headers

## Expected Results

✅ **Before Fix:**
- Setting default voice worked only for current session
- Page reload reverted to Jo
- Runtime restart reverted to Jo
- `tmp/settings.json` never updated with new voice
- Silent failures in console (404 errors for `/settings_get`, `/settings_set`)

✅ **After Fix:**
- Setting default voice persists across page reloads
- Setting default voice persists across runtime restarts
- `tmp/settings.json` properly updates with `"default_voice_id": "dave"`
- All API calls return 200 OK
- No console errors related to settings endpoints
- Speech synthesis uses the selected default voice

## Technical Notes

### Settings Flow

1. **User clicks "Set as default"** → `voice-lab-store.js::setDefaultVoice()`
2. **Frontend calls** → `POST /settings_set` with updated settings object
3. **Backend processes** → `settings_helper.normalize_settings()` + `settings_helper.set_settings()`
4. **Settings written** → `tmp/settings.json` with `default_voice_id`
5. **Frontend calls** → `POST /runtime/tts/default` to update provider
6. **Backend updates** → Provider in-memory state + persists via `set_settings_delta()`
7. **UI updates** → Local state reflects new default

### Persistence Mechanism

- **File**: `tmp/settings.json` stores the persistent configuration
- **Loader**: `preload.py` reads settings on startup and sets provider default
- **Sync**: Both `/settings_set` and `/runtime/tts/default` now persist changes
- **Normalization**: `settings_helper.normalize_settings()` ensures TTS structure is valid

### Error Handling

- Frontend now checks `response.ok` and throws descriptive errors
- Backend returns proper HTTP status codes (400 for validation, 500 for server errors)
- User sees error messages in Voice Lab UI if save fails

## Related Files

- `python/runtime/api/app.py` - FastAPI endpoints
- `python/helpers/settings.py` - Settings normalization and persistence
- `webui/components/settings/speech/voice-lab-store.js` - Frontend store
- `webui/components/chat/speech/speech-store.js` - Speech synthesis store
- `tmp/settings.json` - Persistent settings storage
- `preload.py` - Settings loading on startup
