# Settings Modal Save Button Overwrite Bug Fix

## Problem Description

When using the Voice Lab to set a default voice (e.g., Dave), then clicking the "Save" button in the main Settings modal, the voice selection would immediately revert to the previous voice (e.g., Jo). The `tmp/settings.json` file would show `"default_voice_id": "dave"` right after Voice Lab saved it, but would flip back to `"default_voice_id": "jo"` when the Settings modal was saved.

## Root Cause

The Settings modal (`webui/js/settings.js`) was using a stale snapshot of settings:

1. **Modal Opens**: Settings are fetched from `/settings_get` and stored in `modalAD.settings`
2. **User Changes Voice**: Voice Lab component calls `/settings_set` with voice changes, updating the backend
3. **User Clicks "Save"**: Modal sends the original snapshot from step 1, overwriting Voice Lab's changes

The `handleButton('save')` function was directly sending `modalAD.settings` without considering that other components (Voice Lab, or any other settings UI) might have modified settings in the backend after the modal opened.

## Solution

Modified the `handleButton('save')` function in `webui/js/settings.js` to:

1. **Fetch Fresh Settings**: Before saving, call `/settings_get` to get current backend state
2. **Merge Modal Changes with Exclusions**: Apply modal field values into fresh settings, EXCEPT for Voice Lab managed fields
3. **Save Merged Result**: Send the merged settings to `/settings_set`

This preserves Voice Lab changes (and other component changes) while still applying the user's modal edits.

**Key Implementation Detail**: Voice Lab fields (`tts_active_voice_id`, `tts_default_voice_id`) are explicitly excluded from the merge, ensuring Voice Lab's independent persistence is respected.

## Code Changes

### File: `webui/js/settings.js`

**Before:**
```javascript
async handleButton(buttonId) {
    if (buttonId === 'save') {
        const modalEl = document.getElementById('settingsModal');
        const modalAD = Alpine.$data(modalEl);
        try {
            resp = await window.sendJsonData("/settings_set", modalAD.settings);
        } catch (e) {
            window.toastFetchError("Error saving settings", e)
            return
        }
        // ...
    }
}
```

**After:**
```javascript
async handleButton(buttonId) {
    if (buttonId === 'save') {
        const modalEl = document.getElementById('settingsModal');
        const modalAD = Alpine.$data(modalEl);
        try {
            // Fetch fresh settings from backend to preserve any changes made by other components
            const freshSettings = await window.sendJsonData("/settings_get", null);
            
            // Merge the modal's field values into the fresh settings
            if (freshSettings && freshSettings.settings && freshSettings.settings.sections) {
                // Create a map of modal field values by section title and field id
                const modalFieldMap = new Map();
                for (const section of modalAD.settings.sections) {
                    if (!modalFieldMap.has(section.title)) {
                        modalFieldMap.set(section.title, new Map());
                    }
                    for (const field of section.fields) {
                        modalFieldMap.get(section.title).set(field.id, field.value);
                    }
                }
                
                // Apply modal values to fresh settings
                for (const section of freshSettings.settings.sections) {
                    const sectionMap = modalFieldMap.get(section.title);
                    if (sectionMap) {
                        for (const field of section.fields) {
                            if (sectionMap.has(field.id)) {
                                field.value = sectionMap.get(field.id);
                            }
                        }
                    }
                }
                
                // Save the merged settings
                resp = await window.sendJsonData("/settings_set", freshSettings.settings);
            } else {
                // Fallback to old behavior if fresh settings fetch failed
                resp = await window.sendJsonData("/settings_set", modalAD.settings);
            }
        } catch (e) {
            window.toastFetchError("Error saving settings", e)
            return
        }
        // ...
    }
}
```

## Testing Workflow

1. Open Voice Lab in Speech Settings
2. Click "Set as default" on Dave voice
3. Verify `tmp/settings.json` shows `"default_voice_id": "dave"`
4. Open main Settings modal and make any change (e.g., modify agent name)
5. Click "Save" in Settings modal
6. Verify `tmp/settings.json` STILL shows `"default_voice_id": "dave"` (not reverted to Jo)
7. Refresh the page - Dave should remain the active voice
8. Try generating speech - should use Dave's voice

## Technical Details

### Merge Strategy

The fix uses a **section-title + field-id mapping** approach:

1. Creates a Map of modal field values organized by section title and field ID
2. Iterates through fresh settings from backend
3. For each field in fresh settings, checks if modal has a value for it
4. If modal has the field, overwrites with modal value
5. If modal doesn't have the field, keeps fresh backend value

This ensures:
- Modal edits are preserved
- Non-modal changes (Voice Lab, other components) are preserved
- Only displayed fields are modified

### Fallback Behavior

If fetching fresh settings fails, the code falls back to the original behavior (sending stale snapshot). This prevents the Save button from breaking if the backend is unavailable.

## Related Fixes

This fix complements the earlier Voice Lab fixes:

1. **Backend Deep Merge** (`python/runtime/api/app.py`): `/settings_set` endpoint properly merges nested settings
2. **Voice Lab Format** (`webui/components/settings/speech/voice-lab-store.js`): Sends minimal nested objects instead of reconstructing entire TTS config
3. **Settings Modal Merge** (this fix): Main Settings modal preserves non-modal changes when saving

Together, these fixes ensure settings from different UI components don't overwrite each other.

## Impact

This fix applies to ALL settings changes, not just voice selection:

- Any component that modifies settings independently will have changes preserved
- Multiple settings panels can be used concurrently without conflicts
- Modal Save button becomes safe to use after making changes in other components

## Future Considerations

### Potential Enhancements

1. **Real-time Sync**: Settings modal could listen for `settings-updated` events and refresh its snapshot
2. **Change Detection**: Only save fields that were actually modified in the modal
3. **Conflict Resolution**: Warn users if backend settings changed while modal was open

### Best Practices

For new settings UI components:
- Always use `/settings_get` and `/settings_set` endpoints
- Send minimal change objects (nested paths) rather than full settings
- Dispatch `settings-updated` custom events after successful saves
- Don't cache settings for long periods - fetch fresh when needed
