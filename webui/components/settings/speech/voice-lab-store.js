import { createStore } from "/js/AlpineStore.js";
import { fetchApi } from "/js/api.js";
import { store as speechStore } from "/components/chat/speech/speech-store.js";

function bufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

async function fileToBase64(file) {
  const arrayBuffer = await file.arrayBuffer();
  return bufferToBase64(arrayBuffer);
}

async function loadSettings() {
  const response = await fetchApi("/settings_get", { method: "POST" });
  const payload = await response.json();
  return payload?.settings ?? {};
}

async function persistSettings(patch) {
  const settings = await loadSettings();
  const next = {
    ...settings,
    tts: {
      ...(settings.tts || {}),
      ...patch,
    },
  };
  await fetchApi("/settings_set", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(next),
  });
  return next;
}

const model = {
  voices: [],
  loading: false,
  error: null,
  nameInput: "",
  refTextInput: "",
  fileInput: null,
  defaultVoiceId: null,
  qualityDefault: "q4",

  async init() {
    await this.refresh();
    try {
      const settings = await loadSettings();
      const tts = settings.tts || {};
      const neutts = tts.neutts || {};
      this.defaultVoiceId = neutts.default_voice_id || null;
      this.qualityDefault = neutts.quality_default || "q4";
    } catch (err) {
      console.warn("[voice-lab] Failed to load settings", err);
    }
  },

  async refresh() {
    this.loading = true;
    this.error = null;
    try {
      const response = await fetchApi("/runtime/tts/voices", { method: "GET" });
      const payload = await response.json();
      this.voices = payload.voices || [];
    } catch (err) {
      console.error("[voice-lab] Failed to load voices", err);
      this.error = err?.message || String(err);
    } finally {
      this.loading = false;
    }
  },

  setFile(file) {
    this.fileInput = file;
  },

  resetForm() {
    this.nameInput = "";
    this.refTextInput = "";
    this.fileInput = null;
    const fileEl = document.getElementById("voice-lab-file");
    if (fileEl) {
      fileEl.value = "";
    }
  },

  async registerVoice() {
    if (!this.fileInput) {
      this.error = "Select a 3–15s mono WAV reference clip.";
      return;
    }
    const trimmedName = this.nameInput.trim();
    if (!trimmedName) {
      this.error = "Provide a name for the voice.";
      return;
    }
    const refText = this.refTextInput.trim();
    if (!refText) {
      this.error = "Reference text is required.";
      return;
    }
    this.loading = true;
    this.error = null;
    try {
      const audioBase64 = await fileToBase64(this.fileInput);
      const response = await fetchApi("/runtime/tts/voices", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: trimmedName,
          ref_text: refText,
          audio_base64: audioBase64,
        }),
      });
      if (!response.ok) {
        let msg = await response.text();
        try {
          const j = JSON.parse(msg);
          msg = j.detail || j.error || msg;
        } catch (_) {}
        throw new Error(msg || `HTTP ${response.status}`);
      }
      await this.refresh();
      this.resetForm();
    } catch (err) {
      console.error("[voice-lab] Failed to register voice", err);
      this.error = err?.message || String(err);
    } finally {
      this.loading = false;
    }
  },

  async deleteVoice(voiceId) {
    if (!voiceId) return;
    this.loading = true;
    this.error = null;
    try {
      await fetchApi(`/runtime/tts/voices/${voiceId}`, { method: "DELETE" });
      if (this.defaultVoiceId === voiceId) {
        this.defaultVoiceId = null;
        await persistSettings({
          neutts: { ...(await loadSettings()).tts?.neutts, default_voice_id: null },
        });
        await fetchApi("/runtime/tts/default", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ voice_id: null }),
        });
      }
      await this.refresh();
      speechStore.tts_active_voice_id = this.defaultVoiceId;
    } catch (err) {
      console.error("[voice-lab] Failed to delete voice", err);
      this.error = err?.message || String(err);
    } finally {
      this.loading = false;
    }
  },

  async resetVoices() {
    this.loading = true;
    this.error = null;
    try {
      await fetchApi("/runtime/tts/voices/reset", { method: "POST" });
      this.defaultVoiceId = null;
      await persistSettings({
        neutts: { ...(await loadSettings()).tts?.neutts, default_voice_id: null },
      });
      await this.refresh();
      speechStore.tts_active_voice_id = this.defaultVoiceId;
    } catch (err) {
      console.error("[voice-lab] Failed to reset voices", err);
      this.error = err?.message || String(err);
    } finally {
      this.loading = false;
    }
  },

  async setDefaultVoice(voiceId) {
    this.loading = true;
    this.error = null;
    try {
      const settings = await loadSettings();
      const neutts = { ...(settings.tts?.neutts || {}), default_voice_id: voiceId };
      await persistSettings({ stream_default: settings.tts?.stream_default ?? true, neutts });
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
  },

  async previewVoice(voice) {
    if (!voice) return;
    try {
      await speechStore.stop();
      await speechStore.speakWithNeutts("NeuTTS sample playback.", true, () => false, voice.id);
    } catch (err) {
      console.error("[voice-lab] Preview failed", err);
      if (window.toast) {
        window.toast(err?.message || "Playback failed", "error", 5000);
      }
    }
  },
};

const store = createStore("voiceLab", model);

export { store };
