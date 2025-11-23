import { createStore } from "/js/AlpineStore.js";
import { fetchApi } from "/js/api.js";
import { store as speechStore } from "/components/chat/speech/speech-store.js";
import { store as microphoneStore } from "/components/settings/speech/microphone-setting-store.js";

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

const model = {
  voices: [],
  loading: false,
  error: null,
  nameInput: "",
  refTextInput: "",
  fileInput: null,
  defaultVoiceId: null,
  qualityDefault: "q4",
  
  // Recording state
  showRecording: false,
  recording: false,
  audioBlob: null,
  audioURL: null,
  recordingDuration: 0,
  formattedDuration: "00:00",
  mediaRecorder: null,
  audioChunks: [],
  recordingStartTime: null,
  timerInterval: null,

  async init() {
    await this.refresh();
    try {
      const settings = await loadSettings();
      const tts = settings.tts || {};
      const supertonic = tts.supertonic || {};
      this.defaultVoiceId = supertonic.default_voice_id || null;
      this.qualityDefault = supertonic.quality_default || "q4";
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
    // Use recorded audio if available, otherwise fall back to file upload
    if (!this.audioBlob && !this.fileInput) {
      this.error = "Please record a voice or upload a WAV file.";
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
    
    // Validate recording duration (3-15 seconds)
    if (this.audioBlob && this.recordingDuration) {
      if (this.recordingDuration < 3) {
        this.error = "Recording is too short. Please record for at least 3 seconds.";
        return;
      }
      if (this.recordingDuration > 15) {
        this.error = "Recording is too long. Please record for no more than 15 seconds.";
        return;
      }
    }
    
    this.loading = true;
    this.error = null;
    try {
      let audioBase64;
      
      if (this.audioBlob) {
        // Use recorded audio
        audioBase64 = await bufferToBase64(await this.audioBlob.arrayBuffer());
      } else if (this.fileInput) {
        // Fall back to file upload
        audioBase64 = await fileToBase64(this.fileInput);
      } else {
        throw new Error("No audio source available");
      }
      
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
      this.hideRecordingScreen();
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
          supertonic: { ...(await loadSettings()).tts?.supertonic, default_voice_id: null },
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
        supertonic: { ...(await loadSettings()).tts?.supertonic, default_voice_id: null },
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
      const supertonic = { ...(settings.tts?.supertonic || {}), default_voice_id: voiceId };
      // Just pass the nested supertonic object - persistSettings will handle merging
      await persistSettings({ supertonic });
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
      await speechStore.speakWithNeutts("Supertonic sample playback.", true, () => false, voice.id);
    } catch (err) {
      console.error("[voice-lab] Preview failed", err);
      if (window.toast) {
        window.toast(err?.message || "Playback failed", "error", 5000);
      }
    }
  },

  // Recording interface methods
  showRecordingScreen() {
    this.showRecording = true;
    this.resetRecording();
  },

  hideRecordingScreen() {
    this.showRecording = false;
    this.stopRecording();
    this.resetRecording();
  },

  async requestMicrophonePermission() {
    try {
      console.log("[voice-lab] Requesting microphone permission...");
      
      // First request basic permission to populate device list
      await navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => stream.getTracks().forEach(track => track.stop()));
      
      // Load devices after permission granted
      await microphoneStore.loadDevices();
      
      // Now get the selected device from microphone settings
      const selectedDevice = microphoneStore.getSelectedDevice();
      console.log("[voice-lab] Using microphone device:", selectedDevice?.label || "default");
      
      // Use browser's native sample rate - we'll resample later
      const constraints = {
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: false
        }
      };
      
      // Add deviceId constraint if specific device selected
      if (selectedDevice && selectedDevice.deviceId && selectedDevice.deviceId !== "default") {
        constraints.audio.deviceId = { exact: selectedDevice.deviceId };
      }
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      
      console.log("[voice-lab] Microphone stream obtained");
      const tracks = stream.getAudioTracks();
      if (tracks.length > 0) {
        const settings = tracks[0].getSettings();
        console.log("[voice-lab] Microphone settings:", settings);
        console.log("[voice-lab] Using device:", tracks[0].label);
      }
      
      return stream;
    } catch (err) {
      console.error("[voice-lab] Microphone access denied or device unavailable:", err);
      this.error = `Microphone error: ${err.message}. Please check device selection in Speech settings.`;
      throw err;
    }
  },

  async startRecording() {
    try {
      this.error = null;
      const stream = await this.requestMicrophonePermission();
      
      // Create MediaRecorder with proper MIME type for WAV
      this.mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      this.audioChunks = [];
      this.recordingStartTime = Date.now();
      
      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };
      
      this.mediaRecorder.onstop = () => {
        this.processRecording();
      };
      
      this.mediaRecorder.start(1000); // Collect data every second for more reliable chunks
      this.recording = true;
      
      // Start timer
      this.timerInterval = setInterval(() => {
        this.updateTimer();
      }, 100);
      
    } catch (err) {
      console.error("[voice-lab] Failed to start recording", err);
      this.error = err?.message || "Failed to start recording";
    }
  },

  stopRecording() {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.mediaRecorder.stop();
      this.recording = false;
      
      // Stop timer
      if (this.timerInterval) {
        clearInterval(this.timerInterval);
        this.timerInterval = null;
      }
      
      // Stop microphone stream
      if (this.mediaRecorder.stream) {
        this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
      }
    }
  },

  updateTimer() {
    if (this.recordingStartTime) {
      const elapsed = Date.now() - this.recordingStartTime;
      this.recordingDuration = elapsed / 1000; // Convert to seconds
      
      const minutes = Math.floor(this.recordingDuration / 60);
      const seconds = Math.floor(this.recordingDuration % 60);
      this.formattedDuration = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
  },

  async processRecording() {
    try {
      console.log("[voice-lab] Processing recording, chunks:", this.audioChunks.length);
      
      if (this.audioChunks.length === 0) {
        this.error = "No audio data recorded. Please try again.";
        return;
      }
      
      // Create blob from chunks
      const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
      console.log("[voice-lab] Created audio blob, size:", audioBlob.size);
      
      if (audioBlob.size === 0) {
        this.error = "Recording produced no data. Please check microphone and try again.";
        return;
      }
      
      this.audioBlob = audioBlob;
      
      // Convert to WAV format (16kHz mono for NeuTTS compatibility)
      console.log("[voice-lab] Starting WAV conversion...");
      const wavBlob = await this.convertToWAV(audioBlob);
      console.log("[voice-lab] WAV conversion complete, size:", wavBlob.size);
      
      this.audioBlob = wavBlob;
      
      // Create URL for preview
      this.audioURL = URL.createObjectURL(wavBlob);
      console.log("[voice-lab] Preview URL created:", this.audioURL);
      
      // Auto-fill reference text with script
      if (!this.refTextInput.trim()) {
        this.refTextInput = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.";
      }
      
      console.log("[voice-lab] Recording processed successfully");
      
    } catch (err) {
      console.error("[voice-lab] Failed to process recording", err);
      this.error = `Failed to process recording: ${err.message || 'Unknown error'}. Please try again.`;
    }
  },

  async convertToWAV(audioBlob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = async () => {
        try {
          const arrayBuffer = reader.result;
          console.log("[voice-lab] Audio blob size:", arrayBuffer.byteLength);
          
          if (arrayBuffer.byteLength === 0) {
            reject(new Error("Audio data is empty"));
            return;
          }
          
          const audioContext = new (window.AudioContext || window.webkitAudioContext)();
          console.log("[voice-lab] AudioContext created, decoding audio data...");
          
          let audioBuffer;
          try {
            audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
          } catch (decodeError) {
            console.error("[voice-lab] Failed to decode audio data:", decodeError);
            reject(new Error("Could not decode audio - microphone may not be working"));
            return;
          }
          
          // DEBUG: Log audio buffer info
          console.log("[voice-lab] Audio buffer decoded successfully:", {
            duration: audioBuffer.duration,
            sampleRate: audioBuffer.sampleRate,
            channels: audioBuffer.numberOfChannels,
            length: audioBuffer.length
          });
          
          if (audioBuffer.length === 0) {
            reject(new Error("Audio buffer is empty - no sound was captured"));
            return;
          }
          
          // Get the channel data
          const sourceData = audioBuffer.getChannelData(0);
          
          // Check if audio has any content at all - sample across entire recording
          let maxAmplitude = 0;
          let sumSquares = 0;
          const sampleStep = Math.max(1, Math.floor(sourceData.length / 1000)); // Sample 1000 points across recording
          for (let i = 0; i < sourceData.length; i += sampleStep) {
            const absValue = Math.abs(sourceData[i]);
            if (absValue > maxAmplitude) {
              maxAmplitude = absValue;
            }
            sumSquares += sourceData[i] * sourceData[i];
          }
          const rms = Math.sqrt(sumSquares / sourceData.length);
          console.log("[voice-lab] Audio analysis:", {
            maxAmplitude,
            rms,
            duration: audioBuffer.duration,
            samples: sourceData.length
          });
          
          // Warn if audio seems very quiet but don't block it - let backend validate
          if (maxAmplitude < 0.001) {
            console.warn("[voice-lab] Audio amplitude is very low. You may need to speak louder or check microphone settings.");
          }
          
          // Create 16kHz mono buffer using Web Audio API properly
          const targetSampleRate = 16000;
          const ratio = audioBuffer.sampleRate / targetSampleRate;
          const targetLength = Math.floor(audioBuffer.length / ratio);
          
          console.log("[voice-lab] Resampling from", audioBuffer.sampleRate, "Hz to", targetSampleRate, "Hz");
          console.log("[voice-lab] Target length:", targetLength, "samples");
          
          if (targetLength === 0) {
            reject(new Error("Resampling calculation resulted in zero length"));
            return;
          }
          
          // Create offline context for resampling
          const offlineContext = new OfflineAudioContext(1, targetLength, targetSampleRate);
          const source = offlineContext.createBufferSource();
          source.buffer = audioBuffer;
          source.connect(offlineContext.destination);
          source.start(0);
          
          // Render the resampled audio
          console.log("[voice-lab] Starting offline rendering...");
          const renderedBuffer = await offlineContext.startRendering();
          console.log("[voice-lab] Rendered buffer length:", renderedBuffer.length);
          
          if (renderedBuffer.length === 0) {
            reject(new Error("Resampling produced no audio data"));
            return;
          }
          
          // Convert to WAV
          const wavBlob = this.audioBufferToWAV(renderedBuffer);
          console.log("[voice-lab] WAV blob size:", wavBlob.size);
          
          if (wavBlob.size === 0) {
            reject(new Error("WAV conversion produced no data"));
            return;
          }
          
          await audioContext.close();
          resolve(wavBlob);
        } catch (err) {
          console.error("[voice-lab] Conversion error:", err);
          reject(new Error(`Audio conversion failed: ${err.message}`));
        }
      };
      reader.onerror = (err) => {
        console.error("[voice-lab] FileReader error:", err);
        reject(new Error("Failed to read audio data"));
      };
      reader.readAsArrayBuffer(audioBlob);
    });
  },

  audioBufferToWAV(audioBuffer) {
    const length = audioBuffer.length;
    const sampleRate = audioBuffer.sampleRate;
    const channels = audioBuffer.numberOfChannels;
    const format = 1; // PCM
    const bitDepth = 16;
    
    const arrayBuffer = new ArrayBuffer(44 + length * 2);
    const view = new DataView(arrayBuffer);
    
    // WAV header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, channels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length * 2, true);
    
    // Convert float samples to 16-bit PCM
    let offset = 44;
    for (let i = 0; i < length; i++) {
      const sample = Math.max(-1, Math.min(1, audioBuffer.getChannelData(0)[i]));
      view.setInt16(offset, sample * 0x7FFF, true);
      offset += 2;
    }
    
    return new Blob([arrayBuffer], { type: 'audio/wav' });
  },

  async playRecording() {
    if (!this.audioURL) return;
    
    try {
      const audio = new Audio(this.audioURL);
      await audio.play();
    } catch (err) {
      console.error("[voice-lab] Failed to play recording", err);
      this.error = "Failed to play recording";
    }
  },

  resetRecording() {
    this.recording = false;
    this.audioBlob = null;
    this.audioURL = null;
    this.recordingDuration = 0;
    this.formattedDuration = "00:00";
    this.audioChunks = [];
    this.recordingStartTime = null;
    
    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }
    
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      this.stopRecording();
    }
  },
};

const store = createStore("voiceLab", model);

export { store };
