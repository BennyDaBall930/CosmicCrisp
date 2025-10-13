import { fetchApi } from "/js/api.js";

const GT = typeof globalThis !== "undefined" ? globalThis : {};
const DBG = typeof GT.__COSMIC_TTS_DEBUG === "boolean" ? GT.__COSMIC_TTS_DEBUG : true;
const log = (...args) => {
  if (DBG && typeof console !== "undefined" && console.debug) {
    console.debug("[tts-stream]", ...args);
  }
};

export async function playStreamedTTS({
  text,
  voiceId = null,
  sampleRate = 24000,
  stream = true,
  endpoint = "/runtime/tts/speak",
}) {
  if (!text || !text.trim()) return null;

  const ctx = new AudioContext();
  let effectiveSampleRate = sampleRate;
  const deviceRate = ctx.sampleRate;
  let needResample = Math.abs(deviceRate - effectiveSampleRate) > 1;
  const capture = GT.__COSMIC_TTS_CAPTURE ? { buffers: [], rate: deviceRate } : null;

  if (ctx.state !== "running") {
    await ctx.resume();
  }

  const activeSources = new Set();
  let nextStart = ctx.currentTime;
  const abort = new AbortController();

  const state = {
    totalSamples: 0,
    chunks: 0,
    closed: false,
    cleanupTimer: null,
  };

  const cleanup = async ({ immediate = false } = {}) => {
    if (state.closed) return;
    state.closed = true;
    if (state.cleanupTimer) {
      clearTimeout(state.cleanupTimer);
      state.cleanupTimer = null;
    }

    try {
      activeSources.forEach((source) => {
        try {
          source.stop();
        } catch (_) {}
        try {
          source.disconnect();
        } catch (_) {}
      });
    } catch (_) {}
    activeSources.clear();
    nextStart = ctx.currentTime;

    if (immediate) {
      try {
        await ctx.close();
      } catch (_) {}
    }
  };

  const parseWavHeader = (bytes) => {
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (
      dv.byteLength < 12 ||
      dv.getUint32(0, false) !== 0x52494646 ||
      dv.getUint32(8, false) !== 0x57415645
    ) {
      throw new Error("Not a RIFF/WAVE stream");
    }
    let offset = 12;
    let fmt = null;
    let blockAlign = 0;

    while (offset + 8 <= dv.byteLength) {
      const id = dv.getUint32(offset, false);
      const size = dv.getUint32(offset + 4, true);
      offset += 8;

      if (id === 0x666d7420) {
        const audioFormat = dv.getUint16(offset + 0, true);
        const numChannels = dv.getUint16(offset + 2, true);
        const sr = dv.getUint32(offset + 4, true);
        const bitsPerSample = dv.getUint16(offset + 14, true);
        blockAlign = dv.getUint16(offset + 12, true);
        fmt = { audioFormat, numChannels, sampleRate: sr, bitsPerSample };
      }

      if (id === 0x64617461) {
        if (!fmt) {
          throw new Error("Missing fmt chunk before data");
        }
        if (fmt.audioFormat !== 1 || fmt.bitsPerSample !== 16) {
          throw new Error("Unsupported WAV encoding");
        }
        if (!blockAlign) {
          blockAlign = (fmt.numChannels * fmt.bitsPerSample) >> 3;
        }
        return { dataOffset: offset, fmt, blockAlign };
      }

      offset += size + (size & 1);
    }

    return null;
  };

  const int16leToMonoFloat32 = (int16, channels) => {
    if (channels === 1) {
      const out = new Float32Array(int16.length);
      for (let i = 0; i < int16.length; i += 1) {
        out[i] = Math.max(-1, Math.min(1, int16[i] / 32768));
      }
      return out;
    }
    const frames = Math.floor(int16.length / channels);
    const out = new Float32Array(frames);
    for (let i = 0, f = 0; f < frames; f += 1, i += channels) {
      const L = int16[i];
      const R = int16[i + 1];
      out[f] = Math.max(-1, Math.min(1, ((L + R) * 0.5) / 32768));
    }
    return out;
  };

  const makeResampler = (fromRate, toRate) => {
    if (Math.abs(fromRate - toRate) <= 1) {
      return { process: (x) => x };
    }
    const step = fromRate / toRate;
    let prev = 0;
    let havePrev = false;
    let frac = 0;

    return {
      process(input) {
        if (!input.length) return input;
        if (!havePrev) {
          prev = input[0];
          havePrev = true;
        }
        const out = new Float32Array(Math.ceil((input.length + 1) / step) + 2);
        let oi = 0;
        for (let i = 0; i < input.length; i += 1) {
          const curr = input[i];
          while (frac <= 1) {
            out[oi++] = prev + (curr - prev) * frac;
            frac += step;
          }
          frac -= 1;
          prev = curr;
        }
        return out.subarray(0, oi);
      },
    };
  };

  let resampler = makeResampler(effectiveSampleRate, deviceRate);

  const sampleStats = (buffer) => {
    if (!buffer.length) return { min: 0, max: 0 };
    let min = buffer[0];
    let max = buffer[0];
    for (let i = 1; i < buffer.length; i += 1) {
      const value = buffer[i];
      if (value < min) min = value;
      if (value > max) max = value;
    }
    return { min, max };
  };

  const schedule = (samplesF32) => {
    if (!samplesF32.length) return;
    const buffer = ctx.createBuffer(1, samplesF32.length, deviceRate);
    buffer.copyToChannel(samplesF32, 0, 0);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    const now = ctx.currentTime;
    const safety = 0.03;
    const startAt = Math.max(nextStart, now + safety);
    source.start(startAt);
    nextStart = startAt + buffer.duration;

    source.onended = () => {
      activeSources.delete(source);
    };

    activeSources.add(source);

    if (capture) {
      capture.buffers.push(samplesF32.slice());
    }
  };

  const normalizedVoice = voiceId === "None" || voiceId === "null" || voiceId === "undefined" ? null : voiceId;
  const requestBody = {
    text,
    voice_id: normalizedVoice || undefined,
    stream,
  };

  console.log("[TTS-DEBUG] Making TTS request to", endpoint, "with body:", requestBody);
  const doFetch = (path, options) => (typeof fetchApi === "function" ? fetchApi(path, options) : fetch(path, options));

  const response = await doFetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    signal: abort.signal,
    body: JSON.stringify(requestBody),
  });

  console.log("[TTS-DEBUG] TTS response status:", response.status, response.ok ? "OK" : "ERROR");

  if (!response.ok || !response.body) {
    let errorMessage = "TTS stream failed";
    try {
      const errorText = await response.text();
      if (errorText) {
        errorMessage = `TTS server error: ${errorText}`;
        log("Server error response:", errorText);
      }
    } catch (err) {
      log("Could not read server error response:", err);
    }
    await cleanup({ immediate: true });
    throw new Error(errorMessage);
  }

  const reader = response.body.getReader();
  const serverSampleRate = response.headers.get("X-NeuTTS-Sample-Rate");
  if (serverSampleRate && !Number.isNaN(Number(serverSampleRate))) {
    effectiveSampleRate = Number(serverSampleRate);
    needResample = Math.abs(deviceRate - effectiveSampleRate) > 1;
    resampler = makeResampler(effectiveSampleRate, deviceRate);
  }

  let headerBytes = new Uint8Array(0);
  let headerParsed = false;
  let fmt = null;
  let blockAlign = 2;
  let remainder = new Uint8Array(0);

  const finished = (async () => {
    try {
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        if (!value?.length) continue;

        let bytes = value;

        if (!headerParsed) {
          const merged = new Uint8Array(headerBytes.length + bytes.length);
          merged.set(headerBytes, 0);
          merged.set(bytes, headerBytes.length);
          const parsed = parseWavHeader(merged);
          if (!parsed) {
            headerBytes = merged;
            continue;
          }
          headerParsed = true;
          fmt = parsed.fmt;
          blockAlign = parsed.blockAlign;
          bytes = merged.subarray(parsed.dataOffset);
          headerBytes = new Uint8Array(0);
          log("wav fmt", fmt);
        }

        if (remainder.length) {
          const merged = new Uint8Array(remainder.length + bytes.length);
          merged.set(remainder, 0);
          merged.set(bytes, remainder.length);
          bytes = merged;
          remainder = new Uint8Array(0);
        }

        const whole = bytes.byteLength - (bytes.byteLength % blockAlign);
        if (whole === 0) {
          remainder = bytes;
          continue;
        }
        if (whole < bytes.byteLength) {
          remainder = bytes.subarray(whole);
        }

        const samples16 = new Int16Array(bytes.buffer, bytes.byteOffset, whole >> 1);
        const mono = int16leToMonoFloat32(samples16, fmt.numChannels);
        const resampled = needResample ? resampler.process(mono) : mono;
        const stats = sampleStats(resampled);
        state.chunks += 1;
        state.totalSamples += resampled.length;
        log("chunk", {
          chunkIndex: state.chunks,
          srcSamples: mono.length,
          resampled: resampled.length,
          min: Number(stats.min.toFixed(4)),
          max: Number(stats.max.toFixed(4)),
        });
        schedule(resampled);
      }

      if (capture && capture.buffers.length) {
        const total = capture.buffers.reduce((acc, buf) => acc + buf.length, 0);
        const combined = new Float32Array(total);
        let ptr = 0;
        for (const buf of capture.buffers) {
          combined.set(buf, ptr);
          ptr += buf.length;
        }
        GT.__COSMIC_TTS_CAPTURE_FLOATS = combined;
        GT.__COSMIC_TTS_CAPTURE_SAMPLE_RATE = deviceRate;
        GT.__COSMIC_TTS_CAPTURE_BUFFERS = capture.buffers;
      }

      return { aborted: false, totalSamples: state.totalSamples };
    } catch (err) {
      if (err?.name === "AbortError") {
        return { aborted: true, totalSamples: state.totalSamples };
      }
      throw err;
    }
  })();

  const stop = async () => {
    try {
      abort.abort();
    } catch (_) {}
    try {
      await reader.cancel();
    } catch (_) {}
    await cleanup({ immediate: true });
  };

  return { stop, finished, sampleRate: effectiveSampleRate };
}
