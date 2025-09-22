// webui/js/lib/tts-stream.js
import { fetchApi } from "/js/api.js";

const GT = typeof globalThis !== "undefined" ? globalThis : {};
const DBG = typeof GT.__COSMIC_TTS_DEBUG === "boolean" ? GT.__COSMIC_TTS_DEBUG : true;
const log = (...a) => { if (DBG && console?.debug) console.debug("[tts-stream]", ...a); };

/**
 * Gapless streamed TTS player (no AudioWorklet).
 * - Robust WAV header parsing (no fixed 44-byte assumption).
 * - Byte/frame carryover between reads so we never split 16‑bit frames.
 * - Optional stateful resampling to the device AudioContext rate.
 * - Schedules chunked AudioBufferSourceNode back-to-back with a small safety lead.
 * - Returns { stop, finished }, where finished resolves with { totalSamples, aborted }.
 */
export async function playStreamedTTS({
  text,
  style = {},
  targetChars = 200,
  joinMs = 5,
  sampleRate = 24000,             // server PCM16 stream rate
  firstChunkChars = 0,
  endpoint = "/synthesize_stream" // keep consistent with backend
}) {
  if (!text || !text.trim()) return null;

  // Use device's native rate; browsers may ignore requested rate anyway.
  const ctx = new AudioContext();
  const deviceRate = ctx.sampleRate;
  const needResample = Math.abs(deviceRate - sampleRate) > 1;

  const capture = GT.__COSMIC_TTS_CAPTURE ? { bufs: [], rate: deviceRate } : null;

  if (ctx.state !== "running") await ctx.resume();
  log("AudioContext", { state: ctx.state, deviceRate, streamRate: sampleRate, needResample });

  const active = new Set();
  let nextStart = ctx.currentTime;

  const state = {
    totalSamples: 0,
    chunks: 0,
    cleanupTimer: null,
    closed: false
  };

  const abort = new AbortController();

  const cleanup = async ({ immediate = false } = {}) => {
    if (state.closed) return;
    state.closed = true;
    if (state.cleanupTimer) { clearTimeout(state.cleanupTimer); state.cleanupTimer = null; }
    try {
      active.forEach(s => { try { s.stop(); } catch {} try { s.disconnect(); } catch {} });
      active.clear();
    } catch {}
    try { await ctx.close(); } catch {}
    log("cleanup", { immediate });
  };

  // ---------------- WAV helpers ----------------
  function parseWavHeader(bytes) {
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (dv.byteLength < 12 ||
        dv.getUint32(0, false) !== 0x52494646 || // "RIFF"
        dv.getUint32(8, false) !== 0x57415645) { // "WAVE"
      throw new Error("Not a RIFF/WAVE stream");
    }
    let off = 12;
    let fmt = null;
    let blockAlign = 0;

    while (off + 8 <= dv.byteLength) {
      const id = dv.getUint32(off + 0, false);
      const sz = dv.getUint32(off + 4, true);
      off += 8;

      // "fmt "
      if (id === 0x666d7420) {
        const audioFormat   = dv.getUint16(off + 0, true);
        const numChannels   = dv.getUint16(off + 2, true);
        const sr            = dv.getUint32(off + 4, true);
        const bitsPerSample = dv.getUint16(off + 14, true);
        blockAlign          = dv.getUint16(off + 12, true);
        fmt = { audioFormat, numChannels, sampleRate: sr, bitsPerSample };
      }

      // "data"
      if (id === 0x64617461) {
        if (!fmt) throw new Error("fmt chunk missing before data");
        if (fmt.audioFormat !== 1 || fmt.bitsPerSample !== 16)
          throw new Error(`Unsupported WAV format: format=${fmt.audioFormat}, bits=${fmt.bitsPerSample}`);
        return { dataOffset: off, fmt, blockAlign: blockAlign || ((fmt.numChannels * fmt.bitsPerSample) >> 3) };
      }

      // Chunks are padded to even size.
      off += sz + (sz & 1);
    }
    return null; // header incomplete
  }

  function int16leToMonoFloat32(int16, numChannels) {
    if (numChannels === 1) {
      const out = new Float32Array(int16.length);
      for (let i = 0; i < int16.length; i++) out[i] = Math.max(-1, Math.min(1, int16[i] / 32768));
      return out;
    }
    // Downmix stereo: (L+R)/2
    const frames = (int16.length / numChannels) | 0;
    const out = new Float32Array(frames);
    for (let f = 0, i = 0; f < frames; f++, i += numChannels) {
      const L = int16[i], R = int16[i + 1];
      out[f] = Math.max(-1, Math.min(1, ((L + R) / 2) / 32768));
    }
    return out;
  }

  function makeResampler(fromRate, toRate) {
    if (Math.abs(fromRate - toRate) <= 1) return { process: x => x };
    const step = fromRate / toRate;
    let prev = 0, havePrev = false, frac = 0;
    return {
      process(input) {
        if (!input.length) return input;
        let out = new Float32Array(Math.ceil((input.length + 1) / step) + 2);
        let oi = 0;
        if (!havePrev) { prev = input[0]; havePrev = true; }
        for (let i = 0; i < input.length; i++) {
          const cur = input[i];
          while (frac <= 1) {
            out[oi++] = prev + (cur - prev) * frac;
            frac += step;
          }
          frac -= 1;
          prev = cur;
        }
        return out.subarray(0, oi);
      }
    };
  }
  const resampler = makeResampler(sampleRate, deviceRate);

  function schedule(samplesF32) {
    if (!samplesF32.length) return;
    const buf = ctx.createBuffer(1, samplesF32.length, deviceRate);
    buf.copyToChannel(samplesF32, 0);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);

    const SAFETY = 0.03; // 30ms lead to avoid underruns
    const now = ctx.currentTime;
    const startAt = Math.max(nextStart, now + SAFETY);
    src.start(startAt);
    nextStart = startAt + buf.duration;

    src.onended = () => active.delete(src);
    active.add(src);

    state.totalSamples += samplesF32.length;
    state.chunks += 1;

    if (capture) capture.bufs.push(samplesF32.slice());
  }

  // --------------- fetch & stream ----------------
  const body = JSON.stringify({
    text,
    style,
    target_chars: targetChars,
    join_silence_ms: joinMs,
    first_chunk_chars: firstChunkChars || undefined
  });

  const doFetch = (p, o) => (typeof fetchApi === "function" ? fetchApi(p, o) : fetch(p, o));

  const resp = await doFetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    signal: abort.signal,
    body
  });

  if (!resp.ok || !resp.body) {
    await cleanup({ immediate: true });
    throw new Error("TTS stream failed");
  }

  const reader = resp.body.getReader();

  let headerBuf = new Uint8Array(0);
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

        // Parse header (accumulate until 'data' found)
        if (!headerParsed) {
          const merged = new Uint8Array(headerBuf.length + bytes.length);
          merged.set(headerBuf, 0);
          merged.set(bytes, headerBuf.length);
          const parsed = parseWavHeader(merged);
          if (!parsed) { headerBuf = merged; continue; }
          headerParsed = true;
          fmt = parsed.fmt;
          blockAlign = parsed.blockAlign;
          bytes = merged.subarray(parsed.dataOffset);
          headerBuf = new Uint8Array(0);
          log("WAV fmt", { ch: fmt.numChannels, sr: fmt.sampleRate, bits: fmt.bitsPerSample, blockAlign });
        }

        // Prepend any remainder from the prior read
        if (remainder.length) {
          const merged = new Uint8Array(remainder.length + bytes.length);
          merged.set(remainder, 0);
          merged.set(bytes, remainder.length);
          bytes = merged;
          remainder = new Uint8Array(0);
        }

        // Keep only whole frames; stash tail for next read (NEVER drop)
        const whole = bytes.byteLength - (bytes.byteLength % blockAlign);
        if (whole === 0) { remainder = bytes; continue; }
        if (whole < bytes.byteLength) remainder = bytes.subarray(whole);

        // Convert PCM16LE -> mono float32
        const samples16 = new Int16Array(bytes.buffer, bytes.byteOffset, whole >> 1);
        const monoF32 = int16leToMonoFloat32(samples16, fmt.numChannels);

        // Resample if device rate differs (stateful)
        const outF32 = needResample ? resampler.process(monoF32) : monoF32;

        // Schedule for playback
        schedule(outF32);
      }

      // Schedule close a moment after the last scheduled buffer ends.
      const msRemaining = Math.max(0, (nextStart - ctx.currentTime) * 1000) + 150;
      state.cleanupTimer = setTimeout(() => {
        cleanup({ immediate: false }).catch(() => {});
      }, msRemaining);
      log("finish: scheduled cleanup", { msRemaining, chunks: state.chunks, totalSamples: state.totalSamples });

      // expose capture (optional)
      if (capture && capture.bufs.length) {
        const total = capture.bufs.reduce((n, b) => n + b.length, 0);
        const all = new Float32Array(total);
        let p = 0; for (const b of capture.bufs) { all.set(b, p); p += b.length; }
        GT.__COSMIC_TTS_CAPTURE_FLOATS = all;
        GT.__COSMIC_TTS_CAPTURE_SAMPLE_RATE = deviceRate;
      }

      return { totalSamples: state.totalSamples, aborted: false };
    } catch (err) {
      if (err?.name === "AbortError") {
        await cleanup({ immediate: true });
        return { totalSamples: state.totalSamples, aborted: true };
      }
      await cleanup({ immediate: true });
      throw err;
    }
  })();

  const stop = async () => {
    try { abort.abort(); } catch {}
    try { await reader.cancel(); } catch {}
    await cleanup({ immediate: true });
  };

  return { stop, finished };
}
