class PCMPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.readIndex = 0;
    this.port.onmessage = (e) => {
      const msg = e.data;
      if (msg && msg.type === "append" && msg.samples) {
        let f32;
        if (msg.samples instanceof Float32Array) {
          f32 = msg.samples;
        } else if (msg.samples instanceof Int16Array) {
          const s16 = msg.samples;
          f32 = new Float32Array(s16.length);
          for (let i = 0; i < s16.length; i += 1) {
            f32[i] = Math.max(-1, Math.min(1, s16[i] / 32768));
          }
        } else {
          const arr = Array.from(msg.samples || []);
          f32 = new Float32Array(arr.length);
          for (let i = 0; i < arr.length; i += 1) {
            const val = arr[i];
            f32[i] = typeof val === "number" ? Math.max(-1, Math.min(1, val)) : 0;
          }
        }
        if (f32.length) {
          this.queue.push(f32);
        }
      } else if (msg && msg.type === "clear") {
        this.queue = [];
        this.readIndex = 0;
      }
    };
  }

  process(inputs, outputs) {
    const output = outputs[0][0];
    let written = 0;
    while (written < output.length) {
      if (this.queue.length === 0) {
        output.fill(0, written);
        break;
      }
      const current = this.queue[0];
      const need = output.length - written;
      const available = current.length - this.readIndex;
      const take = Math.min(need, available);
      output.set(current.subarray(this.readIndex, this.readIndex + take), written);
      written += take;
      this.readIndex += take;
      if (this.readIndex >= current.length) {
        this.queue.shift();
        this.readIndex = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm-player", PCMPlayerProcessor);
