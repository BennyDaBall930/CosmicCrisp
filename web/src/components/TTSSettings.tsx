import React from "react";

type NeuTTSConfig = {
  backbone_repo: string;
  codec_repo: string;
  backbone_device: string;
  codec_device: string;
  model_cache_dir: string;
  quality_default: "q4" | "q8";
  stream_chunk_seconds: number;
};

type TTSSettings = {
  provider: "neutts";
  sample_rate: number;
  stream_default: boolean;
  neutts: NeuTTSConfig;
};

export function TTSSettingsPanel(props: {
  value: TTSSettings;
  onChange: (v: TTSSettings) => void;
}) {
  const v = props.value;
  const neutts = v.neutts;

  const update = (patch: Partial<TTSSettings>) => props.onChange({ ...v, ...patch });
  const updateNeuTTS = (patch: Partial<NeuTTSConfig>) => update({ neutts: { ...neutts, ...patch } });

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">NeuTTS-Air</h3>
      <p className="text-sm text-gray-500">
        NeuTTS-Air runs locally with Metal acceleration and watermarked outputs. Configure defaults below and manage custom voices from the Voice Lab panel.
      </p>

      <label className="flex items-center justify-between">
        <span>Streaming (default)</span>
        <input
          type="checkbox"
          checked={v.stream_default}
          onChange={(e) => update({ stream_default: e.target.checked })}
        />
      </label>

      <label className="flex items-center justify-between">
        <span>Sample rate (Hz)</span>
        <input
          type="number"
          className="border rounded px-2 py-1 w-32 text-right"
          value={v.sample_rate}
          onChange={(e) => update({ sample_rate: Number(e.target.value) || 24000 })}
          min={8000}
          step={1000}
        />
      </label>

      <label className="flex items-center justify-between">
        <span>Quality</span>
        <select
          className="border rounded px-2 py-1"
          value={neutts.quality_default}
          onChange={(e) => updateNeuTTS({ quality_default: e.target.value as NeuTTSConfig["quality_default"] })}
        >
          <option value="q4">Q4 (fast)</option>
          <option value="q8">Q8 (high fidelity)</option>
        </select>
      </label>

      <div className="space-y-2">
        <label className="flex flex-col">
          <span className="mb-1">Backbone repository</span>
          <input
            className="border rounded px-2 py-1"
            value={neutts.backbone_repo}
            onChange={(e) => updateNeuTTS({ backbone_repo: e.target.value })}
          />
        </label>
        <label className="flex flex-col">
          <span className="mb-1">Codec repository</span>
          <input
            className="border rounded px-2 py-1"
            value={neutts.codec_repo}
            onChange={(e) => updateNeuTTS({ codec_repo: e.target.value })}
          />
        </label>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <label className="flex flex-col">
          <span className="mb-1">Backbone device</span>
          <input
            className="border rounded px-2 py-1"
            value={neutts.backbone_device}
            onChange={(e) => updateNeuTTS({ backbone_device: e.target.value })}
          />
        </label>
        <label className="flex flex-col">
          <span className="mb-1">Codec device</span>
          <input
            className="border rounded px-2 py-1"
            value={neutts.codec_device}
            onChange={(e) => updateNeuTTS({ codec_device: e.target.value })}
          />
        </label>
      </div>

      <label className="flex flex-col">
        <span className="mb-1">Model cache directory</span>
        <input
          className="border rounded px-2 py-1"
          value={neutts.model_cache_dir}
          onChange={(e) => updateNeuTTS({ model_cache_dir: e.target.value })}
        />
      </label>

      <label className="flex flex-col">
        <span className="mb-1">Stream chunk seconds</span>
        <input
          type="number"
          className="border rounded px-2 py-1 w-32 text-right"
          value={neutts.stream_chunk_seconds}
          onChange={(e) =>
            updateNeuTTS({ stream_chunk_seconds: Number(e.target.value) || 0.32 })
          }
          min={0.1}
          max={1.0}
          step={0.02}
        />
      </label>
    </div>
  );
}
