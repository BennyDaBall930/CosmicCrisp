import React from "react";

type TTSSettings = {
  engine: "chatterbox";
  chatterbox: {
    device: "auto" | "mps" | "cuda" | "cpu";
    multilingual: boolean;
    language_id: string;
    exaggeration: number;
    cfg: number;
    audio_prompt_path?: string | null;
    max_chars: number;
    join_silence_ms: number;
  };
};

const LANGS = [
  { id: "en", label: "English" },
  { id: "fr", label: "French" },
  { id: "zh", label: "Chinese" },
  { id: "es", label: "Spanish" },
  { id: "de", label: "German" },
  { id: "ja", label: "Japanese" },
  { id: "ko", label: "Korean" },
  { id: "it", label: "Italian" },
  { id: "pt", label: "Portuguese" },
  { id: "ru", label: "Russian" },
  { id: "ar", label: "Arabic" },
  { id: "hi", label: "Hindi" },
  { id: "nl", label: "Dutch" },
  { id: "sv", label: "Swedish" },
  { id: "da", label: "Danish" },
  { id: "no", label: "Norwegian" },
  { id: "fi", label: "Finnish" },
  { id: "pl", label: "Polish" },
  { id: "cs", label: "Czech" },
  { id: "tr", label: "Turkish" },
  { id: "el", label: "Greek" },
  { id: "th", label: "Thai" },
  { id: "vi", label: "Vietnamese" },
];

export function TTSSettingsPanel(props: {
  value: TTSSettings;
  onChange: (v: TTSSettings) => void;
}) {
  const v = props.value;
  const cb = v.chatterbox;

  const set = (patch: Partial<typeof cb>) =>
    props.onChange({ ...v, chatterbox: { ...cb, ...patch } });

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Text-to-Speech</h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <label className="flex items-center justify-between">
          <span>TTS Engine</span>
          <select
            className="border rounded px-2 py-1"
            value={v.engine}
            onChange={() => {
              /* locked to chatterbox for now */
            }}
          >
            <option value="chatterbox">Chatterbox</option>
          </select>
        </label>

        <label className="flex items-center justify-between">
          <span>Device</span>
          <select
            className="border rounded px-2 py-1"
            value={cb.device ?? "auto"}
            onChange={(e) => set({ device: e.target.value as any })}
          >
            <option value="auto">Auto</option>
            <option value="mps">Apple GPU (MPS)</option>
            <option value="cuda">CUDA</option>
            <option value="cpu">CPU</option>
          </select>
        </label>

        <label className="flex items-center justify-between">
          <span>Multilingual</span>
          <input
            type="checkbox"
            checked={cb.multilingual}
            onChange={(e) => set({ multilingual: e.target.checked })}
          />
        </label>

        <label className="flex items-center justify-between">
          <span>Language</span>
          <select
            className="border rounded px-2 py-1"
            value={cb.language_id}
            onChange={(e) => set({ language_id: e.target.value })}
            disabled={!cb.multilingual}
          >
            {LANGS.map((l) => (
              <option key={l.id} value={l.id}>
                {l.label}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col">
          <span className="mb-1">Emotion (exaggeration)</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={cb.exaggeration}
            onChange={(e) => set({ exaggeration: parseFloat(e.target.value) })}
          />
          <small className="text-gray-500">
            0.5 is default; increase for more intensity (can speed speech).
          </small>
        </label>

        <label className="flex flex-col">
          <span className="mb-1">CFG (style/pacing)</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={cb.cfg}
            onChange={(e) => set({ cfg: parseFloat(e.target.value) })}
          />
          <small className="text-gray-500">
            Lower (~0.3) slows pacing; 0.35 default.
          </small>
        </label>

        <label className="flex flex-col md:col-span-2">
          <span className="mb-1">Reference Voice (wav)</span>
          <input
            type="text"
            placeholder="/path/to/voice.wav"
            value={cb.audio_prompt_path ?? ""}
            onChange={(e) => set({ audio_prompt_path: e.target.value })}
          />
          <small className="text-gray-500">
            Optional zero-shot voice cloning clip (7–20s recommended).
          </small>
        </label>

        <label className="flex items-center justify-between">
          <span>Chunk Size (chars)</span>
          <input
            className="border rounded px-2 py-1 w-32"
            type="number"
            min={140}
            max={2000}
            value={cb.max_chars}
            onChange={(e) => set({ max_chars: parseInt(e.target.value, 10) })}
          />
        </label>

        <label className="flex items-center justify-between">
          <span>Join Gap (ms)</span>
          <input
            className="border rounded px-2 py-1 w-32"
            type="number"
            min={0}
            max={2000}
            value={cb.join_silence_ms}
            onChange={(e) =>
              set({ join_silence_ms: parseInt(e.target.value, 10) })
            }
          />
        </label>
      </div>
    </div>
  );
}
