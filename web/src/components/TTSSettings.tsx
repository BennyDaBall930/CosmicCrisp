import React from "react";

type XTTSSettings = {
  model_id: string;
  device: "auto" | "cuda" | "mps" | "cpu";
  language: string;
  speaker?: string | null;
  speaker_wav_path?: string | null;
  sample_rate: number;
  join_silence_ms: number;
  max_chars: number;
};

type TTSSettings = {
  engine: "chatterbox" | "xtts" | "browser";
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
  xtts: XTTSSettings;
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
  const xtts = v.xtts;

  const setChatterbox = (patch: Partial<typeof cb>) =>
    props.onChange({ ...v, chatterbox: { ...cb, ...patch } });

  const setXTTS = (patch: Partial<typeof xtts>) =>
    props.onChange({ ...v, xtts: { ...xtts, ...patch } });

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold">Text-to-Speech</h3>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <label className="flex items-center justify-between">
          <span>TTS Engine</span>
          <select
            className="border rounded px-2 py-1"
            value={v.engine}
            onChange={(event) =>
              props.onChange({ ...v, engine: event.target.value as TTSSettings["engine"] })
            }
          >
            <option value="chatterbox">Chatterbox</option>
            <option value="xtts">Coqui XTTS</option>
            <option value="browser">Browser</option>
          </select>
        </label>

        {v.engine === "chatterbox" ? (
          <label className="flex items-center justify-between">
            <span>Device</span>
            <select
              className="border rounded px-2 py-1"
              value={cb.device ?? "auto"}
              onChange={(e) => setChatterbox({ device: e.target.value as any })}
            >
              <option value="auto">Auto</option>
              <option value="mps">Apple GPU (MPS)</option>
              <option value="cuda">CUDA</option>
              <option value="cpu">CPU</option>
            </select>
          </label>
        ) : v.engine === "xtts" ? (
          <label className="flex items-center justify-between">
            <span>Device</span>
            <select
              className="border rounded px-2 py-1"
              value={xtts.device ?? "auto"}
              onChange={(e) => setXTTS({ device: e.target.value as any })}
            >
              <option value="auto">Auto</option>
              <option value="cuda">CUDA</option>
              <option value="mps">Apple GPU (MPS)</option>
              <option value="cpu">CPU</option>
            </select>
          </label>
        ) : null}

        {v.engine === "chatterbox" ? (
          <label className="flex items-center justify-between">
            <span>Multilingual</span>
            <input
              type="checkbox"
              checked={cb.multilingual}
              onChange={(e) => setChatterbox({ multilingual: e.target.checked })}
            />
          </label>
        ) : null}

        {v.engine === "chatterbox" ? (
          <label className="flex items-center justify-between">
            <span>Language</span>
            <select
              className="border rounded px-2 py-1"
              value={cb.language_id}
              onChange={(e) => setChatterbox({ language_id: e.target.value })}
              disabled={!cb.multilingual}
            >
              {LANGS.map((l) => (
                <option key={l.id} value={l.id}>
                  {l.label}
                </option>
              ))}
            </select>
          </label>
        ) : null}

        {v.engine === "chatterbox" ? (
          <label className="flex flex-col">
            <span className="mb-1">Emotion (exaggeration)</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={cb.exaggeration}
              onChange={(e) =>
                setChatterbox({ exaggeration: parseFloat(e.target.value) })
              }
            />
            <small className="text-gray-500">
              0.5 is default; increase for more intensity (can speed speech).
            </small>
          </label>
        ) : null}

        {v.engine === "chatterbox" ? (
          <label className="flex flex-col">
            <span className="mb-1">CFG (style/pacing)</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={cb.cfg}
              onChange={(e) => setChatterbox({ cfg: parseFloat(e.target.value) })}
            />
            <small className="text-gray-500">
              Lower (~0.3) slows pacing; 0.35 default.
            </small>
          </label>
        ) : null}

        {v.engine === "chatterbox" ? (
          <label className="flex flex-col md:col-span-2">
            <span className="mb-1">Reference Voice (wav)</span>
            <input
              type="text"
              placeholder="/path/to/voice.wav"
              value={cb.audio_prompt_path ?? ""}
              onChange={(e) => setChatterbox({ audio_prompt_path: e.target.value })}
            />
            <small className="text-gray-500">
              Optional zero-shot voice cloning clip (7–20s recommended).
            </small>
          </label>
        ) : null}

        {v.engine === "chatterbox" ? (
          <label className="flex items-center justify-between">
            <span>Chunk Size (chars)</span>
            <input
              className="border rounded px-2 py-1 w-32"
              type="number"
              min={140}
              max={2000}
              value={cb.max_chars}
              onChange={(e) =>
                setChatterbox({ max_chars: parseInt(e.target.value, 10) })
              }
            />
          </label>
        ) : null}

        {v.engine === "chatterbox" ? (
          <label className="flex items-center justify-between">
            <span>Join Gap (ms)</span>
            <input
              className="border rounded px-2 py-1 w-32"
              type="number"
              min={0}
              max={2000}
              value={cb.join_silence_ms}
              onChange={(e) =>
                setChatterbox({ join_silence_ms: parseInt(e.target.value, 10) })
              }
            />
          </label>
        ) : null}

        {v.engine === "xtts" ? (
          <>
            <label className="flex flex-col md:col-span-2">
              <span className="mb-1">Model (Coqui repo or path)</span>
              <input
                type="text"
                placeholder="tts_models/multilingual/multi-dataset/xtts_v2"
                value={xtts.model_id}
                onChange={(e) => setXTTS({ model_id: e.target.value })}
              />
            </label>

            <label className="flex items-center justify-between">
              <span>Language</span>
              <input
                className="border rounded px-2 py-1 w-32"
                type="text"
                value={xtts.language}
                onChange={(e) => setXTTS({ language: e.target.value })}
              />
            </label>

            <label className="flex items-center justify-between">
              <span>Speaker preset</span>
              <input
                className="border rounded px-2 py-1"
                type="text"
                placeholder="female-en-5"
                value={xtts.speaker ?? ""}
                onChange={(e) => setXTTS({ speaker: e.target.value })}
              />
            </label>

            <label className="flex flex-col md:col-span-2">
              <span className="mb-1">Reference voice (wav)</span>
              <input
                type="text"
                placeholder="/path/to/speaker.wav"
                value={xtts.speaker_wav_path ?? ""}
                onChange={(e) => setXTTS({ speaker_wav_path: e.target.value })}
              />
              <small className="text-gray-500">
                Optional 5–10 second clip to clone a custom voice.
              </small>
            </label>

            <label className="flex items-center justify-between">
              <span>Sample rate</span>
              <input
                className="border rounded px-2 py-1 w-24"
                type="number"
                min={8000}
                value={xtts.sample_rate}
                onChange={(e) => setXTTS({ sample_rate: parseInt(e.target.value, 10) })}
              />
            </label>

            <label className="flex items-center justify-between">
              <span>Chunk Size (chars)</span>
              <input
                className="border rounded px-2 py-1 w-24"
                type="number"
                min={100}
                value={xtts.max_chars}
                onChange={(e) => setXTTS({ max_chars: parseInt(e.target.value, 10) })}
              />
            </label>

            <label className="flex items-center justify-between">
              <span>Join Gap (ms)</span>
              <input
                className="border rounded px-2 py-1 w-24"
                type="number"
                min={0}
                value={xtts.join_silence_ms}
                onChange={(e) => setXTTS({ join_silence_ms: parseInt(e.target.value, 10) })}
              />
            </label>
          </>
        ) : null}

        {v.engine === "browser" ? (
          <div className="md:col-span-2 text-sm text-gray-500">
            Browser speech synthesis uses your system voices. Configure voices in
            your browser or operating system preferences.
          </div>
        ) : null}
      </div>
    </div>
  );
}
