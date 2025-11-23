# Custom Voice Registration Guide for NeuTTS

This guide explains how to register your own custom voices with the ONNX decoder configuration.

## Overview

Since the ONNX decoder is **decode-only**, you cannot directly upload WAV files through the Voice Lab interface. Instead, you must:

1. **Encode** your audio file in a separate environment with the full `neucodec` package
2. **Copy** the encoded `.pt` file into the voices directory
3. **Restart** the application to load the new voice

---

## Prerequisites

- Audio file (WAV format recommended)
- 3-15 seconds duration (optimal for voice cloning)
- Clean recording at 16kHz or higher sample rate
- Text transcript of what's spoken in the audio
- Docker installed (recommended) OR a separate Python environment

---

## Method 1: Using Docker (Recommended)

This is the easiest method as it handles all dependencies automatically.

### Step 1: Prepare Your Audio

Place your audio file and its transcript in the samples directory:

```bash
# Create samples directory if it doesn't exist
mkdir -p data/tts/neutts/samples

# Copy your files
cp /path/to/your-voice.wav data/tts/neutts/samples/
cp /path/to/your-voice.txt data/tts/neutts/samples/
```

The `.txt` file should contain the exact text spoken in the audio file.

### Step 2: Run the Encoding Script

Use the provided Docker script with your custom voice:

```bash
# Make the script executable
chmod +x scripts/encode_custom_voice.sh

# Run it
./scripts/encode_custom_voice.sh \
    --audio data/tts/neutts/samples/your-voice.wav \
    --name "Your Voice Name"
```

The script will:
- Build a Docker image with compatible dependencies
- Encode your audio to `.pt` format
- Create the voice directory with all required files
- Place it in `data/tts/neutts/voices/your-voice-name/`

### Step 3: Restart the Application

```bash
./run.sh
```

Your custom voice will now appear in the Voice Lab!

---

## Method 2: Manual Encoding (Advanced)

If you prefer not to use Docker, you can create a separate Python environment.

### Step 1: Create Encoding Environment

```bash
# Create a separate virtual environment
python3 -m venv venv_encode
source venv_encode/bin/activate

# Install encoding dependencies
pip install torch==2.5.1 torchaudio==2.5.1 transformers==4.30.0 neucodec librosa numpy
```

**Important:** Do NOT activate this environment in your main project! It has incompatible dependencies.

### Step 2: Encode Your Audio

```bash
# Still in venv_encode
python scripts/encode_neutts_reference.py \
    --ref_audio /path/to/your-voice.wav \
    --ref_text "The text spoken in the audio" \
    --voice_name "Your Voice Name" \
    --output_dir data/tts/neutts/voices/your-voice-name
```

### Step 3: Deactivate and Restart

```bash
# Exit the encoding environment
deactivate

# Restart your main application
./run.sh
```

---

## Method 3: Batch Encoding Multiple Voices

To encode multiple voices at once, create a batch script:

```bash
#!/bin/bash
# encode_my_voices.sh

./scripts/encode_custom_voice.sh \
    --audio data/tts/neutts/samples/alice.wav \
    --name "Alice"

./scripts/encode_custom_voice.sh \
    --audio data/tts/neutts/samples/bob.wav \
    --name "Bob"

./scripts/encode_custom_voice.sh \
    --audio data/tts/neutts/samples/charlie.wav \
    --name "Charlie"

echo "All voices encoded! Restart the application."
```

---

## Voice Directory Structure

Each voice needs these files:

```
data/tts/neutts/voices/your-voice-name/
├── ref.codes.pt       # Encoded audio codes (PyTorch tensor)
├── ref.txt            # Transcript of reference audio
└── meta.json          # Voice metadata
```

### Example `meta.json`:

```json
{
  "id": "your-voice-name",
  "name": "Your Voice Name",
  "created_at": 1730336400.0,
  "ref_text": "The text spoken in the reference audio",
  "sample_rate": 24000,
  "quality": "q4",
  "watermarked": true
}
```

---

## Tips for Best Results

### Audio Quality
- Use high-quality recordings without background noise
- Avoid music, echo, or multiple speakers
- Clear, consistent speaking style works best

### Duration
- **Minimum:** 3 seconds (shorter may produce unstable results)
- **Optimal:** 5-10 seconds
- **Maximum:** 15 seconds (longer will be trimmed)

### Content
- Natural speech with varied intonation
- Avoid monotone or robotic delivery
- Include the exact transcript for best cloning accuracy

### Recording Settings
- Sample rate: 16kHz or higher (will be resampled automatically)
- Format: WAV, MP3, or other common audio formats
- Mono channel (stereo will be converted to mono)

---

## Troubleshooting

### "Voice not showing up in Voice Lab"

Check that all three files exist:
```bash
ls data/tts/neutts/voices/your-voice-name/
# Should show: ref.codes.pt  ref.txt  meta.json
```

Validate the voice:
```bash
python scripts/validate_voices.py
```

### "Docker build failed"

Ensure Docker Desktop is running:
```bash
docker ps
```

Try pulling the base image first:
```bash
docker pull python:3.10-slim
```

### "Audio sounds wrong or distorted"

- Verify the transcript matches the audio exactly
- Check that the audio file isn't corrupted
- Try re-encoding with a different audio segment

### "Permission denied errors"

Make scripts executable:
```bash
chmod +x scripts/*.sh
```

Or run with bash explicitly:
```bash
bash scripts/encode_custom_voice.sh --audio your-voice.wav --name "Your Voice"
```

---

## Validation

After encoding, validate your voice:

```bash
# Check voice files exist and are valid
python scripts/validate_voices.py

# List all available voices
python -c "from python.runtime.container import get_tts_provider; print(get_tts_provider().list_voices())"
```

---

## Removing Voices

To remove a custom voice:

```bash
# Delete the voice directory
rm -rf data/tts/neutts/voices/your-voice-name

# Restart the application
./run.sh
```

---

## Advanced: Using Pre-Encoded Files

If someone shares a pre-encoded voice with you:

1. Get the three required files:
   - `ref.codes.pt`
   - `ref.txt`
   - `meta.json`

2. Create the voice directory:
   ```bash
   mkdir -p data/tts/neutts/voices/shared-voice
   ```

3. Copy the files:
   ```bash
   cp ref.codes.pt data/tts/neutts/voices/shared-voice/
   cp ref.txt data/tts/neutts/voices/shared-voice/
   cp meta.json data/tts/neutts/voices/shared-voice/
   ```

4. Restart the application

---

## Summary

The workflow is:
1. **Prepare** audio + transcript
2. **Encode** using Docker or separate environment
3. **Restart** application
4. **Use** your custom voice in Voice Lab!

For questions or issues, check the troubleshooting section or the main documentation at `docs/neutts_voice_registration_guide.md`.
