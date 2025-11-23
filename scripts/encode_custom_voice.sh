#!/bin/bash
# Encode a single custom voice using Docker

set -e

# Parse arguments
AUDIO_FILE=""
VOICE_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --audio)
            AUDIO_FILE="$2"
            shift 2
            ;;
        --name)
            VOICE_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --audio <audio_file> --name <voice_name>"
            exit 1
            ;;
    esac
done

# Validate arguments
if [ -z "$AUDIO_FILE" ] || [ -z "$VOICE_NAME" ]; then
    echo "❌ Missing required arguments"
    echo ""
    echo "Usage: $0 --audio <audio_file> --name <voice_name>"
    echo ""
    echo "Example:"
    echo "  $0 --audio data/tts/neutts/samples/my-voice.wav --name \"My Voice\""
    echo ""
    exit 1
fi

# Check if audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
    echo "❌ Audio file not found: $AUDIO_FILE"
    exit 1
fi

# Check for transcript file
AUDIO_BASE="${AUDIO_FILE%.*}"
TRANSCRIPT_FILE="${AUDIO_BASE}.txt"

if [ ! -f "$TRANSCRIPT_FILE" ]; then
    echo "❌ Transcript file not found: $TRANSCRIPT_FILE"
    echo ""
    echo "Please create a .txt file with the same name as your audio file"
    echo "containing the exact text spoken in the audio."
    echo ""
    exit 1
fi

echo "🎙️  NeuTTS Custom Voice Encoder"
echo "================================"
echo ""
echo "Voice Name: $VOICE_NAME"
echo "Audio File: $AUDIO_FILE"
echo "Transcript: $TRANSCRIPT_FILE"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop:"
    echo "   https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# Get absolute paths
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIO_ABS="$(cd "$(dirname "$AUDIO_FILE")" && pwd)/$(basename "$AUDIO_FILE")"
TRANSCRIPT_ABS="$(cd "$(dirname "$TRANSCRIPT_FILE")" && pwd)/$(basename "$TRANSCRIPT_FILE")"

# Create output directory name
VOICE_ID=$(echo "$VOICE_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
OUTPUT_DIR="${PROJECT_DIR}/data/tts/neutts/voices/${VOICE_ID}"

echo "Building Docker image with compatible dependencies..."
echo ""

docker build -t neutts-encoder - << 'DOCKERFILE_END'
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install compatible PyTorch/torchaudio versions for neucodec
# neucodec needs torch>=2.5.1 and transformers==4.30.0 (for HubertModel)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    transformers==4.30.0 \
    librosa \
    neucodec

WORKDIR /workspace
DOCKERFILE_END

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Encoding ${VOICE_NAME}..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run encoding in Docker - pass transcript file path, not contents
docker run --rm \
    -v "${PROJECT_DIR}/scripts:/scripts:ro" \
    -v "$(dirname "$AUDIO_ABS"):/audio:ro" \
    -v "${OUTPUT_DIR}:/output" \
    neutts-encoder \
    python /scripts/encode_neutts_reference.py \
        --ref_audio "/audio/$(basename "$AUDIO_ABS")" \
        --ref_text "/audio/$(basename "$TRANSCRIPT_ABS")" \
        --voice_name "$VOICE_NAME" \
        --output_dir /output

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Voice encoded successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Voice directory: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Validate: python scripts/validate_voices.py"
echo "  2. Restart:  ./run.sh"
echo ""
echo "Your voice will appear in the Voice Lab after restart!"
echo ""
