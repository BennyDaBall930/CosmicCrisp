#!/bin/bash
# Docker-based voice encoding for NeuTTS with CORRECT dependency versions

set -e

echo "🐳 NeuTTS Voice Encoding via Docker (Fixed Dependencies)"
echo "=========================================================="
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop:"
    echo "   https://www.docker.com/products/docker-desktop/"
    exit 1
fi

echo "✓ Docker found"
echo ""

# Get project directory
PROJECT_DIR="/Users/benjaminstout/Desktop/CosmicCrisp"
SAMPLES_DIR="${PROJECT_DIR}/data/tts/neutts/samples"
VOICES_DIR="${PROJECT_DIR}/data/tts/neutts/voices"

# Create voices directory if needed
mkdir -p "${VOICES_DIR}"

echo "Building Docker image with compatible dependencies..."
docker build -t neutts-encoder - << 'DOCKERFILE_END'
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install compatible PyTorch/torchaudio versions for neucodec
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    transformers==4.30.0 \
    librosa \
    neucodec \
    numpy

WORKDIR /workspace
DOCKERFILE_END

echo ""
echo "Encoding voices in Docker container..."
echo ""

# Encode Dave
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Encoding Dave..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker run --rm \
    -v "${PROJECT_DIR}/scripts:/scripts:ro" \
    -v "${SAMPLES_DIR}:/samples:ro" \
    -v "${VOICES_DIR}:/output" \
    neutts-encoder \
    python /scripts/encode_neutts_reference.py \
        --ref_audio /samples/dave.wav \
        --voice_name "Dave" \
        --output_dir /output/dave

echo ""
echo ""

# Encode Jo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Encoding Jo..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
docker run --rm \
    -v "${PROJECT_DIR}/scripts:/scripts:ro" \
    -v "${SAMPLES_DIR}:/samples:ro" \
    -v "${VOICES_DIR}:/output" \
    neutts-encoder \
    python /scripts/encode_neutts_reference.py \
        --ref_audio /samples/jo.wav \
        --voice_name "Jo" \
        --output_dir /output/jo

echo ""
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Voices encoded successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Encoded voices are ready at:"
echo "  ${VOICES_DIR}/dave/"
echo "  ${VOICES_DIR}/jo/"
echo ""
echo "Verify with: python scripts/validate_voices.py"
echo "Then restart the application: ./run.sh"
echo ""
