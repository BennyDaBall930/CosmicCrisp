#!/bin/bash
# Docker-based voice encoding for NeuTTS
# This bypasses local dependency conflicts by using a controlled Docker environment

set -e

echo "🐳 NeuTTS Voice Encoding via Docker"
echo "===================================="
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop:"
    echo "   https://www.docker.com/products/docker-desktop/"
    exit 1
fi

echo "✓ Docker found"
echo ""

# Create Dockerfile for encoding
cat > /tmp/neutts_encoder.Dockerfile <<'EOF'
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers==4.30.0 \
    torch==2.0.0 \
    torchaudio==2.0.0 \
    librosa \
    neucodec \
    numpy

WORKDIR /workspace
EOF

echo "Building Docker image for voice encoding..."
docker build -t neutts-encoder -f /tmp/neutts_encoder.Dockerfile /tmp

echo ""
echo "Encoding voices in Docker container..."
echo ""

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMPLES_DIR="${PROJECT_DIR}/data/tts/neutts/samples"
VOICES_DIR="${PROJECT_DIR}/data/tts/neutts/voices"

# Create voices directory if needed
mkdir -p "${VOICES_DIR}"

# Run encoding in Docker
docker run --rm \
    -v "${PROJECT_DIR}/scripts:/scripts:ro" \
    -v "${SAMPLES_DIR}:/samples:ro" \
    -v "${VOICES_DIR}:/output" \
    neutts-encoder \
    bash -c '
        cd /workspace
        
        echo "Encoding Dave..."
        python /scripts/encode_neutts_reference.py \
            --ref_audio /samples/dave.wav \
            --voice_name "Dave" \
            --output_dir /output/dave
        
        echo ""
        echo "Encoding Jo..."
        python /scripts/encode_neutts_reference.py \
            --ref_audio /samples/jo.wav \
            --voice_name "Jo" \
            --output_dir /output/jo
        
        echo ""
        echo "✅ Encoding complete!"
    '

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Voices encoded successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Encoded voices are ready at:"
echo "  ${VOICES_DIR}/dave/"
echo "  ${VOICES_DIR}/jo/"
echo ""
echo "You can now run the application:"
echo "  ./run.sh"
echo ""
