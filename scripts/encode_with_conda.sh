#!/bin/bash
# Encode a custom voice using conda environment

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
    exit 1
fi

echo "🎙️  NeuTTS Custom Voice Encoder (Conda)"
echo "========================================"
echo ""
echo "Voice Name: $VOICE_NAME"
echo "Audio File: $AUDIO_FILE"
echo "Transcript: $TRANSCRIPT_FILE"
echo ""

# Get project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Create output directory
VOICE_ID=$(echo "$VOICE_NAME" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
OUTPUT_DIR="${PROJECT_DIR}/data/tts/neutts/voices/${VOICE_ID}"

# Check if conda environment exists
ENV_NAME="neutts_encode"
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating conda environment: ${ENV_NAME}"
    echo ""
    
    # Create conda environment with Python 3.10
    conda create -n ${ENV_NAME} python=3.10 -y
    
    # Activate and install packages
    echo ""
    echo "Installing packages in conda environment..."
    echo ""
    
    # Use conda run to execute commands in the environment
    conda run -n ${ENV_NAME} pip install torch==2.4.0 torchaudio==2.4.0
    conda run -n ${ENV_NAME} pip install transformers==4.30.0
    conda run -n ${ENV_NAME} pip install librosa
    
    # Try to install neucodec and let it resolve dependencies
    echo ""
    echo "Installing neucodec (may upgrade transformers)..."
    conda run -n ${ENV_NAME} pip install neucodec
    
    echo ""
    echo "✅ Conda environment created"
else
    echo "Using existing conda environment: ${ENV_NAME}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Encoding ${VOICE_NAME}..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run encoding in conda environment
conda run -n ${ENV_NAME} python scripts/encode_neutts_reference.py \
    --ref_audio "${AUDIO_FILE}" \
    --ref_text "${TRANSCRIPT_FILE}" \
    --voice_name "${VOICE_NAME}" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Voice encoded successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Voice directory: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "  1. Restart:  ./run.sh"
echo ""
echo "Your voice will appear in the Voice Lab after restart!"
echo ""
