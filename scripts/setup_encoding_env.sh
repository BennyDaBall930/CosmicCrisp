#!/bin/bash
# Setup script for NeuTTS encoding environment
# This creates a minimal environment just for encoding reference audio

set -e

echo "🔧 Setting up NeuTTS encoding environment..."
echo ""

# Check if venv_encode already exists
if [ -d "venv_encode" ]; then
    echo "⚠️  venv_encode directory already exists"
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing venv_encode..."
        rm -rf venv_encode
    else
        echo "Keeping existing environment. Exiting."
        exit 0
    fi
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv_encode

# Activate it
source venv_encode/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install compatible dependencies
echo ""
echo "Installing dependencies..."
echo "  - transformers==4.30.0 (compatible with HubertModel)"
echo "  - neucodec (voice encoder)"
echo "  - torch, torchaudio, librosa (audio processing)"
echo ""

pip install \
    transformers==4.30.0 \
    neucodec \
    torch \
    torchaudio \
    librosa \
    numpy

echo ""
echo "✅ Encoding environment setup complete!"
echo ""
echo "To use this environment:"
echo "  source venv_encode/bin/activate"
echo ""
echo "To encode voices:"
echo "  python scripts/encode_neutts_reference.py --ref_audio your_audio.wav --voice_name \"Your Name\""
echo ""
echo "When done, deactivate with:"
echo "  deactivate"
