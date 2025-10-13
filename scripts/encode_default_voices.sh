#!/bin/bash
# Batch encode the default NeuTTS voices (Dave and Jo)
# Run this in the encoding environment: source venv_encode/bin/activate

set -e

echo "🎤 Encoding default NeuTTS voices..."
echo ""

# Check if we're in the right environment
if ! python -c "import neucodec" 2>/dev/null; then
    echo "❌ ERROR: neucodec not found"
    echo ""
    echo "You need to run this in the encoding environment:"
    echo "  source venv_encode/bin/activate"
    echo ""
    echo "If you haven't set it up yet:"
    echo "  bash scripts/setup_encoding_env.sh"
    exit 1
fi

# Check if sample files exist
SAMPLES_DIR="data/tts/neutts/samples"
if [ ! -f "$SAMPLES_DIR/dave.wav" ] || [ ! -f "$SAMPLES_DIR/jo.wav" ]; then
    echo "❌ ERROR: Sample files not found in $SAMPLES_DIR"
    echo ""
    echo "Expected files:"
    echo "  - $SAMPLES_DIR/dave.wav"
    echo "  - $SAMPLES_DIR/dave.txt"
    echo "  - $SAMPLES_DIR/jo.wav"
    echo "  - $SAMPLES_DIR/jo.txt"
    exit 1
fi

echo "Found sample files ✓"
echo ""

# Encode Dave
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Encoding Dave..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/encode_neutts_reference.py \
    --ref_audio "$SAMPLES_DIR/dave.wav" \
    --voice_name "Dave"

if [ $? -eq 0 ]; then
    echo "✅ Dave encoded successfully"
else
    echo "❌ Dave encoding failed"
    exit 1
fi

echo ""
echo ""

# Encode Jo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Encoding Jo..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python scripts/encode_neutts_reference.py \
    --ref_audio "$SAMPLES_DIR/jo.wav" \
    --voice_name "Jo"

if [ $? -eq 0 ]; then
    echo "✅ Jo encoded successfully"
else
    echo "❌ Jo encoding failed"
    exit 1
fi

echo ""
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All voices encoded successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Encoded voices are ready at:"
echo "  data/tts/neutts/voices/dave/"
echo "  data/tts/neutts/voices/jo/"
echo ""
echo "Next steps:"
echo "  1. Deactivate encoding environment: deactivate"
echo "  2. Activate main environment: source venv/bin/activate"
echo "  3. Start the application - voices will be available!"
echo ""
