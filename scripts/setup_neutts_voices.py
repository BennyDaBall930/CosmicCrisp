#!/usr/bin/env python3
"""Setup default NeuTTS voices (Dave and Jo) from sample audio files."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from python.runtime.audio.neutts_provider import NeuttsProvider

def setup_voices():
    """Register Dave and Jo voices from sample files."""
    print("Setting up NeuTTS voices...")
    
    # Initialize provider with full neucodec (can encode and decode)
    provider = NeuttsProvider(
        backbone_repo="neuphonic/neutts-air-q4-gguf",
        codec_repo="neuphonic/neucodec",
        backbone_device="mps",
        codec_device="cpu",
    )
    
    samples_dir = project_root / "data" / "tts" / "neutts" / "samples"
    
    # Register Dave
    dave_wav = samples_dir / "dave.wav"
    dave_txt = samples_dir / "dave.txt"
    
    if dave_wav.exists() and dave_txt.exists():
        print(f"\nRegistering Dave voice from {dave_wav}...")
        dave_text = dave_txt.read_text(encoding="utf-8").strip()
        print(f"Reference text: {dave_text}")
        try:
            dave_id = provider.register_voice("Dave", str(dave_wav), dave_text)
            print(f"✅ Dave voice registered with ID: {dave_id}")
        except Exception as e:
            print(f"❌ Failed to register Dave: {e}")
    else:
        print(f"⚠️  Dave samples not found at {samples_dir}")
    
    # Register Jo
    jo_wav = samples_dir / "jo.wav"
    jo_txt = samples_dir / "jo.txt"
    
    if jo_wav.exists() and jo_txt.exists():
        print(f"\nRegistering Jo voice from {jo_wav}...")
        jo_text = jo_txt.read_text(encoding="utf-8").strip()
        print(f"Reference text: {jo_text}")
        try:
            jo_id = provider.register_voice("Jo", str(jo_wav), jo_text)
            print(f"✅ Jo voice registered with ID: {jo_id}")
            
            # Set Jo as default
            provider.set_default_voice(jo_id)
            print(f"✅ Set Jo as default voice")
        except Exception as e:
            print(f"❌ Failed to register Jo: {e}")
    else:
        print(f"⚠️  Jo samples not found at {samples_dir}")
    
    # List registered voices
    voices = provider.list_voices()
    print(f"\n📋 Total registered voices: {len(voices)}")
    for voice in voices:
        print(f"  - {voice['name']} (ID: {voice['id']})")

if __name__ == "__main__":
    setup_voices()
