#!/usr/bin/env python3
"""
Standalone script to encode NeuTTS reference audio.
Run this in a separate environment with compatible dependencies:
  - transformers==4.30.0 (or compatible version with HubertModel)
  - neucodec
  - torch, torchaudio, librosa

This script saves encoded .pt files that can be used with the ONNX decoder
for inference in the main environment.
"""

import sys
import torch
from pathlib import Path
from librosa import load

def encode_reference(ref_audio_path: str, ref_text: str, output_dir: str, voice_name: str):
    """Encode reference audio to codes for NeuTTS voice cloning."""
    
    print(f"\n=== Encoding {voice_name} ===")
    print(f"Audio: {ref_audio_path}")
    print(f"Text: {ref_text}")
    
    # Import neucodec (requires compatible environment)
    try:
        from neucodec import NeuCodec
    except ImportError as e:
        print("\n❌ ERROR: Failed to import neucodec")
        print("This script requires a compatible environment with:")
        print("  - transformers==4.30.0 (or compatible version)")
        print("  - neucodec")
        print("  - torch, torchaudio, librosa")
        print("\nCreate a separate environment:")
        print("  python -m venv venv_encode")
        print("  source venv_encode/bin/activate")
        print("  pip install transformers==4.30.0 neucodec torch torchaudio librosa")
        sys.exit(1)
    
    # Initialize codec
    print("Loading NeuCodec...")
    codec = NeuCodec.from_pretrained("neuphonic/neucodec")
    codec.eval().to("cpu")
    
    # Load and encode reference audio (must be 16kHz)
    print(f"Loading audio at 16kHz...")
    wav, sr = load(ref_audio_path, sr=16000, mono=True)
    duration = len(wav) / 16000
    print(f"Audio duration: {duration:.1f}s")
    
    if duration < 3.0:
        print(f"⚠️  WARNING: Audio is shorter than recommended 3s minimum")
    if duration > 15.0:
        print(f"⚠️  WARNING: Audio is longer than recommended 15s maximum")
        print(f"Trimming to 15s...")
        wav = wav[:int(15.0 * 16000)]
    
    # Encode to tensor
    print("Encoding audio to codes...")
    wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    
    with torch.no_grad():
        ref_codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
    
    print(f"Generated {ref_codes.shape[0]} codes")
    
    # Save outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save codes as .pt file
    codes_file = output_path / "ref.codes.pt"
    torch.save(ref_codes, codes_file)
    print(f"✅ Saved codes to: {codes_file}")
    
    # Save text file
    text_file = output_path / "ref.txt"
    text_file.write_text(ref_text.strip(), encoding='utf-8')
    print(f"✅ Saved text to: {text_file}")
    
    # Save metadata
    import json
    import time
    meta = {
        "id": voice_name.lower().replace(" ", "_"),
        "name": voice_name,
        "created_at": time.time(),
        "ref_text": ref_text.strip(),
        "sample_rate": 24000,
        "quality": "q4",
        "watermarked": True
    }
    meta_file = output_path / "meta.json"
    meta_file.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f"✅ Saved metadata to: {meta_file}")
    
    print(f"\n✅ {voice_name} encoding complete!")
    print(f"Voice directory: {output_path}")
    

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Encode NeuTTS reference audio in a compatible environment"
    )
    parser.add_argument(
        "--ref_audio",
        type=str,
        required=True,
        help="Path to reference audio file"
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        help="Reference text (or path to .txt file)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for encoded voice"
    )
    parser.add_argument(
        "--voice_name",
        type=str,
        default="Custom Voice",
        help="Name for this voice"
    )
    
    args = parser.parse_args()
    
    # Get reference text
    ref_text = args.ref_text
    if not ref_text:
        # Try to find .txt file with same name as audio
        audio_path = Path(args.ref_audio)
        txt_path = audio_path.with_suffix('.txt')
        if txt_path.exists():
            ref_text = txt_path.read_text(encoding='utf-8').strip()
            print(f"Found reference text file: {txt_path}")
        else:
            print("ERROR: --ref_text required (or provide .txt file with same name as audio)")
            sys.exit(1)
    elif Path(ref_text).exists():
        # ref_text is a path to a file
        ref_text = Path(ref_text).read_text(encoding='utf-8').strip()
    
    # Set output directory
    output_dir = args.output_dir
    if not output_dir:
        # Default to data/tts/neutts/voices/{voice_id}/
        voice_id = args.voice_name.lower().replace(" ", "_")
        output_dir = f"data/tts/neutts/voices/{voice_id}"
    
    encode_reference(
        ref_audio_path=args.ref_audio,
        ref_text=ref_text,
        output_dir=output_dir,
        voice_name=args.voice_name
    )


if __name__ == "__main__":
    main()
