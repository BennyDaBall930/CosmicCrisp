#!/usr/bin/env python3
"""
Validation utility for NeuTTS encoded voices.
Checks that voice directories have all required files and correct formats.
"""

import sys
import json
from pathlib import Path
import torch
import numpy as np

def validate_voice(voice_dir: Path) -> tuple[bool, list[str]]:
    """
    Validate a voice directory.
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # Check directory exists
    if not voice_dir.exists():
        return False, [f"Directory does not exist: {voice_dir}"]
    
    # Check meta.json
    meta_path = voice_dir / "meta.json"
    if not meta_path.exists():
        errors.append("Missing meta.json")
    else:
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            required_fields = ["id", "name", "created_at", "ref_text"]
            missing = [f for f in required_fields if f not in meta]
            if missing:
                errors.append(f"meta.json missing fields: {', '.join(missing)}")
        except Exception as e:
            errors.append(f"Invalid meta.json: {e}")
    
    # Check ref.txt
    txt_path = voice_dir / "ref.txt"
    if not txt_path.exists():
        errors.append("Missing ref.txt")
    else:
        try:
            text = txt_path.read_text(encoding='utf-8').strip()
            if not text:
                errors.append("ref.txt is empty")
        except Exception as e:
            errors.append(f"Cannot read ref.txt: {e}")
    
    # Check for codes files
    pt_path = voice_dir / "ref.codes.pt"
    npy_path = voice_dir / "ref.codes.npy"
    
    if not pt_path.exists() and not npy_path.exists():
        errors.append("Missing codes file (need ref.codes.pt or ref.codes.npy)")
    else:
        # Validate .pt file if it exists
        if pt_path.exists():
            try:
                codes = torch.load(pt_path, map_location='cpu')
                if not isinstance(codes, torch.Tensor):
                    errors.append("ref.codes.pt does not contain a tensor")
                elif codes.dim() != 1:
                    errors.append(f"ref.codes.pt has wrong shape: {codes.shape}, expected 1D")
                elif codes.numel() == 0:
                    errors.append("ref.codes.pt is empty")
                else:
                    # Valid!
                    pass
            except Exception as e:
                errors.append(f"Cannot load ref.codes.pt: {e}")
        
        # Validate .npy file if it exists
        if npy_path.exists():
            try:
                codes = np.load(npy_path)
                if codes.ndim != 1:
                    errors.append(f"ref.codes.npy has wrong shape: {codes.shape}, expected 1D")
                elif codes.size == 0:
                    errors.append("ref.codes.npy is empty")
            except Exception as e:
                errors.append(f"Cannot load ref.codes.npy: {e}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def main():
    voices_dir = Path("data/tts/neutts/voices")
    
    if not voices_dir.exists():
        print(f"❌ Voices directory not found: {voices_dir}")
        print("No voices have been registered yet.")
        return
    
    # Find all voice directories
    voice_dirs = [d for d in voices_dir.iterdir() if d.is_dir()]
    
    if not voice_dirs:
        print(f"No voice directories found in {voices_dir}")
        return
    
    print(f"🔍 Validating {len(voice_dirs)} voice(s)...\n")
    
    all_valid = True
    for voice_dir in sorted(voice_dirs):
        voice_id = voice_dir.name
        
        # Try to load metadata for voice name
        meta_path = voice_dir / "meta.json"
        voice_name = voice_id
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
                voice_name = meta.get('name', voice_id)
            except:
                pass
        
        is_valid, errors = validate_voice(voice_dir)
        
        if is_valid:
            print(f"✅ {voice_name} ({voice_id})")
            
            # Show additional info
            pt_path = voice_dir / "ref.codes.pt"
            npy_path = voice_dir / "ref.codes.npy"
            
            if pt_path.exists():
                codes = torch.load(pt_path, map_location='cpu')
                print(f"   Format: PyTorch (.pt)")
                print(f"   Codes: {codes.numel()} tokens")
            elif npy_path.exists():
                codes = np.load(npy_path)
                print(f"   Format: NumPy (.npy)")
                print(f"   Codes: {codes.size} tokens")
            
            # Show reference text
            txt_path = voice_dir / "ref.txt"
            if txt_path.exists():
                text = txt_path.read_text(encoding='utf-8').strip()
                preview = text[:60] + "..." if len(text) > 60 else text
                print(f"   Text: \"{preview}\"")
            
            print()
        else:
            all_valid = False
            print(f"❌ {voice_name} ({voice_id})")
            for error in errors:
                print(f"   - {error}")
            print()
    
    if all_valid:
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("✅ All voices are valid!")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    else:
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("⚠️  Some voices have errors")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        sys.exit(1)


if __name__ == "__main__":
    main()
