from huggingface_hub import snapshot_download
import os

target_dir = "data/tts/supertonic"
os.makedirs(target_dir, exist_ok=True)

print(f"Downloading Supertone/supertonic to {target_dir}...")
snapshot_download(repo_id="Supertone/supertonic", local_dir=target_dir)
print("Download complete.")
