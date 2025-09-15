#!/bin/bash

# Install yt-dlp and ffmpeg
sudo apt-get update && sudo apt-get install -y yt-dlp ffmpeg

# Install yt-dlp using pip
pip install --upgrade yt-dlp

# Call the Python script to download the video
#!/usr/bin/env bash
set -euo pipefail

# Run the downloader relative to this script's location, not an absolute /a0 path
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$SCRIPT_DIR/download_video.py" "$1"
