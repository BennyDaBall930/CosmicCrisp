## Environment
macOS on Apple Silicon running the Apple Zero runtime natively.
workspace root is the CosmicCrisp repository; operate inside this tree.
default shell is bash/zsh without sudo; request approval before any privileged or destructive command.
prefer relative paths anchored at the project root; stage files under `tmp/` or relevant package folders.
python 3.11 virtual environment lives in `venv/`; node, playwright, ffmpeg, and other tooling are installed locally.
use quoted heredocs for multi-line writes and verify filesystem changes before reporting success.
