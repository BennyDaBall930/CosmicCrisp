# Users installation guide for macOS

Apple Zero is an AI agent framework based on the open-source project Agent Zero. This guide explains how to install and run Apple Zero natively on macOS.

## macOS Setup

### Quick Setup Script
1. Clone the repository and enter the project directory.
2. Run the setup script:
   ```bash
   ./dev/macos/setup.sh
   ```
3. Start the agent:
   ```bash
   ./dev/macos/run.sh
   ```
4. Open your browser and navigate to `http://localhost:8080`.

### Manual Installation
If you prefer manual setup or encounter issues:

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install ffmpeg portaudio poppler tesseract libsndfile coreutils gnu-sed jq wget git

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install
```

## Settings Configuration
Create a `.env` file with your API keys.

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Additional keys for other providers can be added as needed. Start the UI with `./dev/macos/run.sh` and configure settings through the web interface.

## Choosing Your LLMs
In the settings UI select the desired provider for the chat, utility and embedding models. You can mix and match providers based on your API keys.

## Installing and Using Ollama (Local Models)
1. Install Ollama via Homebrew:
   ```bash
   brew install ollama
   ```
2. Start the Ollama server with `ollama serve`.
3. Download models, e.g. `ollama pull llama3.2`.
4. In Apple Zero settings choose **Ollama** as provider and set the base URL to `http://localhost:11434`.

## Using Apple Zero on your mobile device
When Apple Zero runs on your Mac it listens on `http://localhost:8080`.

- Local network access: `http://<YOUR_MAC_IP>:8080`
- Optional: enable a tunnel in settings to expose your instance on the internet. Remember to set username and password for security.

## How to update Apple Zero
1. Pull the latest changes:
   ```bash
   git pull
   ```
2. Rerun the setup script to install new dependencies:
   ```bash
   ./dev/macos/setup.sh
   ```
3. Restart Apple Zero with `./dev/macos/run.sh`.

## In-Depth Guide for Full Binaries Installation
For advanced customization refer to the project README and explore the scripts under `dev/macos/`.

### Conclusion
After following these instructions you should have Apple Zero running natively on your Mac. Explore the framework and create your own agents!
