# Apple Zero - macOS Native Port

A comprehensive AI agent framework running natively on macOS without Docker dependency. This port provides full terminal integration, native PTY management, and seamless macOS compatibility.

## 🎯 Features

- **Native macOS Execution**: Runs directly on macOS without Docker containerization
- **Real Terminal Integration**: Uses native PTY for authentic terminal experience
- **Permission Management**: Smart prompts for sudo and dangerous operations
- **Session Management**: Multiple concurrent terminal sessions with cleanup
- **Web UI**: Modern browser-based interface with terminal settings
- **Voice Support**: Isabella voice integration via Kokoro TTS
- **MCP/A2A Support**: Full compatibility with existing MCP servers and A2A protocol
- **File Operations**: Native file browsing, editing, and management

## 🚀 Quick Start

### Prerequisites

- macOS 10.15 or later
- Python 3.8 or later
- Homebrew (will be installed automatically if not present)

### Installation

1. **Run the setup script:**
   ```bash
   cd APPLE_ZERO_MACOS
   ./dev/macos/setup.sh
   ```

2. **Start the agent:**
   ```bash
   ./dev/macos/run.sh
   ```

3. **Open your browser:**
   ```
   http://localhost:8080
   ```

That's it! The setup script will:
- Install Homebrew if needed
- Install required system dependencies (ffmpeg, portaudio, etc.)
- Create a Python virtual environment
- Install all Python dependencies
- Set up the development environment

## 🛠️ Manual Installation

If you prefer manual setup or encounter issues:

### System Dependencies

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install ffmpeg portaudio poppler tesseract libsndfile coreutils gnu-sed jq wget git

# Install Python dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install Playwright browsers into project cache
PLAYWRIGHT_BROWSERS_PATH=./tmp/playwright playwright install chromium
```

### Environment Setup

Create a `.env` file with your API keys:

```env
# OpenAI API Key (required)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Other LLM providers
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key

# Optional: Search providers
PERPLEXITY_API_KEY=your_perplexity_key

# Optional: Browser automation
PLAYWRIGHT_BROWSERS_PATH=./browsers
```

## 🏗️ Architecture

### Core Components

- **PTY Adapter** (`python/adapters/terminal/macos_pty.py`): Native macOS terminal interface using pexpect/ptyprocess
- **Terminal Manager** (`python/services/terminal_manager.py`): Session lifecycle management and permission gating
- **API Layer** (`python/api/terminal_*.py`): REST endpoints for terminal operations
- **UI Components** (`webui/components/settings/terminal/`): Browser-based terminal settings interface

### Key Changes from Docker Version

1. **Native PTY Integration**: Replaced SSH/Docker with direct PTY spawning
2. **Permission System**: Added sudo/delete command confirmation dialogs
3. **Session Management**: Multi-session support with automatic cleanup
4. **Signal Handling**: Proper zombie process prevention
5. **macOS Optimization**: Native file paths, environment variables, and shell integration

## 🔧 Configuration

### Terminal Settings

Access terminal settings through the web UI:
- **Shell Path**: Choose between `/bin/bash`, `/bin/zsh`, or custom shell
- **Terminal Size**: Adjust rows and columns
- **Environment Variables**: Set custom environment for sessions
- **Permissions**: Configure timeout for sudo prompts
- **Idle Timeout**: Automatic session cleanup after inactivity

### API Endpoints

The system exposes REST API endpoints for terminal operations:

- `POST /api/terminal/start` - Create new terminal session
- `POST /api/terminal/write` - Write to terminal (with permission checking)
- `POST /api/terminal/confirm` - Confirm permission requests
- `GET /api/terminal/stream/{session_id}` - Stream terminal output (SSE)
- `GET /api/terminal/sessions` - List active sessions
- `POST /api/terminal/resize` - Resize terminal window
- `GET /api/terminal/settings` - Get terminal configuration
- `POST /api/terminal/settings` - Update terminal configuration

## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Run all tests
python3 tests/test_terminal_adapter.py
python3 tests/test_terminal_api.py

# Or with pytest (if available)
pytest tests/ -v
```

Tests cover:
- PTY adapter functionality (spawn, read, write, resize)
- Permission system (sudo detection, confirmation flow)
- Session management (create, cleanup, concurrent sessions)
- API integration (REST endpoints, error handling)

## 🔒 Security

### Permission System

The system includes built-in security for dangerous operations:

- **Sudo Commands**: Require explicit user confirmation
- **File Deletion**: `rm -rf`, `rm -f` commands require approval
- **System Paths**: Operations in `/etc/`, `/usr/`, `/System/` require approval
- **Wildcard Operations**: Commands with `*` patterns require approval

### Session Management

- Sessions automatically timeout after inactivity
- Maximum concurrent session limits
- Proper process cleanup prevents resource leaks
- Signal handling prevents zombie processes

## 🎵 Voice Integration

The system uses Kokoro TTS with Isabella voice:

```python
# Voice configuration in python/helpers/kokoro_tts.py
_voice = "bf_isabella"  # Female Isabella voice
_speed = 1.1           # Slightly faster than default
```

To disable voice features, set `ENABLE_TTS=false` in your `.env` file.

## 🔌 MCP Server Integration

Full compatibility with Model Context Protocol servers:

```bash
# Example: Add a new MCP server
# The system will automatically detect and integrate MCP servers
# configured in your MCP configuration files
```

## 📊 Monitoring and Debugging

### Logs

Logs are written to the `logs/` directory:
- `terminal_manager.log` - Session management events
- `pty_adapter.log` - PTY operation logs
- `api.log` - API request/response logs

### Debug Mode

Enable debug logging by setting:

```env
LOG_LEVEL=DEBUG
PYTHONPATH=/path/to/project:$PYTHONPATH
```

## ⚡ Performance

### Optimizations

- **Efficient PTY Management**: Threaded readers for non-blocking I/O
- **Buffer Management**: Circular buffers prevent memory bloat
- **Session Cleanup**: Automatic garbage collection of idle sessions
- **Signal Handling**: Proper process lifecycle management

### Resource Usage

- **Memory**: ~50-100MB base usage + ~5-10MB per active session
- **CPU**: Minimal when idle, scales with terminal activity
- **Disk**: Logs rotate automatically, temp files cleaned on exit

## 🚨 Troubleshooting

### Common Issues

**PTY spawn fails:**
```bash
# Check shell exists
which /bin/bash
# Check permissions
ls -la /bin/bash
```

**Permission errors:**
```bash
# Ensure proper file permissions
chmod +x dev/macos/setup.sh
chmod +x dev/macos/run.sh
```

**Import errors:**
```bash
# Activate virtual environment
source venv/bin/activate
# Reinstall dependencies
pip install -r requirements.txt
```

**Port already in use:**
```bash
# Kill existing processes
lsof -ti:8080 | xargs kill -9
```

### Getting Help

1. Check the logs in `logs/` directory
2. Run tests to verify functionality
3. Ensure all dependencies are installed
4. Check macOS compatibility (10.15+)

## 🤝 Contributing

This is a complete port of Apple Zero for native macOS execution. The architecture maintains compatibility with the original while adding macOS-specific optimizations.

### Development

```bash
# Setup development environment
./dev/macos/setup.sh

# Run in development mode
export FLASK_ENV=development
./dev/macos/run.sh
```

## 📄 License

Same license as the original Apple Zero project.

## 🐞 Recent Bug Fixes (September 2025)

- **Corrected Development Mode Detection:** Resolved an issue where the application would incorrectly enter a development mode on standalone macOS, causing multiple tool failures. The `is_development()` check is now more specific, ensuring stability in native environments.
- **Fixed Search Engine:** Replaced a non-functional search provider with DuckDuckGo to ensure reliable web search capabilities.
- **Stabilized Code Execution:** Corrected the Python interpreter path to use the standard `python` command, resolving `ipython: command not found` errors and ensuring consistent code execution.
- **Unified Application Port:** Standardized the application to run on port `8080` for both the backend and frontend, fixing the "backend is disconnected" error and simplifying the user experience.

## 🎉 Acknowledgments

- Original Apple Zero team for the foundation
- pexpect/ptyprocess developers for PTY management
- Kokoro TTS team for voice synthesis
- macOS terminal emulation standards

## CosmicCrisp Backend

A lightweight FastAPI service implementing streaming agent endpoints.

### Running

```
uv run python -m cosmiccrisp.api.app
# or
python -m cosmiccrisp.api.app
```

### Examples

```
curl -N -X POST localhost:8000/run -H "Content-Type: application/json" -d '{"goal":"Find the latest docs"}'

curl -N -X POST localhost:8000/chat -H "Content-Type: application/json" -d '{"session_id":"s1","message":"hello"}'
```
