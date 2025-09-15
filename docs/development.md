# Apple Zero Development Guide

Apple Zero is an AI agent framework based on the open-source project Agent Zero. This guide walks through setting up a development environment on macOS.

## Prerequisites
- macOS 10.15 or later
- Python 3.8+
- Git

## Step 1: Clone the repository
```bash
git clone https://github.com/your-org/apple-zero.git
cd apple-zero
```

## Step 2: Install dependencies
Use the setup script to install Homebrew packages and Python requirements:
```bash
./dev/macos/setup.sh
```
This script creates a virtual environment in `venv/` and installs all dependencies.

## Step 3: Launch Apple Zero
```bash
./dev/macos/run.sh
```
The web interface will be available at `http://localhost:8080`.

## Step 4: Open in your IDE
Open the project folder in your preferred IDE (VS Code, PyCharm, etc.). Activate the virtual environment from `venv/` and start developing.

### Running Tests
```bash
pytest
```

## Step 5: Debugging
You can run `run_ui.py` directly from your IDE for interactive debugging. Ensure the virtual environment is activated and required environment variables are set.

## Step 6: Contributing
1. Create feature branches locally
2. Commit your changes with descriptive messages
3. Submit a pull request to the main repository

Happy coding! 💻🍎
