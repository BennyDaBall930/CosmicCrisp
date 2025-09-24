#!/bin/bash

# Apple Zero macOS Setup Script
# Prepares the environment for native macOS execution

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Apple Zero - macOS Native Setup${NC}"
echo "======================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Change to project directory
cd "${PROJECT_DIR}"

echo "Project directory: ${PROJECT_DIR}"

# --- Step 1: Check for Homebrew ---
echo -e "\n${GREEN}Step 1: Checking for Homebrew...${NC}"
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}Homebrew not found. Installing...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew is already installed."
fi

# --- Step 2: Install System Dependencies ---
echo -e "\n${GREEN}Step 2: Installing system dependencies with Homebrew...${NC}"
# Prefer FFmpeg 6 so torchaudio can locate the expected libav* dylibs.
brew install ffmpeg@6 || brew upgrade ffmpeg@6
brew install portaudio poppler tesseract libsndfile coreutils gnu-sed jq wget git tmux

FFMPEG6_PREFIX="/opt/homebrew/opt/ffmpeg@6"
if [ -d "${FFMPEG6_PREFIX}/lib" ]; then
    echo "Configuring environment variables for FFmpeg@6 runtime…"
    # Ensure the dylibs are discoverable during future launches (DYLD_* read at exec time).
    export DYLD_LIBRARY_PATH="${FFMPEG6_PREFIX}/lib:${DYLD_LIBRARY_PATH}"
    export DYLD_FALLBACK_LIBRARY_PATH="${FFMPEG6_PREFIX}/lib:${DYLD_FALLBACK_LIBRARY_PATH}"
    export LDFLAGS="-L${FFMPEG6_PREFIX}/lib ${LDFLAGS}"
    export CPPFLAGS="-I${FFMPEG6_PREFIX}/include ${CPPFLAGS}"
    export PKG_CONFIG_PATH="${FFMPEG6_PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH}"
    export COSMIC_FFMPEG_LIB_DIR="${FFMPEG6_PREFIX}/lib"
fi

# --- Step 3: Setup Python Virtual Environment ---
echo -e "\n${GREEN}Step 3: Setting up Python virtual environment...${NC}"

# Check for Python 3.11
echo "Checking for Python 3.11..."
if ! command -v python3.11 &> /dev/null; then
    echo -e "${RED}Error: Python 3.11 is not installed.${NC}"
    echo "Please install it with Homebrew:"
    echo -e "${YELLOW}brew install python@3.11${NC}"
    exit 1
fi
echo "Python 3.11 found."

# Remove existing virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Removing existing virtual environment...${NC}"
    rm -rf venv
fi

echo "Creating new virtual environment with Python 3.11..."
python3.11 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

# Load environment variables from .env file
if [ -f ".env" ]; then
    echo "Loading .env file..."
    set -a
    source .env
    set +a
fi

# --- Step 4: Install Python Dependencies ---
echo -e "\n${GREEN}Step 4: Installing Python dependencies...${NC}"
export PIP_NO_BUILD_ISOLATION=1
pip install --upgrade pip
pip install "numpy<2" wheel
echo "Installing root requirements..."
pip install -r requirements.txt

# Install SearXNG dependencies up front to avoid runtime installs
if [ -f "searxng/requirements.txt" ]; then
    echo "Installing SearXNG requirements..."
    pip install -r searxng/requirements.txt -r searxng/requirements-server.txt
fi

# --- Step 5: Setup Mem0 with Apple Zero memories ---
echo -e "\n${GREEN}Step 5: Setting up Mem0 with Apple Zero memories...${NC}"
if [ -f "scripts/setup_mem0.py" ]; then
    echo "Running mem0 setup script..."
    python scripts/setup_mem0.py
else
    echo -e "${YELLOW}Warning: mem0 setup script not found at scripts/setup_mem0.py${NC}"
fi

# --- Step 6: Install Playwright Browsers ---
echo -e "\n${GREEN}Step 6: Installing Playwright browsers...${NC}"
# Install Chromium into project-local cache so runtime finds it
export PLAYWRIGHT_BROWSERS_PATH="$(pwd)/tmp/playwright"
playwright install chromium
# Optionally also prefetch the lightweight headless shell
playwright install chromium --only-shell || true

echo -e "\n${GREEN}Setup complete!${NC}"
echo "You can now run the application with:"
echo -e "${YELLOW}./dev/macos/run.sh${NC}"

# Deactivate virtual environment
deactivate
