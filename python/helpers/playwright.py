
from pathlib import Path
import os
import subprocess
from python.helpers import files


# Ensure Playwright browsers are installed under project-local cache.
# We prefer full Chromium (not just the headless shell) because browser_use
# launches a persistent context that requires the full browser binary.

def _find_full_chromium(pw_cache: Path) -> Path | None:
    # macOS path
    mac = next(
        pw_cache.glob(
            "chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium"
        ),
        None,
    )
    if mac and mac.exists():
        return mac
    # Linux path
    linux = next(pw_cache.glob("chromium-*/chrome-linux/chrome"), None)
    if linux and linux.exists():
        return linux
    # Windows path (unlikely in this environment, but harmless)
    win = next(pw_cache.glob("chromium-*/chrome-win/chrome.exe"), None)
    if win and win.exists():
        return win
    return None


def _find_headless_shell(pw_cache: Path) -> Path | None:
    return next(
        pw_cache.glob("chromium_headless_shell-*/chrome-*/headless_shell"), None
    )


def get_playwright_cache_dir():
    return files.get_abs_path("tmp/playwright")


def ensure_playwright_binary():
    cache = Path(get_playwright_cache_dir())
    cache.mkdir(parents=True, exist_ok=True)

    # First, prefer the full Chromium install.
    full = _find_full_chromium(cache)
    if not full:
        # If missing, install Chromium into our project cache.
        env = os.environ.copy()
        env["PLAYWRIGHT_BROWSERS_PATH"] = str(cache)
        try:
            subprocess.check_call(["playwright", "install", "chromium"], env=env)
        except FileNotFoundError:
            # Fallback to module invocation if CLI shim isn't on PATH
            subprocess.check_call(
                ["python", "-m", "playwright", "install", "chromium"], env=env
            )
        full = _find_full_chromium(cache)

    # Optionally prefetch the headless shell as an optimization if absent.
    if not _find_headless_shell(cache):
        env = os.environ.copy()
        env["PLAYWRIGHT_BROWSERS_PATH"] = str(cache)
        try:
            subprocess.check_call(
                ["playwright", "install", "chromium", "--only-shell"], env=env
            )
        except Exception:
            # Non-fatal: browser_use will work with full Chromium alone
            pass

    if not full:
        raise Exception("Playwright Chromium not found after installation")
    return full
