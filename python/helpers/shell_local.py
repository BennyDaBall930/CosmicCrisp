import os
import time
import re
import asyncio
from typing import Optional, Tuple
from python.adapters.terminal.macos_pty import get_adapter, MacOSPTYAdapter
from python.helpers import files


def clean_string(text: str) -> str:
    """Clean ANSI escape sequences and control characters from string."""
    if not text:
        return text
    
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = ansi_escape.sub('', text)
    
    # Remove other control characters except newline and tab
    text = ''.join(char for char in text if char in '\n\t' or (ord(char) >= 32 and ord(char) < 127))
    
    return text


class LocalInteractiveSession:
    """
    Local interactive shell session using macOS PTY adapter.
    Provides a compatible interface for code_execution_tool.py.
    """
    
    def __init__(self):
        self.adapter: MacOSPTYAdapter = get_adapter()
        self.session_id: Optional[str] = None
        self.full_output = ''
        # Start inside the HoneyCrisp project root so relative paths save in-repo
        try:
            self.cwd = files.get_base_dir()
        except Exception:
            # Fallback to home if helper unavailable for any reason
            self.cwd = os.path.expanduser("~")
    
    async def connect(self):
        """Connect to a new PTY session."""
        # Pick a sensible bash: prefer Homebrew GNU bash if present, else $A0_SHELL, else $SHELL, else /bin/bash
        preferred_shells = []
        hb_bash = "/opt/homebrew/bin/bash"
        if os.path.exists(hb_bash):
            preferred_shells.append(hb_bash)
        # Allow override via env
        if os.getenv("A0_SHELL"):
            preferred_shells.insert(0, os.getenv("A0_SHELL") or "")
        if os.getenv("SHELL"):
            preferred_shells.append(os.getenv("SHELL") or "")
        preferred_shells.append("/bin/bash")
        command = next((sh for sh in preferred_shells if sh and os.path.exists(sh)), "/bin/bash")

        # Spawn a new PTY session
        self.session_id, _ = self.adapter.spawn_pty(
            command=command,
            cwd=self.cwd,
            rows=40,
            cols=120
        )
        
        # Clear any initial output
        await asyncio.sleep(0.5)
        self.adapter.read_all_from_pty(self.session_id)
        self.full_output = ''
        # Make bash safer and less noisy for programmatic use
        try:
            # disable history expansion to avoid `!` issues
            self.adapter.write_to_pty(self.session_id, "set +H\n")
        except Exception:
            pass
    
    async def close(self):
        """Close the PTY session."""
        if self.session_id:
            self.adapter.kill_pty(self.session_id)
            self.session_id = None
    
    async def send_command(self, command: str):
        """Send a command to the PTY session."""
        if not self.session_id:
            raise Exception("Shell not connected")
        
        # Reset output buffer
        self.full_output = ""
        
        # Send command with newline
        self.adapter.write_to_pty(self.session_id, command + "\n")
    
    async def read_output(self, timeout: float = 0, reset_full_output: bool = False) -> Tuple[str, Optional[str]]:
        """
        Read output from the PTY session.
        
        Args:
            timeout: Maximum time to wait for output
            reset_full_output: Whether to reset the full output buffer
            
        Returns:
            Tuple of (full_output, partial_output)
        """
        if not self.session_id:
            raise Exception("Shell not connected")
        
        if reset_full_output:
            self.full_output = ""
        
        # Read available output
        partial_output = ""
        start_time = time.time()
        
        while True:
            # Read from PTY buffer
            chunk = self.adapter.read_from_pty(self.session_id, timeout=0.1)
            
            if chunk:
                partial_output += chunk
            
            # Check timeout
            if timeout > 0 and (time.time() - start_time) >= timeout:
                break
            
            # If no timeout and no more data, break
            if timeout == 0 and not chunk:
                break
            
            # Small sleep to prevent CPU spinning
            if not chunk:
                await asyncio.sleep(0.01)
        
        # Add to full output
        if partial_output:
            self.full_output += partial_output
        
        # Clean output
        partial_output = clean_string(partial_output)
        clean_full_output = clean_string(self.full_output)
        
        if not partial_output:
            return clean_full_output, None
        return clean_full_output, partial_output
