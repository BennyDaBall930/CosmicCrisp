"""
macOS PTY Adapter for native terminal execution.
Provides low-level PTY operations for terminal session management.
"""

import os
import sys
import signal
import fcntl
import termios
import struct
import select
import time
import logging
from typing import Optional, Dict, Tuple, Union
from queue import Queue, Empty
import threading
import pexpect
from pexpect import pxssh, spawn, EOF, TIMEOUT
from ptyprocess import PtyProcess, PtyProcessUnicode

logger = logging.getLogger(__name__)


class MacOSPTYAdapter:
    """Main PTY management class for macOS native terminal execution."""
    
    def __init__(self):
        self.processes: Dict[str, PtyProcessUnicode] = {}
        self.readers: Dict[str, threading.Thread] = {}
        self.buffers: Dict[str, Queue] = {}
        self.lock = threading.Lock()
        self._stop_events: Dict[str, threading.Event] = {}
        
    def spawn_pty(self, 
                  command: str = "/bin/bash",
                  cwd: str = None,
                  env: Dict[str, str] = None,
                  rows: int = 40,
                  cols: int = 120) -> Tuple[str, PtyProcessUnicode]:
        """
        Create new PTY subprocess.
        
        Args:
            command: Shell command to spawn (default: /bin/bash)
            cwd: Working directory for the process
            env: Environment variables dict
            rows: Terminal rows
            cols: Terminal columns
            
        Returns:
            Tuple of (session_id, PTYProcess instance)
        """
        try:
            # Generate unique session ID
            session_id = f"pty_{int(time.time() * 1000000)}"
            
            # Prepare environment
            process_env = os.environ.copy()
            if env:
                process_env.update(env)
                
            # Set terminal type
            process_env['TERM'] = 'xterm-256color'
            process_env['LANG'] = 'en_US.UTF-8'
            process_env['LC_ALL'] = 'en_US.UTF-8'
            
            # Spawn PTY process
            logger.info(f"Spawning PTY process: {command} in {cwd}")
            pty_process = PtyProcessUnicode.spawn(
                [command],
                cwd=cwd or os.path.expanduser("~"),
                env=process_env,
                dimensions=(rows, cols)
            )
            
            # Store process
            with self.lock:
                self.processes[session_id] = pty_process
                self.buffers[session_id] = Queue(maxsize=10000)
                self._stop_events[session_id] = threading.Event()
                
            # Start reader thread
            reader_thread = threading.Thread(
                target=self._reader_thread,
                args=(session_id,),
                daemon=True
            )
            reader_thread.start()
            self.readers[session_id] = reader_thread
            
            logger.info(f"PTY process spawned successfully: {session_id}")
            return session_id, pty_process
            
        except Exception as e:
            logger.error(f"Failed to spawn PTY process: {e}")
            raise
            
    def write_to_pty(self, session_id: str, data: Union[str, bytes]) -> None:
        """
        Write data to PTY with proper encoding.
        
        Args:
            session_id: Session identifier
            data: Data to write (string or bytes)
        """
        with self.lock:
            if session_id not in self.processes:
                raise ValueError(f"Session {session_id} not found")
                
            pty_process = self.processes[session_id]
            
        try:
            if isinstance(data, str):
                pty_process.write(data)
            else:
                pty_process.write(data.decode('utf-8', errors='replace'))
                
            logger.debug(f"Wrote {len(data)} bytes to PTY {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to write to PTY {session_id}: {e}")
            raise
            
    def read_from_pty(self, session_id: str, timeout: float = 0.1) -> Optional[str]:
        """
        Non-blocking read with timeout from PTY buffer.
        
        Args:
            session_id: Session identifier
            timeout: Read timeout in seconds
            
        Returns:
            String data or None if no data available
        """
        if session_id not in self.buffers:
            raise ValueError(f"Session {session_id} not found")
            
        try:
            return self.buffers[session_id].get(timeout=timeout)
        except Empty:
            return None
            
    def read_all_from_pty(self, session_id: str) -> str:
        """
        Read all available data from PTY buffer.
        
        Args:
            session_id: Session identifier
            
        Returns:
            All available string data
        """
        if session_id not in self.buffers:
            raise ValueError(f"Session {session_id} not found")
            
        result = []
        buffer = self.buffers[session_id]
        
        while not buffer.empty():
            try:
                result.append(buffer.get_nowait())
            except Empty:
                break
                
        return ''.join(result)
        
    def resize_pty(self, session_id: str, rows: int, cols: int) -> None:
        """
        Resize terminal window.
        
        Args:
            session_id: Session identifier
            rows: New number of rows
            cols: New number of columns
        """
        with self.lock:
            if session_id not in self.processes:
                raise ValueError(f"Session {session_id} not found")
                
            pty_process = self.processes[session_id]
            
        try:
            pty_process.setwinsize(rows, cols)
            logger.info(f"Resized PTY {session_id} to {rows}x{cols}")
            
        except Exception as e:
            logger.error(f"Failed to resize PTY {session_id}: {e}")
            raise
            
    def check_permission_needed(self, command: str) -> Tuple[bool, str]:
        """
        Check if command requires confirmation (sudo/rm).
        
        Args:
            command: Command string to check
            
        Returns:
            Tuple of (needs_permission, reason)
        """
        command_lower = command.strip().lower()
        
        # Check for sudo commands
        if command_lower.startswith('sudo '):
            return True, "This command requires administrative privileges"
            
        # Check for dangerous rm commands
        if command_lower.startswith('rm '):
            # Check for recursive or force flags
            if ' -r' in command_lower or ' -f' in command_lower or ' -rf' in command_lower:
                return True, "This command will permanently delete files"
                
            # Check for wildcards
            if '*' in command:
                return True, "This command uses wildcards and may delete multiple files"
                
        # Check for dangerous directory removals
        if command_lower.startswith('rmdir '):
            return True, "This command will remove a directory"
            
        # Check for system file modifications
        dangerous_paths = ['/etc/', '/usr/', '/bin/', '/sbin/', '/System/', '/Library/']
        for path in dangerous_paths:
            if path in command:
                return True, f"This command affects system files in {path}"
                
        return False, ""
        
    def kill_pty(self, session_id: str) -> None:
        """
        Safely terminate PTY process.
        
        Args:
            session_id: Session identifier
        """
        with self.lock:
            if session_id not in self.processes:
                logger.warning(f"Session {session_id} not found for termination")
                return

            pty_process = self.processes.get(session_id)
            reader_thread = self.readers.get(session_id)
            stop_event = self._stop_events.get(session_id)

        # Signal reader thread to stop
        if stop_event:
            stop_event.set()

        # Try graceful shutdown
        if pty_process and pty_process.isalive():
            try:
                pty_process.write('exit\r\n')
            except OSError:
                pass # Process might have already died

            # Wait for process to terminate
            deadline = time.time() + 2.0 # 2 second timeout
            while time.time() < deadline and pty_process.isalive():
                time.sleep(0.05)

        # Force kill if still alive
        if pty_process and pty_process.isalive():
            logger.warning(f"Graceful shutdown of {session_id} failed. Forcing termination.")
            try:
                pty_process.kill(signal.SIGKILL)
            except Exception as e:
                logger.error(f"Error during force kill of {session_id}: {e}")

        # Wait for reader thread to finish
        if reader_thread and reader_thread.is_alive():
            reader_thread.join(timeout=1.0)
            
        # Clean up resources
        with self.lock:
            if session_id in self.processes:
                del self.processes[session_id]
            if session_id in self.buffers:
                del self.buffers[session_id]
            if session_id in self.readers:
                del self.readers[session_id]
            if session_id in self._stop_events:
                del self._stop_events[session_id]
                    
        logger.info(f"PTY process {session_id} terminated successfully")
            
    def is_alive(self, session_id: str) -> bool:
        """
        Check if PTY process is still alive.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if process is alive, False otherwise
        """
        with self.lock:
            if session_id not in self.processes:
                return False
                
            return self.processes[session_id].isalive()
            
    def get_exit_status(self, session_id: str) -> Optional[int]:
        """
        Get exit status of terminated PTY process.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Exit status code or None if still running
        """
        with self.lock:
            if session_id not in self.processes:
                return None
                
            pty_process = self.processes[session_id]
            
        if not pty_process.isalive():
            return pty_process.exitstatus
            
        return None
        
    def _reader_thread(self, session_id: str):
        """
        Background thread to read PTY output into buffer.
        
        Args:
            session_id: Session identifier
        """
        logger.debug(f"Starting reader thread for {session_id}")
        
        try:
            pty_process = self.processes[session_id]
            buffer = self.buffers[session_id]
            stop_event = self._stop_events[session_id]

            while not stop_event.is_set():
                if not pty_process.isalive():
                    # Process terminated
                    logger.info(f"PTY process {session_id} terminated")
                    break
                    
                try:
                    # Use select to check if data is available for reading
                    if select.select([pty_process.fd], [], [], 0.1)[0]:
                        # Data available, read it
                        output = pty_process.read(4096)
                        
                        if output:
                            # Add to buffer
                            if not buffer.full():
                                buffer.put(output)
                            else:
                                # Buffer full, drop oldest data
                                try:
                                    buffer.get_nowait()
                                    buffer.put(output)
                                except Empty:
                                    pass
                    else:
                        # No data available, small sleep to prevent busy waiting
                        time.sleep(0.01)
                                
                except OSError as e:
                    # Broken pipe or similar - process likely terminated
                    if e.errno in (9, 32):  # EBADF, EPIPE
                        logger.info(f"PTY process {session_id} pipe closed")
                        break
                    logger.error(f"OS error reading from PTY {session_id}: {e}")
                    break
                    
                except Exception as e:
                    logger.error(f"Error reading from PTY {session_id}: {e}")
                    break
                    
        except Exception as e:
            # Catch cases where session was killed and dicts are modified
            if not isinstance(e, KeyError):
                logger.error(f"Reader thread error for {session_id}: {e}")
            
        finally:
            logger.debug(f"Reader thread ending for {session_id}")
            
    def cleanup_all(self):
        """Clean up all PTY processes on shutdown."""
        logger.info("Cleaning up all PTY processes")
        
        session_ids = list(self.processes.keys())
        for session_id in session_ids:
            try:
                self.kill_pty(session_id)
            except Exception as e:
                logger.error(f"Error cleaning up {session_id}: {e}")


# Global instance
_adapter = None


def get_adapter() -> MacOSPTYAdapter:
    """Get singleton PTY adapter instance."""
    global _adapter
    if _adapter is None:
        _adapter = MacOSPTYAdapter()
    return _adapter


# Register cleanup on exit
import atexit
atexit.register(lambda: get_adapter().cleanup_all() if _adapter else None)
