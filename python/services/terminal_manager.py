"""
Terminal Session Manager for macOS native execution.
Manages terminal sessions, permission requests, and session cleanup.
"""

import os
import time
import uuid
import threading
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, AsyncIterator, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty

from python.adapters.terminal.macos_pty import get_adapter, MacOSPTYAdapter
from python.helpers.signal_handler import register_cleanup_callback

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Terminal session status enum."""
    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    TERMINATED = "terminated"
    ERROR = "error"


@dataclass
class TerminalSession:
    """Terminal session data structure."""
    session_id: str
    cwd: str
    started_at: datetime
    last_activity: datetime
    status: SessionStatus
    pty_handle: Any  # PTYProcess instance
    buffer: Queue = field(default_factory=lambda: Queue(maxsize=10000))
    rows: int = 40
    cols: int = 120
    env: Dict[str, str] = field(default_factory=dict)
    
    @property
    def idle_seconds(self) -> float:
        """Get seconds since last activity."""
        return (datetime.now() - self.last_activity).total_seconds()


@dataclass
class PermissionRequest:
    """Permission request for sudo/delete operations."""
    token: str
    command: str
    reason: str
    session_id: str
    created_at: datetime
    expires_at: datetime
    confirmed: bool = False


@dataclass
class TerminalEvent:
    """Event for terminal output streaming."""
    type: str  # stdout, stderr, exit, prompt, system
    data: str
    timestamp: float
    session_id: str


@dataclass
class TerminalConfig:
    """Terminal configuration settings."""
    shell_path: str = "/bin/bash"
    rows: int = 40
    cols: int = 120
    env: Dict[str, str] = field(default_factory=dict)
    working_dir: str = ""
    permission_timeout_seconds: int = 30
    idle_timeout_seconds: int = 300  # 5 minutes
    max_sessions: int = 10


class TerminalSessionManager:
    """
    Singleton managing all terminal sessions, permission requests, cleanup.
    """
    
    def __init__(self, config: Optional[TerminalConfig] = None):
        self.config = config or TerminalConfig()
        self.adapter: MacOSPTYAdapter = get_adapter()
        self.sessions: Dict[str, TerminalSession] = {}
        self.permission_requests: Dict[str, PermissionRequest] = {}
        self.lock = threading.Lock()
        self.cleanup_thread = None
        self.running = False
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
    def create_session(self,
                      cwd: Optional[str] = None,
                      env: Optional[Dict[str, str]] = None,
                      rows: Optional[int] = None,
                      cols: Optional[int] = None) -> str:
        """
        Initialize new terminal session.
        
        Args:
            cwd: Working directory
            env: Environment variables
            rows: Terminal rows
            cols: Terminal columns
            
        Returns:
            Session ID
        """
        # Check max sessions
        if len(self.sessions) >= self.config.max_sessions:
            self._cleanup_idle_sessions()
            if len(self.sessions) >= self.config.max_sessions:
                raise RuntimeError(f"Maximum sessions ({self.config.max_sessions}) reached")
        
        # Use defaults if not provided
        cwd = cwd or self.config.working_dir or os.path.expanduser("~")
        env = env or self.config.env.copy()
        rows = rows or self.config.rows
        cols = cols or self.config.cols
        
        try:
            # Spawn PTY process
            session_id, pty_handle = self.adapter.spawn_pty(
                command=self.config.shell_path,
                cwd=cwd,
                env=env,
                rows=rows,
                cols=cols
            )
            
            # Create session object
            session = TerminalSession(
                session_id=session_id,
                cwd=cwd,
                started_at=datetime.now(),
                last_activity=datetime.now(),
                status=SessionStatus.RUNNING,
                pty_handle=pty_handle,
                rows=rows,
                cols=cols,
                env=env
            )
            
            # Store session
            with self.lock:
                self.sessions[session_id] = session
                
            logger.info(f"Created terminal session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
            
    def get_session(self, session_id: str) -> Optional[TerminalSession]:
        """
        Retrieve active session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            TerminalSession or None
        """
        with self.lock:
            session = self.sessions.get(session_id)
            
        if session:
            # Update last activity
            session.last_activity = datetime.now()
            
            # Check if process is still alive
            if not self.adapter.is_alive(session_id):
                session.status = SessionStatus.TERMINATED
                
        return session
        
    def write_to_session(self, session_id: str, data: str) -> Dict[str, Any]:
        """
        Write to terminal with permission check.
        
        Args:
            session_id: Session identifier
            data: Data to write
            
        Returns:
            Response dict with status and permission info if needed
        """
        session = self.get_session(session_id)
        if not session:
            return {"error": "Session not found"}
            
        if session.status == SessionStatus.TERMINATED:
            return {"error": "Session terminated"}
            
        # Check if permission is needed
        needs_permission, reason = self.adapter.check_permission_needed(data.strip())
        
        if needs_permission:
            # Create permission request
            token = str(uuid.uuid4())
            request = PermissionRequest(
                token=token,
                command=data.strip(),
                reason=reason,
                session_id=session_id,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=self.config.permission_timeout_seconds)
            )
            
            with self.lock:
                self.permission_requests[token] = request
                
            logger.warning(f"Permission required for command: {data.strip()}")
            
            return {
                "status": "permission_required",
                "token": token,
                "command": data.strip(),
                "reason": reason,
                "expires_in": self.config.permission_timeout_seconds
            }
            
        # Write to PTY
        try:
            self.adapter.write_to_pty(session_id, data)
            session.last_activity = datetime.now()
            
            return {"status": "success"}
            
        except Exception as e:
            logger.error(f"Failed to write to session {session_id}: {e}")
            return {"error": str(e)}
            
    def confirm_permission(self, token: str) -> bool:
        """
        Validate and execute permission request.
        
        Args:
            token: Permission token
            
        Returns:
            True if confirmed and executed
        """
        with self.lock:
            request = self.permission_requests.get(token)
            
        if not request:
            logger.warning(f"Permission token not found: {token}")
            return False
            
        # Check expiration
        if datetime.now() > request.expires_at:
            logger.warning(f"Permission token expired: {token}")
            with self.lock:
                del self.permission_requests[token]
            return False
            
        # Mark as confirmed
        request.confirmed = True
        
        # Execute command
        session = self.get_session(request.session_id)
        if not session:
            logger.error(f"Session not found for permission: {request.session_id}")
            return False
            
        try:
            # Write command to PTY
            self.adapter.write_to_pty(request.session_id, request.command + "\n")
            session.last_activity = datetime.now()
            
            # Clean up request
            with self.lock:
                del self.permission_requests[token]
                
            logger.info(f"Permission granted and command executed: {request.command}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute permitted command: {e}")
            return False
            
    async def stream_session(self, session_id: str) -> AsyncIterator[TerminalEvent]:
        """
        Stream terminal output asynchronously.
        
        Args:
            session_id: Session identifier
            
        Yields:
            TerminalEvent objects
        """
        session = self.get_session(session_id)
        if not session:
            yield TerminalEvent(
                type="error",
                data="Session not found",
                timestamp=time.time(),
                session_id=session_id
            )
            return
            
        logger.info(f"Starting stream for session: {session_id}")
        
        try:
            while True:
                # Check if session is still alive
                if session.status == SessionStatus.TERMINATED:
                    yield TerminalEvent(
                        type="exit",
                        data="Session terminated",
                        timestamp=time.time(),
                        session_id=session_id
                    )
                    break
                    
                # Read from PTY buffer
                output = self.adapter.read_from_pty(session_id, timeout=0.1)
                
                if output:
                    yield TerminalEvent(
                        type="stdout",
                        data=output,
                        timestamp=time.time(),
                        session_id=session_id
                    )
                    
                # Small delay to prevent CPU spinning
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Stream error for session {session_id}: {e}")
            yield TerminalEvent(
                type="error",
                data=str(e),
                timestamp=time.time(),
                session_id=session_id
            )
            
    def resize_session(self, session_id: str, rows: int, cols: int) -> bool:
        """
        Resize terminal window.
        
        Args:
            session_id: Session identifier
            rows: New number of rows
            cols: New number of columns
            
        Returns:
            True if successful
        """
        session = self.get_session(session_id)
        if not session:
            return False
            
        try:
            self.adapter.resize_pty(session_id, rows, cols)
            session.rows = rows
            session.cols = cols
            session.last_activity = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Failed to resize session {session_id}: {e}")
            return False
            
    def terminate_session(self, session_id: str) -> bool:
        """
        Terminate a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        with self.lock:
            session = self.sessions.get(session_id)
            
        if not session:
            return False
            
        try:
            # Kill PTY process
            self.adapter.kill_pty(session_id)
            
            # Update status
            session.status = SessionStatus.TERMINATED
            
            # Remove from active sessions
            with self.lock:
                del self.sessions[session_id]
                
            logger.info(f"Terminated session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to terminate session {session_id}: {e}")
            return False
            
    def list_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        List all active sessions.
        
        Returns:
            Dict of session info
        """
        result = {}
        
        with self.lock:
            for session_id, session in self.sessions.items():
                result[session_id] = {
                    "cwd": session.cwd,
                    "started_at": session.started_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "idle_seconds": session.idle_seconds,
                    "status": session.status.value,
                    "rows": session.rows,
                    "cols": session.cols
                }
                
        return result
        
    def cleanup_idle_sessions(self) -> None:
        """Garbage collect idle sessions."""
        self._cleanup_idle_sessions()
        
    def _cleanup_idle_sessions(self) -> None:
        """Internal cleanup method."""
        now = datetime.now()
        sessions_to_remove = []
        
        with self.lock:
            for session_id, session in self.sessions.items():
                # Check idle timeout
                if session.idle_seconds > self.config.idle_timeout_seconds:
                    sessions_to_remove.append(session_id)
                    logger.info(f"Marking session {session_id} for cleanup (idle)")
                    
                # Check if process is dead
                elif not self.adapter.is_alive(session_id):
                    sessions_to_remove.append(session_id)
                    logger.info(f"Marking session {session_id} for cleanup (dead)")
                    
        # Remove idle/dead sessions
        for session_id in sessions_to_remove:
            self.terminate_session(session_id)
            
        # Clean up expired permission requests
        expired_tokens = []
        with self.lock:
            for token, request in self.permission_requests.items():
                if now > request.expires_at:
                    expired_tokens.append(token)
                    
        for token in expired_tokens:
            with self.lock:
                del self.permission_requests[token]
            logger.info(f"Cleaned up expired permission token: {token}")
            
    def _cleanup_thread_worker(self):
        """Background thread for periodic cleanup."""
        logger.info("Cleanup thread started")
        
        while self.running:
            try:
                # Run cleanup every 30 seconds
                time.sleep(30)
                self._cleanup_idle_sessions()
                
            except Exception as e:
                logger.error(f"Cleanup thread error: {e}")
                
        logger.info("Cleanup thread stopped")
        
    def _start_cleanup_thread(self):
        """Start the cleanup background thread."""
        if not self.cleanup_thread or not self.cleanup_thread.is_alive():
            self.running = True
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_thread_worker,
                daemon=True
            )
            self.cleanup_thread.start()
            
    def stop(self):
        """Stop the manager and cleanup all sessions."""
        logger.info("Stopping terminal session manager")
        
        # Stop cleanup thread
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=2)
            
        # Terminate all sessions
        session_ids = list(self.sessions.keys())
        for session_id in session_ids:
            self.terminate_session(session_id)
            
        logger.info("Terminal session manager stopped")


# Global instance
_manager: Optional[TerminalSessionManager] = None


def get_manager() -> TerminalSessionManager:
    """Get singleton terminal manager instance."""
    global _manager
    if _manager is None:
        _manager = TerminalSessionManager()
    return _manager


# Register cleanup on exit
import atexit
atexit.register(lambda: get_manager().stop() if _manager else None)
