"""
Signal handler for managing process cleanup and preventing zombie processes.
"""

import os
import signal
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class SignalHandler:
    """Handles system signals for proper cleanup."""
    
    def __init__(self):
        self.cleanup_callbacks: list[Callable] = []
        self.setup_handlers()
        
    def setup_handlers(self):
        """Setup signal handlers for clean shutdown."""
        try:
            # Handle SIGCHLD to prevent zombie processes
            signal.signal(signal.SIGCHLD, self._handle_sigchld)
            
            # Handle termination signals
            signal.signal(signal.SIGTERM, self._handle_termination)
            signal.signal(signal.SIGINT, self._handle_termination)
            
            logger.info("Signal handlers set up successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup signal handlers: {e}")
            
    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback for shutdown."""
        self.cleanup_callbacks.append(callback)
        
    def _handle_sigchld(self, signum, frame):
        """Handle SIGCHLD to prevent zombie processes."""
        try:
            # Reap all available zombie processes
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                    if pid == 0:
                        # No more zombie processes
                        break
                    logger.debug(f"Reaped zombie process {pid} with status {status}")
                except OSError:
                    # No more child processes
                    break
                    
        except Exception as e:
            logger.error(f"Error handling SIGCHLD: {e}")
            
    def _handle_termination(self, signum, frame):
        """Handle termination signals for clean shutdown."""
        logger.info(f"Received termination signal {signum}")
        
        try:
            # Run cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in cleanup callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
        # Exit gracefully
        os._exit(0)


# Global instance
_signal_handler: Optional[SignalHandler] = None


def get_signal_handler() -> SignalHandler:
    """Get singleton signal handler instance."""
    global _signal_handler
    if _signal_handler is None:
        _signal_handler = SignalHandler()
    return _signal_handler


def register_cleanup_callback(callback: Callable):
    """Register a cleanup callback for shutdown."""
    get_signal_handler().register_cleanup_callback(callback)
