"""
Unit tests for MacOS PTY adapter
Tests the core terminal adapter functionality
"""

import pytest
import asyncio
import os
import tempfile
import time
from unittest.mock import Mock, patch, MagicMock
import sys
import subprocess

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from python.adapters.terminal.macos_pty import MacOSPTYAdapter
from python.services.terminal_manager import TerminalSessionManager, SessionStatus, TerminalSession


class TestMacOSPTYAdapter:
    """Test suite for MacOSPTYAdapter"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.adapter = MacOSPTYAdapter()
        self.pty_process = None
        self.session_id = None

    def teardown_method(self):
        """Cleanup after each test method"""
        # Clean up any PTY processes that might still be running
        if self.session_id:
            try:
                self.adapter.kill_pty(self.session_id)
            except:
                pass
    
    def test_spawn_pty_bash(self):
        """Test spawning PTY with bash shell"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.session_id, self.pty_process = self.adapter.spawn_pty(
                command="/bin/bash",
                cwd=temp_dir,
                env={"TERM": "xterm-256color"},
                rows=24,
                cols=80
            )
            
            assert self.session_id is not None
            assert self.pty_process.isalive()
            
            # Test basic command execution
            self.adapter.write_to_pty(self.session_id, "echo 'hello world'\n")
            time.sleep(0.1)
            
            output = self.adapter.read_all_from_pty(self.session_id)
            assert "hello world" in output
    
    def test_spawn_pty_zsh(self):
        """Test spawning PTY with zsh shell if available"""
        if not os.path.exists("/bin/zsh"):
            pytest.skip("zsh not available on this system")
            
        with tempfile.TemporaryDirectory() as temp_dir:
            self.session_id, self.pty_process = self.adapter.spawn_pty(
                command="/bin/zsh",
                cwd=temp_dir,
                env={"TERM": "xterm-256color"},
                rows=24,
                cols=80
            )
            
            assert self.session_id is not None
            assert self.pty_process.isalive()
            
            # Test basic command execution
            self.adapter.write_to_pty(self.session_id, "echo 'zsh test'\n")
            time.sleep(0.1)
            
            output = self.adapter.read_all_from_pty(self.session_id)
            assert "zsh test" in output
    
    def test_write_read_operations(self):
        """Test write and read operations with various input types"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.session_id, self.pty_process = self.adapter.spawn_pty(
                command="/bin/bash",
                cwd=temp_dir,
                env={"TERM": "xterm-256color"},
                rows=24,
                cols=80
            )
            
            # Test simple command
            self.adapter.write_to_pty(self.session_id, "echo 'test1'\n")
            time.sleep(0.1)
            output = self.adapter.read_all_from_pty(self.session_id)
            assert "test1" in output
            
            # Test special characters
            self.adapter.write_to_pty(self.session_id, "echo 'special: !@#$%^&*()'\n")
            time.sleep(0.1)
            output = self.adapter.read_all_from_pty(self.session_id)
            assert "special:" in output
            
            # Test unicode characters
            self.adapter.write_to_pty(self.session_id, "echo 'unicode: éñ中文'\n")
            time.sleep(0.1)
            output = self.adapter.read_all_from_pty(self.session_id)
            # Note: unicode handling may vary by terminal
            assert "unicode:" in output
    
    def test_resize_operations(self):
        """Test terminal resize functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.session_id, self.pty_process = self.adapter.spawn_pty(
                command="/bin/bash",
                cwd=temp_dir,
                env={"TERM": "xterm-256color"},
                rows=24,
                cols=80
            )
            
            # Test resize
            self.adapter.resize_pty(self.session_id, rows=30, cols=120)
            
            # Test that terminal responds to resize (check COLUMNS environment)
            self.adapter.write_to_pty(self.session_id, "echo $COLUMNS\n")
            time.sleep(0.1)
            output = self.adapter.read_all_from_pty(self.session_id)
            
            # Should contain the new column count
            assert "120" in output
    
    def test_permission_detection(self):
        """Test detection of commands requiring permissions"""
        # Test sudo commands
        assert self.adapter.check_permission_needed("sudo apt update")[0]
        assert self.adapter.check_permission_needed("sudo -s")[0]
        
        # Test rm commands
        assert self.adapter.check_permission_needed("rm -rf /")[0]
        assert self.adapter.check_permission_needed("rm -rf ~/important")[0]
        assert self.adapter.check_permission_needed("rm -r somedir")[0]
        
        # Test safe commands
        assert not self.adapter.check_permission_needed("ls -la")[0]
        assert not self.adapter.check_permission_needed("echo hello")[0]
        assert not self.adapter.check_permission_needed("cd /tmp")[0]
        assert not self.adapter.check_permission_needed("mkdir test")[0]
        assert not self.adapter.check_permission_needed("rm single_file.txt")[0]
    
    def test_process_cleanup(self):
        """Test proper cleanup of PTY processes"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.session_id, self.pty_process = self.adapter.spawn_pty(
                command="/bin/bash",
                cwd=temp_dir,
                env={"TERM": "xterm-256color"},
                rows=24,
                cols=80
            )
            
            assert self.pty_process.isalive()
            
            # Kill the process
            self.adapter.kill_pty(self.session_id)
            
            # Wait a bit for cleanup
            time.sleep(0.5)
            
            # Process should be dead
            assert not self.pty_process.isalive()
    
    def test_timeout_handling(self):
        """Test timeout handling in read operations"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.session_id, self.pty_process = self.adapter.spawn_pty(
                command="/bin/bash",
                cwd=temp_dir,
                env={"TERM": "xterm-256color"},
                rows=24,
                cols=80
            )
            
            # Clear initial shell prompt
            time.sleep(0.2)
            self.adapter.read_all_from_pty(self.session_id)

            # Read with short timeout when no output expected
            start_time = time.time()
            output = self.adapter.read_from_pty(self.session_id, timeout=0.5)
            elapsed = time.time() - start_time
            
            # Should timeout and return within reasonable time
            assert elapsed >= 0.5
            assert output is None
    
    def test_buffer_overflow_scenarios(self):
        """Test handling of large output buffers"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.session_id, self.pty_process = self.adapter.spawn_pty(
                command="/bin/bash",
                cwd=temp_dir,
                env={"TERM": "xterm-256color"},
                rows=24,
                cols=80
            )
            
            # Generate large output
            self.adapter.write_to_pty(self.session_id, "seq 1 1000\n")
            time.sleep(1.0)
            
            # Should be able to read large output without issues
            output = self.adapter.read_all_from_pty(self.session_id)
            assert len(output) > 1000
            assert "1000" in output
    
    def test_error_handling_invalid_command(self):
        """Test error handling for invalid commands"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test spawning with invalid command
            with pytest.raises(Exception):
                self.session_id, self.pty_process = self.adapter.spawn_pty(
                    command="/nonexistent/command",
                    cwd=temp_dir
                )

    def test_graceful_shutdown(self):
        """Test that kill_pty attempts a graceful shutdown first."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.session_id, self.pty_process = self.adapter.spawn_pty(
                command="/bin/bash",
                cwd=temp_dir
            )

            # This command will make the shell sleep for 1.5 seconds upon receiving exit
            self.adapter.write_to_pty(self.session_id, "trap 'sleep 1.5' EXIT\n")
            time.sleep(0.1) # allow time for trap to be set

            # Mock the kill method to check if it's called
            self.pty_process.kill = MagicMock()

            start_time = time.time()
            self.adapter.kill_pty(self.session_id)
            end_time = time.time()

            duration = end_time - start_time

            # The process should be dead
            assert not self.pty_process.isalive()

            # The shutdown should have been graceful, so kill should not be called
            self.pty_process.kill.assert_not_called()

            # The duration should be at least 1.5s due to the trap
            assert duration >= 1.5


class TestTerminalSessionManager:
    """Test suite for TerminalSessionManager"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.manager = TerminalSessionManager()
        # Clear any existing sessions
        self.manager.sessions = {}
        self.manager.permission_requests = {}
    
    def teardown_method(self):
        """Cleanup after each test method"""
        # Clean up all sessions
        for session_id in list(self.manager.sessions.keys()):
            try:
                self.manager.terminate_session(session_id)
            except:
                pass
    
    def test_create_session(self):
        """Test creating a new terminal session"""
        with tempfile.TemporaryDirectory() as temp_dir:
            session_id = self.manager.create_session(
                cwd=temp_dir,
                env={"TERM": "xterm-256color"},
                rows=24,
                cols=80
            )
            
            assert session_id is not None
            assert session_id in self.manager.sessions
            
            session = self.manager.sessions[session_id]
            assert session.session_id == session_id
            assert session.cwd == temp_dir
            assert session.status == SessionStatus.RUNNING
            assert session.pty_handle is not None
            assert session.pty_handle.isalive()
    
    def test_get_session(self):
        """Test retrieving existing sessions"""
        with tempfile.TemporaryDirectory() as temp_dir:
            session_id = self.manager.create_session(
                cwd=temp_dir,
            )
            
            # Test valid session retrieval
            session = self.manager.get_session(session_id)
            assert session is not None
            assert session.session_id == session_id
            
            # Test invalid session retrieval
            invalid_session = self.manager.get_session("nonexistent")
            assert invalid_session is None
    
    def test_write_to_session_safe_command(self):
        """Test writing safe commands to session"""
        with tempfile.TemporaryDirectory() as temp_dir:
            session_id = self.manager.create_session(
                cwd=temp_dir,
            )
            
            result = self.manager.write_to_session(session_id, "echo 'hello'\n")
            
            assert result["status"] == "success"
    
    def test_write_to_session_permission_required(self):
        """Test writing commands that require permission"""
        with tempfile.TemporaryDirectory() as temp_dir:
            session_id = self.manager.create_session(
                cwd=temp_dir,
            )
            
            result = self.manager.write_to_session(session_id, "sudo apt update\n")
            
            assert result["status"] == "permission_required"
            assert "token" in result
            assert result["token"] in self.manager.permission_requests
    
    def test_confirm_permission(self):
        """Test permission confirmation flow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            session_id = self.manager.create_session(
                cwd=temp_dir,
            )
            
            # Request permission
            result = self.manager.write_to_session(session_id, "sudo echo 'test'\n")
            assert result["status"] == "permission_required"
            
            token = result["token"]
            
            # Confirm permission
            confirmed = self.manager.confirm_permission(token)
            assert confirmed is True
            
            # Token should be consumed
            assert token not in self.manager.permission_requests
            
            # Try to confirm again (should fail)
            confirmed_again = self.manager.confirm_permission(token)
            assert confirmed_again is False
    
    def test_cleanup_idle_sessions(self):
        """Test cleanup of idle sessions"""
        with tempfile.TemporaryDirectory() as temp_dir:
            session_id = self.manager.create_session(
                cwd=temp_dir,
            )
            
            # Manually set session as idle for testing
            session = self.manager.sessions[session_id]
            from datetime import datetime, timedelta
            session.last_activity = datetime.now() - timedelta(seconds=3700)
            
            # Run cleanup
            self.manager.cleanup_idle_sessions()
            
            # Session should be removed
            assert session_id not in self.manager.sessions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
