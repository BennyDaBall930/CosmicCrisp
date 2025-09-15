"""
Integration tests for Terminal API
Tests the API layer and session management functionality
"""

import asyncio
import os
import sys
import tempfile
import json
from datetime import timedelta

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import pytest
    pytest_available = True
except ImportError:
    pytest_available = False
    print("Warning: pytest not available, using simple test runner")

from python.services.terminal_manager import TerminalSessionManager, SessionStatus


def simple_assert(condition, message):
    """Simple assertion for when pytest is not available"""
    if not condition:
        raise AssertionError(message)


class TestTerminalAPI:
    """Test suite for Terminal API integration"""
    
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
                session = self.manager.sessions[session_id]
                if hasattr(session, 'pty_handle') and session.pty_handle:
                    session.pty_handle.kill()
            except:
                pass
        self.manager.sessions.clear()
        self.manager.permission_requests.clear()


    def test_session_lifecycle(self):
        """Test full session lifecycle: start -> write -> stream -> close"""
        async def _test():
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create session
                session_id = self.manager.create_session(
                    cwd=temp_dir,
                    env={"TERM": "xterm-256color"},
                    rows=24,
                    cols=80
                )
                
                simple_assert(session_id is not None, "Session creation failed")
                simple_assert(session_id in self.manager.sessions, "Session not in manager")
                
                # Write to session
                result = self.manager.write_to_session(session_id, "echo 'test'\n")
                simple_assert(result["status"] == "success", "Write to session failed")
                
                # Give command time to execute
                await asyncio.sleep(0.2)
                
                # Get session status
                session = self.manager.get_session(session_id)
                simple_assert(session is not None, "Session retrieval failed")
                simple_assert(session.status == SessionStatus.RUNNING, "Session not running")
                
                # Clean up
                self.manager.terminate_session(session_id)
        
        if pytest_available:
            # If pytest is available, this will be run as a pytest
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_test())
            finally:
                loop.close()
        else:
            # Run directly
            asyncio.run(_test())


    def test_permission_flow(self):
        """Test permission request and confirmation flow"""
        async def _test():
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create session
                session_id = self.manager.create_session(
                    cwd=temp_dir,
                    env={"TERM": "xterm-256color"},
                    rows=24,
                    cols=80
                )
                
                # Request permission for sudo command
                result = self.manager.write_to_session(session_id, "sudo echo 'test'\n")
                simple_assert(result["status"] == "permission_required", "Permission not required for sudo")
                simple_assert("token" in result, "Permission token not provided")
                
                token = result["token"]
                simple_assert(token in self.manager.permission_requests, "Permission request not stored")
                
                # Confirm permission
                confirmed = self.manager.confirm_permission(token)
                simple_assert(confirmed is True, "Permission confirmation failed")
                
                # Token should be consumed
                simple_assert(token not in self.manager.permission_requests, "Token not consumed")
                
                # Try to confirm again (should fail)
                confirmed_again = self.manager.confirm_permission(token)
                simple_assert(confirmed_again is False, "Token reuse succeeded (should fail)")
                
                # Clean up
                self.manager.terminate_session(session_id)
        
        if pytest_available:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_test())
            finally:
                loop.close()
        else:
            asyncio.run(_test())


    def test_concurrent_sessions(self):
        """Test handling multiple concurrent sessions"""
        async def _test():
            with tempfile.TemporaryDirectory() as temp_dir:
                session_ids = []
                
                # Create multiple sessions
                for i in range(3):
                    session_id = self.manager.create_session(
                        cwd=temp_dir,
                        env={"TERM": "xterm-256color", "SESSION_NUM": str(i)},
                        rows=24,
                        cols=80
                    )
                    session_ids.append(session_id)
                
                simple_assert(len(session_ids) == 3, "Not all sessions created")
                simple_assert(len(self.manager.sessions) == 3, "Sessions not stored properly")
                
                # Test each session independently
                for i, session_id in enumerate(session_ids):
                    result = self.manager.write_to_session(session_id, f"echo 'session{i}'\n")
                    simple_assert(result["status"] == "success", f"Session {i} write failed")
                
                # Clean up all sessions
                for session_id in session_ids:
                    self.manager.terminate_session(session_id)
        
        if pytest_available:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_test())
            finally:
                loop.close()
        else:
            asyncio.run(_test())


    def test_invalid_session_handling(self):
        """Test error handling for invalid session operations"""
        async def _test():
            # Try to get non-existent session
            session = self.manager.get_session("nonexistent")
            simple_assert(session is None, "Non-existent session returned data")
            
            # Try to write to non-existent session
            result = self.manager.write_to_session("nonexistent", "echo 'test'\n")
            simple_assert("error" in result, "Error not reported for invalid session")
            
            # Try to confirm permission for non-existent token
            confirmed = self.manager.confirm_permission("invalid_token")
            simple_assert(confirmed is False, "Permission confirmation for invalid token succeeded")
        
        if pytest_available:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_test())
            finally:
                loop.close()
        else:
            asyncio.run(_test())


    def test_idle_cleanup(self):
        """Test cleanup of idle sessions"""
        async def _test():
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create session
                session_id = self.manager.create_session(
                    cwd=temp_dir,
                    env={"TERM": "xterm-256color"},
                    rows=24,
                    cols=80
                )
                
                simple_assert(session_id in self.manager.sessions, "Session not created")
                
                # Manually set session as idle for testing
                from datetime import datetime
                session = self.manager.sessions[session_id]
                session.last_activity = datetime.now() - timedelta(hours=1, minutes=1)
                
                # Run cleanup
                self.manager.cleanup_idle_sessions()
                
                # Session should be removed
                simple_assert(session_id not in self.manager.sessions, "Idle session not cleaned up")
        
        if pytest_available:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_test())
            finally:
                loop.close()
        else:
            asyncio.run(_test())


def run_all_tests():
    """Run all tests when pytest is not available"""
    print("Running Terminal API Integration Tests...")
    
    test_instance = TestTerminalAPI()
    
    tests = [
        ("Session Lifecycle", test_instance.test_session_lifecycle),
        ("Permission Flow", test_instance.test_permission_flow),
        ("Concurrent Sessions", test_instance.test_concurrent_sessions),
        ("Invalid Session Handling", test_instance.test_invalid_session_handling),
        ("Idle Cleanup", test_instance.test_idle_cleanup),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_instance.setup_method()
            test_func()
            test_instance.teardown_method()
            print(f"✓ {test_name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}: FAILED - {str(e)}")
            failed += 1
            try:
                test_instance.teardown_method()
            except:
                pass
    
    print(f"\nTest Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    if pytest_available:
        pytest.main([__file__, "-v"])
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)
