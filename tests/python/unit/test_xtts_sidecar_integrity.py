import time
import urllib.error
import urllib.request
from unittest.mock import patch, MagicMock
import pytest


def test_sidecar_health_check_success():
    """Test successful sidecar health check"""

    with patch('python.helpers.xtts_tts._post_json') as mock_post:
        from python.helpers.xtts_tts import SidecarXTTSBackend, XTTSConfig

        mock_post.return_value = {"status": "healthy", "version": "1.0.0"}

        backend = SidecarXTTSBackend(XTTSConfig())
        backend._base_url = "http://127.0.0.1:9999"

        # Should return True for healthy sidecar
        healthy = backend._health_check(timeout=1.0)
        assert healthy is True

        mock_post.assert_called_once()


def test_sidecar_health_check_failure():
    """Test sidecar health check failure scenarios"""

    with patch('urllib.request.urlopen') as mock_urlopen:
        from python.helpers.xtts_tts import SidecarXTTSBackend, XTTSConfig

        # Test connection refused
        mock_urlopen.side_effect = urllib.error.URLError(OSError(61, "Connection refused"))

        backend = SidecarXTTSBackend(XTTSConfig())
        backend._base_url = "http://127.0.0.1:9999"

        healthy = backend._health_check(timeout=1.0)
        assert healthy is False

    with patch('urllib.request.urlopen') as mock_urlopen:
        # Test HTTP error
        mock_resp = MagicMock()
        mock_resp.status = 500
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        backend = SidecarXTTSBackend(XTTSConfig())
        backend._base_url = "http://127.0.0.1:9999"

        healthy = backend._health_check(timeout=1.0)
        assert healthy is False


def test_sidecar_autostart_enabled():
    """Test sidecar auto-start when A0_DISABLE_SIDECAR_AUTOSTART is not set"""

    with patch('python.helpers.xtts_tts.SidecarXTTSBackend._health_check', return_value=False), \
         patch('subprocess.Popen') as mock_popen, \
         patch('os.path.exists', return_value=True), \
         patch('os.getcwd', return_value="/fake/project/root"), \
         patch('time.sleep') as mock_sleep, \
         patch('time.time') as mock_time:

        from python.helpers.xtts_tts import SidecarXTTSBackend, XTTSConfig

        # Mock the health check to succeed after some time
        def mock_health_check(timeout=1.5):
            return True

        # Set up time progression for the waiting loop
        time_values = [0, 0, 0, 0, 10]  # Health check succeeds on 5th call
        mock_time.side_effect = time_values

        backend = SidecarXTTSBackend(XTTSConfig())
        backend._health_check = mock_health_check

        # Should not be disabled by environment
        try:
            del os.environ['A0_DISABLE_SIDECAR_AUTOSTART']
        except KeyError:
            pass

        # Call _ensure_running - should auto-start the sidecar
        backend._ensure_running()

        # Should have started the subprocess
        mock_popen.assert_called_once()
        args, kwargs = mock_popen.call_args
        assert 'bash' in args[0]
        assert './run.sh' in args[0]
        assert kwargs['cwd'] == "/fake/project/root"


def test_sidecar_autostart_disabled():
    """Test sidecar autostart is blocked when A0_DISABLE_SIDECAR_AUTOSTART is set"""

    with patch('python.helpers.xtts_tts.SidecarXTTSBackend._health_check', return_value=False):
        from python.helpers.xtts_tts import SidecarXTTSBackend, XTTSConfig

        backend = SidecarXTTSBackend(XTTSConfig())
        backend._base_url = "http://127.0.0.1:9999"

        # Set environment variable to disable autostart
        import os
        old_value = os.environ.get('A0_DISABLE_SIDECAR_AUTOSTART')
        os.environ['A0_DISABLE_SIDECAR_AUTOSTART'] = '1'

        try:
            with pytest.raises(RuntimeError) as exc_info:
                backend._ensure_running()

            error_msg = str(exc_info.value)
            assert "sidecar is not running" in error_msg.lower()
            assert "set TTS engine to 'browser'" in error_msg.lower()
            assert "setup.sh" in error_msg
            assert "run.sh" in error_msg
            assert "127.0.0.1:9999" in error_msg
        finally:
            # Restore environment
            if old_value is not None:
                os.environ['A0_DISABLE_SIDECAR_AUTOSTART'] = old_value
            else:
                del os.environ['A0_DISABLE_SIDECAR_AUTOSTART']


def test_sidecar_startup_timeout():
    """Test sidecar startup times out after 90 seconds"""

    with patch('python.helpers.xtts_tts.SidecarXTTSBackend._health_check', return_value=False), \
         patch('subprocess.Popen'), \
         patch('os.path.exists', return_value=True), \
         patch('os.getcwd', return_value="/fake/project/root"), \
         patch('time.sleep'), \
         patch('time.time') as mock_time:

        from python.helpers.xtts_tts import SidecarXTTSBackend, XTTSConfig

        # Mock time to simulate timeout (start at 0, end at 91 seconds)
        mock_time.side_effect = [0] + [t + 1 for t in range(91)]

        backend = SidecarXTTSBackend(XTTSConfig())

        with pytest.raises(RuntimeError) as exc_info:
            backend._ensure_running()

        error_msg = str(exc_info.value).lower()
        assert "failed to become healthy" in error_msg
        assert "within timeout" in error_msg


def test_sidecar_already_running():
    """Test that _ensure_running doesn't attempt to start already healthy sidecar"""

    with patch('python.helpers.xtts_tts.SidecarXTTSBackend._health_check', return_value=True), \
         patch('subprocess.Popen') as mock_popen:

        from python.helpers.xtts_tts import SidecarXTTSBackend, XTTSConfig

        backend = SidecarXTTSBackend(XTTSConfig())

        # Should not attempt to start or check subprocess
        backend._ensure_running()

        # Should not have started any subprocess
        mock_popen.assert_not_called()


def test_sidecar_run_script_not_found():
    """Test error when run.sh script doesn't exist"""

    with patch('python.helpers.xtts_tts.SidecarXTTSBackend._health_check', return_value=False), \
         patch('os.path.exists', return_value=False):

        from python.helpers.xtts_tts import SidecarXTTSBackend, XTTSConfig

        backend = SidecarXTTSBackend(XTTSConfig())

        with pytest.raises(RuntimeError) as exc_info:
            backend._ensure_running()

        error_msg = str(exc_info.value).lower()
        assert "run.sh not found" in error_msg


def test_sidecar_synthesis_with_health_checks():
    """Test that synthesis ensures sidecar is healthy before making requests"""

    with patch('python.helpers.xtts_tts._post_json') as mock_post, \
         patch('python.helpers.xtts_tts.SidecarXTTSBackend._health_check') as mock_health:
        from python.helpers.xtts_tts import SidecarXTTSBackend, XTTSConfig

        mock_health.return_value = True
        mock_post.return_value = {
            "audio_b64": "UklGRgAAAA=="  # Minimal WAV file base64
        }

        backend = SidecarXTTSBackend(XTTSConfig())

        # Test synthesis
        result = backend.synthesize("test")

        # Should have checked health
        assert mock_health.called
        # Should have made request
        assert mock_post.called
        # Should have returned bytes
        assert isinstance(result, bytes)


def test_sidecar_retry_on_connection_failure():
    """Test that sidecar retries once on initial connection failure"""

    with patch('python.helpers.xtts_tts._post_json') as mock_post, \
         patch('python.helpers.xtts_tts.SidecarXTTSBackend._health_check') as mock_health, \
         patch('python.helpers.xtts_tts.SidecarXTTSBackend._ensure_running') as mock_ensure:
        from python.helpers.xtts_tts import SidecarXTTSBackend, XTTSConfig

        # First call fails with URLError, second succeeds
        mock_post.side_effect = [
            urllib.error.URLError(OSError(61, "Connection refused")),
            {"audio_b64": "UklGRgAAAA=="}
        ]

        backend = SidecarXTTSBackend(XTTSConfig())

        # Should succeed on retry
        result = backend.synthesize("test")

        assert isinstance(result, bytes)
        assert mock_post.call_count == 2
        assert mock_ensure.call_count == 2  # Called once initially, once on retry
