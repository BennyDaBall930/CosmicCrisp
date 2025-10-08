import json
import sys
from unittest.mock import patch, MagicMock, mock_open
import pytest

from python.api.tts_status import TtsStatus, _get_tts_settings, _check_xtts_health, _check_chatterbox_availability, _check_kokoro_availability


class TestTTSStatusEndpoint:
    """Test the /api/tts_status endpoint functionality"""

    def test_get_tts_settings_basic(self):
        """Test basic TTS settings retrieval"""
        with patch('python.api.tts_status.settings') as mock_settings:
            mock_settings_obj = MagicMock()
            mock_settings_obj.get_settings.return_value = {"tts": {"engine": "xtts", "xtts": {"enabled": True}}}
            mock_settings.get_settings.return_value = mock_settings_obj
            mock_settings.get_default_settings.return_value = {"tts": {"engine": "chatterbox"}}

            result = _get_tts_settings()
            assert result == {"tts": {"engine": "xtts", "xtts": {"enabled": True}}}

    def test_get_tts_settings_fallback(self):
        """Test TTS settings fallback to defaults"""
        with patch('python.api.tts_status.settings') as mock_settings:
            mock_settings_obj = MagicMock()
            mock_settings_obj.get_settings.return_value = {}
            mock_settings.get_settings.return_value = mock_settings_obj
            mock_settings.get_default_settings.return_value = {"tts": {"engine": "browser"}}

            result = _get_tts_settings()
            assert result == {"tts": {"engine": "browser"}}

    @patch('urllib.request.urlopen')
    def test_xtts_health_check_success(self, mock_urlopen):
        """Test XTTS health check with successful response"""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result = _check_xtts_health()
        expected = {"available": True, "healthy": True, "url": "http://127.0.0.1:7055/healthz"}
        assert result == expected

    @patch('urllib.request.urlopen')
    def test_xtts_health_check_server_error(self, mock_urlopen):
        """Test XTTS health check with server error"""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.HTTPError("http://127.0.0.1:7055/healthz", 500, "Internal Server Error", None, None)

        result = _check_xtts_health()
        assert result["available"] is True
        assert result["healthy"] is False
        assert result["status"] == 500

    @patch('urllib.request.urlopen')
    def test_xtts_health_check_connection_refused(self, mock_urlopen):
        """Test XTTS health check with connection refused"""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError(OSError(61, "Connection refused"))

        result = _check_xtts_health()
        assert result["available"] is False
        assert result["healthy"] is False
        assert "error" in result

    @patch('sys.version_info', (3, 10, 0))
    def test_chatterbox_python_3_10_available(self, mock_version):
        """Test Chatterbox availability on Python 3.10"""
        import sys
        original_version = sys.version_info
        sys.version_info = (3, 10, 0)

        try:
            with patch.dict('sys.modules', {'torch': MagicMock(), 'python.helpers.chatterbox_tts': MagicMock()}):
                result = _check_chatterbox_availability()
                assert result["available"] is True
                assert "python_version" in result
        finally:
            sys.version_info = original_version

    @patch('sys.version_info', (3, 12, 0))
    def test_chatterbox_python_3_12_unavailable(self, mock_version):
        """Test Chatterbox unavailability on Python 3.12"""
        import sys
        original_version = sys.version_info
        sys.version_info = (3, 12, 0)

        try:
            result = _check_chatterbox_availability()
            assert result["available"] is False
            assert "Python 3.12+ not supported" in result["reason"]
        finally:
            sys.version_info = original_version

    @patch('sys.version_info', (3, 9, 0))
    def test_chatterbox_python_too_old(self, mock_version):
        """Test Chatterbox unavailability on Python < 3.10"""
        import sys
        original_version = sys.version_info
        sys.version_info = (3, 9, 0)

        try:
            result = _check_chatterbox_availability()
            assert result["available"] is False
            assert "Python >= 3.10 required" in result["reason"]
        finally:
            sys.version_info = original_version

    def test_chatterbox_import_failure(self):
        """Test Chatterbox availability when imports fail"""
        with patch.dict('sys.modules', {}, clear=True):
            with patch('python.api.tts_status.torch', side_effect=ImportError("torch not found")):
                result = _check_chatterbox_availability()
                assert result["available"] is False
                assert "Import failed" in result["reason"]

    def test_kokoro_availability_success(self):
        """Test Kokoro availability when dependencies can be imported"""
        with patch.dict('sys.modules', {
            'torch': MagicMock(),
            'soundfile': MagicMock(),
            'kokoro': MagicMock()
        }):
            result = _check_kokoro_availability()
            assert result["available"] is True

    def test_kokoro_import_failure(self):
        """Test Kokoro unavailability when imports fail"""
        with patch.dict('sys.modules', {}, clear=True):
            with patch('python.api.tts_status.torch', side_effect=ImportError("torch not found")):
                result = _check_kokoro_availability()
                assert result["available"] is False
                assert "Import failed" in result["reason"]

    def test_full_tts_status_response(self):
        """Test the complete TTS status endpoint response"""
        with patch('python.api.tts_status._get_tts_settings') as mock_get_settings, \
             patch('python.api.tts_status._check_xtts_health') as mock_xtts_check, \
             patch('python.api.tts_status._check_chatterbox_availability') as mock_chatterbox_check, \
             patch('python.api.tts_status._check_kokoro_availability') as mock_kokoro_check:

            mock_get_settings.return_value = {"tts": {"engine": "browser"}}
            mock_xtts_check.return_value = {"available": True, "healthy": True}
            mock_chatterbox_check.return_value = {"available": False, "reason": "Python version"}
            mock_kokoro_check.return_value = {"available": True}

            handler = TtsStatus()
            mock_request = MagicMock()
            result = handler.process({}, mock_request)

            assert "active_engine" in result
            assert "engines" in result
            assert result["active_engine"] == "browser"
            assert "browser" in result["engines"]
            assert "xtts" in result["engines"]
            assert "chatterbox" in result["engines"]
            assert "kokoro" in result["engines"]
            assert "piper_vc" in result["engines"]

            # Verify browser is always available
            assert result["engines"]["browser"]["available"] is True
            assert "Client-side TTS" in result["engines"]["browser"]["note"]

    def test_browser_always_available(self):
        """Test that browser TTS is always marked as available"""
        handler = TtsStatus()
        mock_request = MagicMock()
        result = handler.process({}, mock_request)

        assert result["engines"]["browser"]["available"] is True
        assert "requires user interaction" in result["engines"]["browser"]["note"]

    @pytest.mark.parametrize("active_engine", ["browser", "chatterbox", "xtts", "kokoro"])
    def test_active_engine_from_settings(self, active_engine):
        """Test that active engine is correctly read from settings"""
        with patch('python.api.tts_status._get_tts_settings') as mock_get_settings:
            mock_get_settings.return_value = {"tts": {"engine": active_engine}}

            handler = TtsStatus()
            mock_request = MagicMock()
            result = handler.process({}, mock_request)

            assert result["active_engine"] == active_engine
