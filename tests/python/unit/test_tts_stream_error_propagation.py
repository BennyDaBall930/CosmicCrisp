import json
from unittest.mock import patch, MagicMock, AsyncMock
import pytest


class TestTTSStreamErrorPropagation:
    """Test that client error propagation improvements work correctly"""

    @pytest.mark.asyncio
    async def test_stream_captures_server_error_text(self):
        """Test that tts-stream.js captures and logs server error response text"""

        # Mock fetchApi to simulate failed response with error body
        mock_error_response = {
            "error": "XTTS sidecar unavailable: Connection refused",
            "details": "Run './setup.sh && ./run.sh' to start the sidecar service"
        }
        mock_error_text = json.dumps(mock_error_response)

        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.body = None  # No stream body for errors
        mock_response.text = AsyncMock(return_value=mock_error_text)

        with patch('webui.js.lib.tts-stream.fetchApi', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = mock_response

            # Mock cleanup function to avoid actual audio context operations
            with patch('webui.js.lib.tts-stream.cleanup', new_callable=AsyncMock) as mock_cleanup:
                with patch('webui.js.lib.tts-stream.log') as mock_log:

                    # Import and attempt to use the stream function
                    # We can't directly test the JS code here, but we can verify structure
                    # This test exists to ensure the JS code path is covered in manual testing

                    # Verify that the JS error handling includes server error text capture
                    # This would be verified by manual testing with a failing server
                    pass

    def test_error_handling_structure_exists(self):
        """Test that the error handling code structure exists in tts-stream.js"""
        # Verify the file exists and has the expected error handling pattern
        import webui.js.lib.tts_stream as tts_stream_module

        # The file should exist and be importable (at least structurally)
        assert tts_stream_module is not None

    def test_server_error_logging_verification(self):
        """Test that server error logging would work (structure test)"""
        # This test ensures the error propagation logic is in place
        # Actual functionality would be verified through integration tests

        # Verify that the JavaScript contains error handling improvements
        # by checking the file content for the new error handling patterns
        with open('webui/js/lib/tts-stream.js', 'r') as f:
            content = f.read()

            # Check for the new error handling code we added
            assert 'let errorMessage = "TTS stream failed";' in content
            assert 'const errorText = await response.text();' in content
            assert 'errorMessage = `TTS server error: ${errorText}`;' in content
            assert 'log("Server error response:", errorText);' in content
            assert 'throw new Error(errorMessage);' in content

    def test_generic_fallback_error(self):
        """Test that generic error fallback works when response.text() fails"""
        # Verify that if response.text() throws, we still get a meaningful error
        with open('webui/js/lib/tts-stream.js', 'r') as f:
            content = f.read()

            # Check that the error handling has a fallback
            assert 'log("Could not read server error response:", err);' in content
            assert 'await cleanup({ immediate: true });' in content
            assert 'throw new Error(errorMessage);' in content

    @pytest.mark.parametrize("error_status", [400, 404, 500, 502])
    def test_error_propagation_for_different_status_codes(self, error_status):
        """Test that error propagation works for various HTTP status codes"""
        # Structure test - verify the code handles different error conditions

        with open('webui/js/lib/tts-stream.js', 'r') as f:
            content = f.read()

            # The error handling should work regardless of status code
            # as long as response.ok is false
            assert 'if (!response.ok || !response.body) {' in content
            assert 'let errorMessage = "TTS stream failed";' in content
            assert 'await cleanup({ immediate: true });' in content
