import pytest
from unittest.mock import patch, MagicMock


class TestKokoroInitialization:
    """Test Kokoro TTS pipeline initialization and instantation bug fix"""

    @patch('python.helpers.kokoro_tts._pipeline', None)
    @patch('python.helpers.kokoro_tts.threading.Lock')
    def test_pipeline_instantiation_after_imports(self, mock_lock, mock_pipeline):
        """Test that pipeline creation happens inside try block after successful imports"""
        mock_lock_instance = MagicMock()
        mock_lock.return_value = mock_lock_instance
        mock_lock_instance.__enter__.return_value = None

        with patch('sys.exit') as mock_exit:
            mock_exit.side_effect = RuntimeError("sys.exit called")

            with patch('python.helpers.kokoro_tts.KPipeline') as mock_kpipeline:
                mock_pipeline_instance = MagicMock()
                mock_kpipeline.return_value = mock_pipeline_instance

                # Import and call _ensure_pipeline
                from python.helpers.kokoro_tts import _ensure_pipeline

                # Should successfully create pipeline within try block
                result = _ensure_pipeline()

                # Verify pipeline was created and assigned correctly
                mock_kpipeline.assert_called_once_with(lang_code="a", repo_id="hexgrad/Kokoro-82M")
                assert mock_pipeline_instance == result
                assert result is mock_pipeline_instance

    @patch('python.helpers.kokoro_tts._pipeline', None)
    @patch('python.helpers.kokoro_tts.threading.Lock')
    def test_pipeline_creation_is_reachable(self, mock_lock, mock_pipeline):
        """Test that pipeline creation code path is reachable and not unreachable"""
        mock_lock_instance = MagicMock()
        mock_lock.return_value = None
        mock_lock.return_value = mock_lock_instance
        mock_lock_instance.__enter__.return_value = None

        with patch('sys.exit') as mock_exit:
            with patch('python.helpers.kokoro_tts.KPipeline') as mock_kpipeline:
                mock_pipeline_instance = MagicMock()
                mock_kpipeline.return_value = mock_pipeline_instance

                from python.helpers.kokoro_tts import _ensure_pipeline

                # Mock all the import dependencies to avoid actual imports
                with patch.multiple('python.helpers.kokoro_tts',
                                 EspeakWrapper=MagicMock(),
                                 espeakng_loader=MagicMock(),
                                 KPipeline=mock_kpipeline):

                    # This should complete without unreachable code errors
                    result = _ensure_pipeline()

                    # Pipeline should be created and assigned
                    mock_kpipeline.assert_called_once_with(lang_code="a", repo_id="hexgrad/Kokoro-82M")
                    assert result is mock_pipeline_instance

    @patch('python.helpers.kokoro_tts._pipeline', 'existing_pipeline')
    @patch('python.helpers.kokoro_tts.threading.Lock')
    def test_existing_pipeline_returned_without_creation(self, mock_lock, mock_pipeline):
        """Test that existing pipeline is returned without attempting recreation"""
        from python.helpers.kokoro_tts import _ensure_pipeline

        # Pipeline already exists, should return it immediately
        result = _ensure_pipeline()

        # Should not attempt to create new pipeline or acquire lock
        mock_lock.assert_not_called()
        assert result == 'existing_pipeline'

    def test_import_order_and_safety(self):
        """Test that the import sequence and safety guards work correctly"""
        # This test verifies the structure is correct without actually importing
        # the potentially heavy dependencies

        # Verify we can import the module without immediate errors
        import python.helpers.kokoro_tts as kokoro_module

        # Verify the function exists
        assert hasattr(kokoro_module, '_ensure_pipeline')
        assert callable(kokoro_module._ensure_pipeline)

        # Verify module constants
        assert hasattr(kokoro_module, '_DEFAULT_VOICE')
        assert kokoro_module._DEFAULT_VOICE == "am_puck,am_onyx"
        assert kokoro_module._DEFAULT_SPEED == 1.1
        assert kokoro_module._DEFAULT_SAMPLE_RATE == 24_000
