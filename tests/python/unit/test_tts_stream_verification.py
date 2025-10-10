import io
import struct
import wave
import numpy as np
import pytest


def test_wav_header_format():
    """Test that WAV headers are properly formatted"""
    from python.helpers.xtts_tts import wav_header

    # Test mono 16-bit WAV
    sample_rate = 24000
    channels = 1
    bits_per_sample = 16

    header = wav_header(sample_rate, channels, bits_per_sample, data_bytes=None)

    # WAV header should be exactly 44 bytes for PCM format
    assert len(header) == 44, f"Expected 44 bytes, got {len(header)}"

    # Parse the header
    riff_magic = header[0:4]
    assert riff_magic == b'RIFF', f"Expected 'RIFF', got {riff_magic}"

    # File size (will be FFFFFFFF for streaming)
    file_size = struct.unpack('<I', header[4:8])[0]
    assert file_size == 0xFFFFFFFF, f"Expected streaming size, got {file_size}"

    wave_magic = header[8:12]
    assert wave_magic == b'WAVE', f"Expected 'WAVE', got {wave_magic}"

    fmt_magic = header[12:16]
    assert fmt_magic == b'fmt ', f"Expected 'fmt ', got {fmt_magic}"

    # Format chunk size (should be 16 for PCM)
    fmt_size = struct.unpack('<I', header[16:20])[0]
    assert fmt_size == 16, f"Expected 16, got {fmt_size}"

    # Audio format (1 = PCM)
    audio_format = struct.unpack('<H', header[20:22])[0]
    assert audio_format == 1, f"Expected PCM (1), got {audio_format}"

    # Channels
    header_channels = struct.unpack('<H', header[22:24])[0]
    assert header_channels == channels, f"Expected {channels}, got {header_channels}"

    # Sample rate
    header_sample_rate = struct.unpack('<I', header[24:28])[0]
    assert header_sample_rate == sample_rate, f"Expected {sample_rate}, got {header_sample_rate}"

    # Byte rate
    byte_rate = struct.unpack('<I', header[28:32])[0]
    expected_byte_rate = sample_rate * channels * (bits_per_sample // 8)
    assert byte_rate == expected_byte_rate, f"Expected {expected_byte_rate}, got {byte_rate}"

    # Block align
    block_align = struct.unpack('<H', header[32:34])[0]
    expected_block_align = channels * (bits_per_sample // 8)
    assert block_align == expected_block_align, f"Expected {expected_block_align}, got {block_align}"

    # Bits per sample
    header_bits_per_sample = struct.unpack('<H', header[34:36])[0]
    assert header_bits_per_sample == bits_per_sample, f"Expected {bits_per_sample}, got {header_bits_per_sample}"

    data_magic = header[36:40]
    assert data_magic == b'data', f"Expected 'data', got {data_magic}"

    # Data size (streaming)
    data_size = struct.unpack('<I', header[40:44])[0]
    assert data_size == 0xFFFFFFFF, f"Expected streaming size, got {data_size}"


def test_pcm16_data_integrity():
    """Test that PCM16 data is properly formatted and bounded"""

    # Create test PCM data
    sample_rate = 24000
    duration_ms = 500
    frequency = 440  # A4 note

    # Generate sine wave
    samples = int(sample_rate * duration_ms / 1000)
    time_array = np.linspace(0, duration_ms / 1000, samples)
    sine_wave = np.sin(2 * np.pi * frequency * time_array)

    # Convert to int16 PCM
    pcm_bytes = (sine_wave * 32767.0).astype(np.int16).tobytes()

    # Verify data format
    assert len(pcm_bytes) == samples * 2, f"Expected {samples * 2} bytes, got {len(pcm_bytes)}"
    assert len(pcm_bytes) % 2 == 0, "PCM16 data should be even-length"

    # Convert back and verify values are within int16 range
    pcm_data = np.frombuffer(pcm_bytes, dtype=np.int16)
    assert np.all(pcm_data >= -32768), "PCM values should not be less than -32768"
    assert np.all(pcm_data <= 32767), "PCM values should not exceed 32767"

    # Verify approximate sine wave shape (rough check)
    reconstructed = pcm_data.astype(np.float32) / 32767.0
    assert np.min(reconstructed) >= -1.0, "Reconstructed values should be >= -1.0"
    assert np.max(reconstructed) <= 1.0, "Reconstructed values should be <= 1.0"


def test_streaming_audio_concatenation():
    """Test that streamed audio chunks concatenate properly"""

    # Create two chunks of PCM data
    sample_rate = 24000
    chunk_duration_ms = 200

    samples_per_chunk = int(sample_rate * chunk_duration_ms / 1000)

    # First chunk: rising sine wave
    t1 = np.linspace(0, chunk_duration_ms / 1000, samples_per_chunk)
    chunk1_data = (np.sin(2 * np.pi * 220 * t1) * 0.5).astype(np.float32)
    chunk1_pcm = (chunk1_data * 32767.0).astype(np.int16).tobytes()

    # Second chunk: same wave
    t2 = np.linspace(0, chunk_duration_ms / 1000, samples_per_chunk)
    chunk2_data = (np.sin(2 * np.pi * 220 * t2) * 0.5).astype(np.float32)
    chunk2_pcm = (chunk2_data * 32767.0).astype(np.int16).tobytes()

    # Concatenate
    concatenated = chunk1_pcm + chunk2_pcm

    # Verify total length
    total_samples = samples_per_chunk * 2
    assert len(concatenated) == total_samples * 2, f"Expected {total_samples * 2} bytes, got {len(concatenated)}"

    # Reconstruct and verify
    reconstructed = np.frombuffer(concatenated, dtype=np.int16).astype(np.float32) / 32767.0
    assert len(reconstructed) == total_samples
    assert np.all(np.abs(reconstructed) <= 1.0), "All values should be within [-1.0, 1.0]"


def test_wav_file_with_header():
    """Test complete WAV file creation with valid header and data"""

    from python.helpers.xtts_tts import wav_header

    sample_rate = 24000
    samples = 480  # 20ms at 24kHz
    frequency = 1000  # 1kHz test tone

    # Generate sine wave
    t = np.linspace(0, samples / sample_rate, samples)
    sine_wave = np.sin(2 * np.pi * frequency * t)

    # Convert to PCM16
    pcm_data = (sine_wave * 32767.0).astype(np.int16).tobytes()

    # Create WAV file with header
    header = wav_header(sample_rate, channels=1, bits_per_sample=16, data_bytes=len(pcm_data))
    wav_content = header + pcm_data

    # Write to temporary file and verify with wave module
    with io.BytesIO(wav_content) as wav_buffer:
        with wave.open(wav_buffer, 'rb') as wf:
            # Verify header
            assert wf.getnchannels() == 1, "Should be mono"
            assert wf.getsampwidth() == 2, "Should be 16-bit"
            assert wf.getframerate() == sample_rate, f"Sample rate should be {sample_rate}"
            assert wf.getnframes() == samples, f"Should have {samples} frames"

            # Verify data
            frames = wf.readframes(wf.getnframes())
            reconstructed = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0

            # Rough check that we have a sine-like signal
            assert np.min(reconstructed) > -1.0, "Minimum should be > -1.0"
            assert np.max(reconstructed) < 1.0, "Maximum should be < 1.0"
            assert len(reconstructed) == samples, f"Should have {samples} samples"
