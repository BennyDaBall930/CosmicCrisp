import os
import urllib.error

import pytest


@pytest.mark.parametrize("disable_autostart", [True])
def test_xtts_sidecar_connection_refused_gives_actionable_error(monkeypatch, disable_autostart):
    # Force use of sidecar backend regardless of local Coqui availability
    monkeypatch.setenv("TTS_FORCE_SIDECAR", "1")
    # Point to an unlikely port to ensure connection refusal
    monkeypatch.setenv("TTS_SIDECAR_URL", "http://127.0.0.1:59999")
    if disable_autostart:
        monkeypatch.setenv("A0_DISABLE_SIDECAR_AUTOSTART", "1")

    # Make urlopen always raise URLError(ConnectionRefusedError)
    def _raise(*args, **kwargs):
        raise urllib.error.URLError(OSError(61, "Connection refused"))

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _raise)

    from python.helpers.xtts_tts import SidecarXTTSBackend, XTTSConfig

    backend = SidecarXTTSBackend(XTTSConfig())
    with pytest.raises(RuntimeError) as ei:
        backend.synthesize("Hello world")

    msg = str(ei.value)
    # Actionable message should mention sidecar and how to start it
    assert "sidecar" in msg.lower()
    assert "setup.sh" in msg and "run.sh" in msg
    assert "127.0.0.1:59999" in msg
