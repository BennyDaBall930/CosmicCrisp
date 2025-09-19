import threading
from queue import Queue
from typing import Any

import pytest

from python.adapters.terminal.macos_pty import MacOSPTYAdapter


class DummyProcess:
    def __init__(self) -> None:
        self._alive = True
        self.exitstatus = 0
        self.writes: list[str] = []
        self.killed = False

    def isalive(self) -> bool:
        return self._alive

    def write(self, data: str) -> None:
        self.writes.append(data)
        self._alive = False

    def setwinsize(self, rows: int, cols: int) -> None:  # pragma: no cover - trivial setter
        self.rows = rows
        self.cols = cols

    def kill(self, _signal: int) -> None:
        self.killed = True
        self._alive = False


@pytest.fixture()
def adapter() -> MacOSPTYAdapter:
    return MacOSPTYAdapter()


def test_check_permission_needed(adapter: MacOSPTYAdapter):
    needs, reason = adapter.check_permission_needed('sudo rm -rf /')
    assert needs
    assert 'privilege' in reason.lower()
    plain = adapter.check_permission_needed('ls -la')
    assert plain == (False, '')


def test_read_all_from_buffer(adapter: MacOSPTYAdapter):
    session_id = 'session'
    adapter.buffers[session_id] = Queue()
    adapter.buffers[session_id].put('line1')
    adapter.buffers[session_id].put('line2')
    adapter.processes[session_id] = DummyProcess()
    assert adapter.read_all_from_pty(session_id) == 'line1line2'


def test_resize_pty_calls_setwinsize(adapter: MacOSPTYAdapter):
    session_id = 'session'
    proc = DummyProcess()
    adapter.processes[session_id] = proc
    adapter.resize_pty(session_id, rows=50, cols=120)
    assert proc.rows == 50
    assert proc.cols == 120


def test_kill_pty_gracefully_handles_missing_session(adapter: MacOSPTYAdapter, caplog):
    adapter.kill_pty('unknown')
    assert 'unknown' not in adapter.processes


def test_kill_pty_attempts_graceful_shutdown(adapter: MacOSPTYAdapter):
    session_id = 'session'
    proc = DummyProcess()
    adapter.processes[session_id] = proc
    adapter.buffers[session_id] = Queue()
    adapter.readers[session_id] = threading.Thread(target=lambda: None)
    adapter._stop_events[session_id] = threading.Event()
    adapter.kill_pty(session_id)
    assert not proc.isalive()
    assert proc.writes or proc.killed
    assert session_id not in adapter.processes
