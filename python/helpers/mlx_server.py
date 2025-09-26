import os
import sys
import subprocess
import threading
import signal
import time
import json
import socket
import logging
from typing import Optional, Dict, Any

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from python.helpers.print_style import PrintStyle

_PRINTER_INFO = PrintStyle(italic=True, font_color="blue", padding=False, level=logging.DEBUG)
_PRINTER_WARN = PrintStyle(font_color="#FFA500", padding=True, level=logging.WARNING)
_PRINTER_ERROR = PrintStyle(font_color="red", padding=True, level=logging.ERROR)
_INSTANCE: Optional['MLXServerManager'] = None


class MLXServerManager:
    """
    Manages the MLX FastAPI server as a separate subprocess.
    Provides start/stop/status operations with proper process management.
    Singleton instance persists across module reloads using file-based storage.
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._lock = threading.Lock()
        self._status = "stopped"
        self._settings_path = None
        self._port = 8082  # Default MLX server port

        # File-based persistence for singleton across reloads
        from pathlib import Path
        helpers_dir = Path(__file__).parent
        self._persistence_dir = helpers_dir.parent / "tmp" / "mlx_server"
        self._persistence_dir.mkdir(parents=True, exist_ok=True)
        self._persistence_file = self._persistence_dir / "server_state.json"
        self._project_root = helpers_dir.parent.parent
        self._external_pid_file = self._project_root / "tmp" / "mlx.pid"

    def _save_state(self):
        """Save server state to persistent storage."""
        try:
            state = {
                "status": self._status,
                "pid": self._process.pid if self._process else None,
                "port": self._port,
                "settings_path": self._settings_path,
                "timestamp": time.time()
            }
            with open(self._persistence_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            _PRINTER_ERROR.print(f"[MLX Server] Error saving state: {e}")

    def _load_state(self, *, ignore_expiry: bool = False):
        """Load server state from persistent storage."""
        try:
            if not self._persistence_file.exists():
                return None

            with open(self._persistence_file, 'r') as f:
                state = json.load(f)

            # Validate state is recent (within last 5 minutes)
            timestamp = state.get("timestamp", 0)
            if not ignore_expiry and time.time() - timestamp > 300:  # 5 minutes
                return None

            return state
        except Exception as e:
            _PRINTER_ERROR.print(f"[MLX Server] Error loading state: {e}")
            return None

    def _restore_state(self):
        """Restore server state from persistent storage."""
        state = self._load_state()
        if state:
            self._status = state.get("status", "stopped")
            self._port = state.get("port", 8082)
            self._settings_path = state.get("settings_path")

            # Try to restore process if it still exists
            pid = state.get("pid")
            if pid and self._status == "running":
                try:
                    # Check if process still exists
                    os.kill(pid, 0)  # Signal 0 doesn't kill, just checks if process exists
                    # Create a mock process object with the PID
                    self._process = subprocess.Popen.__new__(subprocess.Popen)
                    self._process.pid = pid

                    def _poll() -> Optional[int]:
                        try:
                            os.kill(pid, 0)
                        except (OSError, ProcessLookupError):
                            return 0
                        return None

                    self._process.poll = _poll  # type: ignore[assignment]
                    _PRINTER_INFO.print(f"[MLX Server] Restored running server with PID: {pid}")
                except (OSError, ProcessLookupError):
                    # Process no longer exists
                    self._status = "stopped"
                    self._process = None
                    _PRINTER_INFO.print("[MLX Server] Previous server process no longer exists")

    @classmethod
    def get_instance(cls) -> 'MLXServerManager':
        """Return a process-wide singleton instance."""
        global _INSTANCE

        if _INSTANCE is not None and isinstance(_INSTANCE, cls):
            return _INSTANCE

        instance = cls()
        instance._restore_state()
        _INSTANCE = instance
        return instance

    def start_server(self, settings_path: Optional[str] = None, port: Optional[int] = None) -> Dict[str, Any]:
        """
        Start the MLX FastAPI server as a subprocess.

        Args:
            settings_path: Path to settings JSON file (optional, uses default if not provided)
            port: Port to run the server on (optional, uses default if not provided)

        Returns:
            Dict with success status and message
        """
        with self._lock:
            if self._status == "running":
                return {"success": False, "message": "MLX server is already running"}

            try:
                # Determine settings path
                if settings_path is None:
                    # Use default settings path
                    from python.helpers import files
                    self._settings_path = files.get_abs_path("tmp/settings.json")
                else:
                    self._settings_path = settings_path

                # Set port
                if port is not None:
                    self._port = port

                # Check if settings file exists
                if not os.path.exists(self._settings_path):
                    return {"success": False, "message": f"Settings file not found: {self._settings_path}"}

                # Validate settings contain MLX configuration
                with open(self._settings_path, 'r') as f:
                    settings_data = json.load(f)

                # Check MLX server enabled setting
                if not settings_data.get("mlx_server_enabled", False):
                    return {"success": False, "message": "MLX provider is not enabled in settings"}

                # Get model path from apple_mlx_model_path setting
                model_path = settings_data.get("apple_mlx_model_path", "")
                if not model_path:
                    return {"success": False, "message": "MLX model path is not configured"}

                if not os.path.exists(model_path):
                    return {"success": False, "message": f"MLX model path does not exist: {model_path}"}

                # Check if the port is available before starting
                if not self._check_port_available(self._port):
                    if self._check_server_health():
                        self._status = "running"
                        self._process = None
                        self._save_state()
                        _PRINTER_INFO.print(
                            f"[MLX Server] Detected existing server on port {self._port}; "
                            "adopting unmanaged instance."
                        )
                        return {
                            "success": True,
                            "message": f"MLX server already running on port {self._port}"
                        }
                    return {
                        "success": False,
                        "message": (
                            f"Port {self._port} is already in use. Please stop any existing "
                            "MLX server or choose a different port."
                        ),
                    }

                # Start the server subprocess
                cmd = [
                    sys.executable,
                    "-m",
                    "python.models.apple_mlx_provider",
                    self._settings_path,
                ]

                _PRINTER_INFO.print(f"[MLX Server] Starting server with command: {' '.join(cmd)}")

                # Start as subprocess with proper environment
                env = os.environ.copy()
                env["PYTHONPATH"] = os.getcwd()

                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=os.getcwd(),
                    preexec_fn=os.setsid  # Create new process group for better signal handling
                )

                self._status = "starting"
                self._save_state()
                _PRINTER_INFO.print(f"[MLX Server] Server process started with PID: {self._process.pid}")
                try:
                    self._external_pid_file.parent.mkdir(parents=True, exist_ok=True)
                    if self._process and self._process.pid:
                        self._external_pid_file.write_text(str(self._process.pid))
                except Exception as e:
                    _PRINTER_WARN.print(f"[MLX Server] Warning: unable to write PID file: {e}")

                # Wait for server to start and model to load (MLX models can take time)
                _PRINTER_INFO.print("[MLX Server] Waiting for server to initialize and load model...")
                time.sleep(15)  # Increased from 2 to 15 seconds for MLX model loading

                # Check if server is actually running by testing health endpoint
                if self._check_server_health():
                    self._status = "running"
                    self._save_state()
                    _PRINTER_INFO.print("[MLX Server] Server started successfully")
                    return {"success": True, "message": "MLX server started successfully"}
                else:
                    # Server didn't start properly, clean up
                    self._cleanup_process()
                    self._status = "stopped"
                    self._save_state()
                    return {"success": False, "message": "MLX server failed to start - health check failed"}

            except Exception as e:
                self._status = "stopped"
                _PRINTER_ERROR.print(f"[MLX Server] Error starting server: {e}")
                return {"success": False, "message": f"Failed to start MLX server: {str(e)}"}

    def stop_server(self) -> Dict[str, Any]:
        """
        Stop the MLX FastAPI server subprocess.

        Returns:
            Dict with success status and message
        """
        with self._lock:
            if self._status != "running":
                return {"success": False, "message": "MLX server is not running"}

            try:
                _PRINTER_INFO.print("[MLX Server] Stopping server...")

                managed_process = self._process and self._process.poll() is None
                target_pid: Optional[int] = None

                if managed_process and self._process:
                    target_pid = self._process.pid
                    try:
                        os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                        for _ in range(10):
                            if self._process.poll() is not None:
                                break
                            time.sleep(1)
                    except (OSError, ProcessLookupError):
                        pass  # Process might have already exited

                    if self._process and self._process.poll() is None:
                        _PRINTER_WARN.print("[MLX Server] Force killing server process...")
                        try:
                            os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                            self._process.wait(timeout=5)
                        except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
                            pass
                else:
                    target_pid = self._find_external_pid()
                    if target_pid is None:
                        message = "MLX server process is not managed and PID could not be determined"
                        _PRINTER_WARN.print(f"[MLX Server] {message}")
                        return {"success": False, "message": message}

                    if not self._terminate_pid(target_pid):
                        message = f"Failed to terminate external MLX server process {target_pid}"
                        _PRINTER_WARN.print(f"[MLX Server] {message}")
                        return {"success": False, "message": message}

                if target_pid is not None:
                    for _ in range(5):
                        if self._check_port_available(self._port):
                            break
                        time.sleep(1)
                    else:
                        message = f"Port {self._port} is still in use after stop attempt"
                        _PRINTER_WARN.print(f"[MLX Server] {message}")
                        return {"success": False, "message": message}

                self._cleanup_process()
                self._status = "stopped"
                self._save_state()
                _PRINTER_INFO.print("[MLX Server] Server stopped successfully")
                return {"success": True, "message": "MLX server stopped successfully"}

            except Exception as e:
                _PRINTER_ERROR.print(f"[MLX Server] Error stopping server: {e}")
                self._cleanup_process()
                self._status = "stopped"
                return {"success": False, "message": f"Error stopping MLX server: {str(e)}"}

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the MLX server.

        Returns:
            Dict with status information
        """
        with self._lock:
            # Refresh persisted state when no active process is tracked
            if not self._process:
                self._restore_state()

            managed_process = False

            if self._process:
                if self._process.poll() is None:
                    managed_process = True
                else:
                    # Process handle is stale, clean it up
                    self._cleanup_process()

            # Always probe server health so externally launched instances are detected
            healthy = self._check_server_health()

            previous_status = self._status

            if managed_process:
                current_status = "running" if healthy else "unhealthy"
            else:
                current_status = "running" if healthy else "stopped"

            self._status = current_status

            if self._status != previous_status:
                self._save_state()

            return {
                "status": current_status,
                "pid": self._process.pid if managed_process and self._process else None,
                "port": self._port,
                "settings_path": self._settings_path,
                "managed": managed_process,
            }

    def _check_port_available(self, port: int) -> bool:
        """
        Check if a port is available for binding.

        Returns:
            True if port is available, False if already in use
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', port))
                return result != 0  # 0 means connection successful (port in use)
        except Exception:
            return False

    def _check_server_health(self) -> bool:
        """
        Check if the MLX server is healthy by calling the health endpoint.

        Returns:
            True if server is healthy, False otherwise
        """
        if not HAS_HTTPX:
            return False

        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"http://127.0.0.1:{self._port}/healthz")
                return response.status_code == 200
        except Exception:
            return False

    def _cleanup_process(self):
        """Clean up the process reference."""
        if self._process:
            try:
                if self._process.stdout:
                    self._process.stdout.close()
                if self._process.stderr:
                    self._process.stderr.close()
            except Exception:
                pass
            self._process = None
        try:
            if self._external_pid_file.exists():
                self._external_pid_file.unlink()
        except Exception:
            pass

    def _find_external_pid(self) -> Optional[int]:
        """Locate a PID for an MLX server started outside this manager."""
        state = self._load_state(ignore_expiry=True)
        pid = state.get("pid") if state else None
        if pid and self._pid_is_alive(pid):
            return pid

        if self._external_pid_file.exists():
            try:
                raw = self._external_pid_file.read_text().strip()
                if raw:
                    pid = int(raw)
                    if self._pid_is_alive(pid):
                        return pid
            except Exception:
                pass

        return None

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _terminate_pid(self, pid: int) -> bool:
        """Attempt to terminate an external PID and confirm it exited."""
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            return True

        for _ in range(10):
            if not self._pid_is_alive(pid):
                break
            time.sleep(1)
        else:
            try:
                os.kill(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                return True
            for _ in range(5):
                if not self._pid_is_alive(pid):
                    break
                time.sleep(1)
            else:
                return False

        return True

# Convenience functions
def start_mlx_server(settings_path: Optional[str] = None) -> Dict[str, Any]:
    """Start the MLX server."""
    return MLXServerManager.get_instance().start_server(settings_path)


def stop_mlx_server() -> Dict[str, Any]:
    """Stop the MLX server."""
    return MLXServerManager.get_instance().stop_server()


def get_mlx_server_status() -> Dict[str, Any]:
    """Get MLX server status."""
    return MLXServerManager.get_instance().get_status()


def is_mlx_server_running() -> bool:
    """Check if MLX server is running."""
    status = get_mlx_server_status()
    return status["status"] == "running"
