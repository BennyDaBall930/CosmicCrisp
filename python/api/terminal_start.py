"""
Terminal Start API Handler
Creates a new terminal session for macOS native execution.
"""

from python.helpers.api import ApiHandler, Request, Response
from python.services.terminal_manager import get_manager, TerminalConfig
from python.helpers.print_style import PrintStyle
import os


class TerminalStart(ApiHandler):
    @classmethod
    def requires_loopback(cls) -> bool:
        return True

    @classmethod
    def requires_csrf(cls) -> bool:
        # CLI agent authenticates via session cookie but doesn't set the CSRF cookie; header is sufficient.
        # To avoid 403s from non-browser clients on localhost, disable CSRF here and rely on loopback + optional basic auth.
        return False
    async def process(self, input: dict, request: Request) -> dict | Response:
        """
        Create a new terminal session.
        
        Expected input:
        {
            "cwd": "/path/to/dir",  # optional
            "cols": 120,  # optional
            "rows": 40,  # optional
            "env": {}  # optional environment variables
        }
        """
        try:
            # Get terminal manager
            manager = get_manager()
            
            # Parse input parameters
            cwd = input.get("cwd", os.path.expanduser("~"))
            cols = input.get("cols", 120)
            rows = input.get("rows", 40)
            env = input.get("env", {})
            
            # Create session
            session_id = manager.create_session(
                cwd=cwd,
                env=env,
                rows=rows,
                cols=cols
            )
            
            # Log creation
            PrintStyle(
                background_color="#2E7D32",
                font_color="white",
                bold=True,
                padding=True
            ).print(f"Terminal session created: {session_id}")
            
            return {
                "success": True,
                "session_id": session_id,
                "cwd": cwd,
                "rows": rows,
                "cols": cols
            }
            
        except Exception as e:
            PrintStyle(
                background_color="#C62828",
                font_color="white",
                bold=True,
                padding=True
            ).print(f"Failed to create terminal session: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
