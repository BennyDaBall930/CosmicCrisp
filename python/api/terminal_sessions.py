"""
Terminal Sessions API Handler
Lists all active terminal sessions.
"""

from python.helpers.api import ApiHandler, Request, Response
from python.services.terminal_manager import get_manager
from python.helpers.print_style import PrintStyle


class TerminalSessions(ApiHandler):
    @classmethod
    def requires_loopback(cls) -> bool:
        return True

    @classmethod
    def requires_csrf(cls) -> bool:
        return False
    async def process(self, input: dict, request: Request) -> dict | Response:
        """
        List all active terminal sessions.
        
        Returns:
        {
            "success": true,
            "sessions": {
                "pty_123456": {
                    "cwd": "/home/user",
                    "started_at": "2024-01-01T12:00:00",
                    "last_activity": "2024-01-01T12:05:00",
                    "idle_seconds": 300,
                    "status": "running",
                    "rows": 40,
                    "cols": 120
                }
            },
            "count": 1
        }
        """
        try:
            # Get terminal manager
            manager = get_manager()
            
            # Get all sessions
            sessions = manager.list_sessions()
            
            PrintStyle(
                background_color="#1976D2",
                font_color="white",
                bold=True,
                padding=True
            ).print(f"Found {len(sessions)} active terminal sessions")
            
            return {
                "success": True,
                "sessions": sessions,
                "count": len(sessions)
            }
            
        except Exception as e:
            PrintStyle(
                background_color="#C62828",
                font_color="white",
                bold=True,
                padding=True
            ).print(f"Failed to list terminal sessions: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
