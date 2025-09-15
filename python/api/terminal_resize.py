"""
Terminal Resize API Handler
Resizes a terminal session window.
"""

from python.helpers.api import ApiHandler, Request, Response
from python.services.terminal_manager import get_manager
from python.helpers.print_style import PrintStyle


class TerminalResize(ApiHandler):
    @classmethod
    def requires_loopback(cls) -> bool:
        return True

    @classmethod
    def requires_csrf(cls) -> bool:
        return False
    async def process(self, input: dict, request: Request) -> dict | Response:
        """
        Resize a terminal session.
        
        Expected input:
        {
            "session_id": "pty_123456",
            "rows": 50,
            "cols": 150
        }
        
        Returns:
        {
            "success": true,
            "message": "Terminal resized to 50x150"
        }
        """
        try:
            # Get terminal manager
            manager = get_manager()
            
            # Parse input parameters
            session_id = input.get("session_id")
            rows = input.get("rows", 40)
            cols = input.get("cols", 120)
            
            if not session_id:
                return {
                    "success": False,
                    "error": "session_id is required"
                }
            
            # Validate dimensions
            if not (10 <= rows <= 200 and 40 <= cols <= 300):
                return {
                    "success": False,
                    "error": "Invalid dimensions. Rows must be 10-200, cols must be 40-300"
                }
            
            # Resize terminal
            success = manager.resize_session(session_id, rows, cols)
            
            if success:
                PrintStyle(
                    background_color="#2E7D32",
                    font_color="white",
                    bold=True,
                    padding=True
                ).print(f"Terminal {session_id} resized to {rows}x{cols}")
                
                return {
                    "success": True,
                    "message": f"Terminal resized to {rows}x{cols}"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to resize terminal. Session may not exist."
                }
                
        except Exception as e:
            PrintStyle(
                background_color="#C62828",
                font_color="white",
                bold=True,
                padding=True
            ).print(f"Failed to resize terminal: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
