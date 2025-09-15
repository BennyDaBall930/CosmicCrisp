"""
Terminal Write API Handler
Writes data to a terminal session with permission gating.
"""

from python.helpers.api import ApiHandler, Request, Response
from python.services.terminal_manager import get_manager
from python.helpers.print_style import PrintStyle


class TerminalWrite(ApiHandler):
    @classmethod
    def requires_loopback(cls) -> bool:
        return True

    @classmethod
    def requires_csrf(cls) -> bool:
        return False
    async def process(self, input: dict, request: Request) -> dict | Response:
        """
        Write data to a terminal session.
        
        Expected input:
        {
            "session_id": "pty_123456",
            "data": "ls -la\\n"
        }
        
        Returns:
        {
            "success": true,
            "status": "success" | "permission_required",
            "token": "uuid-token",  # if permission required
            "command": "sudo rm -rf /",  # if permission required
            "reason": "This command requires...",  # if permission required
            "expires_in": 30  # if permission required
        }
        """
        try:
            # Get terminal manager
            manager = get_manager()
            
            # Parse input parameters
            session_id = input.get("session_id")
            data = input.get("data", "")
            
            if not session_id:
                return {
                    "success": False,
                    "error": "session_id is required"
                }
            
            # Write to session
            result = manager.write_to_session(session_id, data)
            
            # Check for errors
            if "error" in result:
                PrintStyle(
                    background_color="#C62828",
                    font_color="white",
                    bold=True,
                    padding=True
                ).print(f"Terminal write error: {result['error']}")
                
                return {
                    "success": False,
                    "error": result["error"]
                }
            
            # Check if permission is required
            if result.get("status") == "permission_required":
                PrintStyle(
                    background_color="#FF9800",
                    font_color="white",
                    bold=True,
                    padding=True
                ).print(f"Permission required for: {result['command']}")
                
                return {
                    "success": True,
                    "status": "permission_required",
                    "token": result["token"],
                    "command": result["command"],
                    "reason": result["reason"],
                    "expires_in": result["expires_in"]
                }
            
            # Success - data written
            return {
                "success": True,
                "status": "success"
            }
            
        except Exception as e:
            PrintStyle(
                background_color="#C62828",
                font_color="white",
                bold=True,
                padding=True
            ).print(f"Failed to write to terminal: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
