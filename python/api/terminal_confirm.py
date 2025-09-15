"""
Terminal Confirm API Handler
Confirms and executes permission-gated commands.
"""

from python.helpers.api import ApiHandler, Request, Response
from python.services.terminal_manager import get_manager
from python.helpers.print_style import PrintStyle


class TerminalConfirm(ApiHandler):
    @classmethod
    def requires_loopback(cls) -> bool:
        return True

    @classmethod
    def requires_csrf(cls) -> bool:
        return False
    async def process(self, input: dict, request: Request) -> dict | Response:
        """
        Confirm a permission request to execute a gated command.
        
        Expected input:
        {
            "token": "uuid-token-from-permission-request"
        }
        
        Returns:
        {
            "success": true,
            "message": "Permission granted and command executed"
        }
        """
        try:
            # Get terminal manager
            manager = get_manager()
            
            # Parse input parameters
            token = input.get("token")
            
            if not token:
                return {
                    "success": False,
                    "error": "token is required"
                }
            
            # Confirm permission
            confirmed = manager.confirm_permission(token)
            
            if confirmed:
                PrintStyle(
                    background_color="#2E7D32",
                    font_color="white",
                    bold=True,
                    padding=True
                ).print(f"Permission granted for token: {token[:8]}...")
                
                return {
                    "success": True,
                    "message": "Permission granted and command executed"
                }
            else:
                PrintStyle(
                    background_color="#C62828",
                    font_color="white",
                    bold=True,
                    padding=True
                ).print(f"Permission token invalid or expired: {token[:8]}...")
                
                return {
                    "success": False,
                    "error": "Invalid or expired token"
                }
                
        except Exception as e:
            PrintStyle(
                background_color="#C62828",
                font_color="white",
                bold=True,
                padding=True
            ).print(f"Failed to confirm permission: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
