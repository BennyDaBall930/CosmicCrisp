"""
Terminal Settings API Handler
Manages terminal configuration settings.
"""

from python.helpers.api import ApiHandler, Request, Response
from python.services.terminal_manager import get_manager, TerminalConfig
from python.helpers.print_style import PrintStyle
import os


class TerminalSettings(ApiHandler):
    @classmethod
    def requires_loopback(cls) -> bool:
        return True

    @classmethod
    def requires_csrf(cls) -> bool:
        return False
    async def process(self, input: dict, request: Request) -> dict | Response:
        """
        Get or update terminal settings.
        
        Expected input for GET:
        {
            "action": "get"
        }
        
        Expected input for UPDATE:
        {
            "action": "update",
            "settings": {
                "shell_path": "/bin/zsh",
                "rows": 40,
                "cols": 120,
                "permission_timeout_seconds": 30,
                "idle_timeout_seconds": 300,
                "max_sessions": 10
            }
        }
        
        Returns:
        {
            "success": true,
            "settings": { ... current settings ... }
        }
        """
        try:
            # Get terminal manager
            manager = get_manager()
            
            # Parse action
            action = input.get("action", "get")
            
            if action == "get":
                # Get current settings
                config = manager.config
                settings = {
                    "shell_path": config.shell_path,
                    "rows": config.rows,
                    "cols": config.cols,
                    "permission_timeout_seconds": config.permission_timeout_seconds,
                    "idle_timeout_seconds": config.idle_timeout_seconds,
                    "max_sessions": config.max_sessions,
                    "working_dir": config.working_dir or os.path.expanduser("~")
                }
                
                PrintStyle(
                    background_color="#1976D2",
                    font_color="white",
                    bold=True,
                    padding=True
                ).print("Terminal settings retrieved")
                
                return {
                    "success": True,
                    "settings": settings
                }
                
            elif action == "update":
                # Update settings
                new_settings = input.get("settings", {})
                
                if not new_settings:
                    return {
                        "success": False,
                        "error": "settings object is required for update"
                    }
                
                # Update configuration
                config = manager.config
                
                # Update allowed fields
                if "shell_path" in new_settings:
                    shell_path = new_settings["shell_path"]
                    # Validate shell exists
                    if not os.path.exists(shell_path):
                        return {
                            "success": False,
                            "error": f"Shell not found: {shell_path}"
                        }
                    config.shell_path = shell_path
                    
                if "rows" in new_settings:
                    rows = new_settings["rows"]
                    if not (10 <= rows <= 200):
                        return {
                            "success": False,
                            "error": "Rows must be between 10 and 200"
                        }
                    config.rows = rows
                    
                if "cols" in new_settings:
                    cols = new_settings["cols"]
                    if not (40 <= cols <= 300):
                        return {
                            "success": False,
                            "error": "Cols must be between 40 and 300"
                        }
                    config.cols = cols
                    
                if "permission_timeout_seconds" in new_settings:
                    timeout = new_settings["permission_timeout_seconds"]
                    if not (5 <= timeout <= 300):
                        return {
                            "success": False,
                            "error": "Permission timeout must be between 5 and 300 seconds"
                        }
                    config.permission_timeout_seconds = timeout
                    
                if "idle_timeout_seconds" in new_settings:
                    idle = new_settings["idle_timeout_seconds"]
                    if not (60 <= idle <= 3600):
                        return {
                            "success": False,
                            "error": "Idle timeout must be between 60 and 3600 seconds"
                        }
                    config.idle_timeout_seconds = idle
                    
                if "max_sessions" in new_settings:
                    max_sessions = new_settings["max_sessions"]
                    if not (1 <= max_sessions <= 50):
                        return {
                            "success": False,
                            "error": "Max sessions must be between 1 and 50"
                        }
                    config.max_sessions = max_sessions
                    
                if "working_dir" in new_settings:
                    working_dir = new_settings["working_dir"]
                    if working_dir and not os.path.exists(working_dir):
                        return {
                            "success": False,
                            "error": f"Working directory not found: {working_dir}"
                        }
                    config.working_dir = working_dir
                
                PrintStyle(
                    background_color="#2E7D32",
                    font_color="white",
                    bold=True,
                    padding=True
                ).print("Terminal settings updated")
                
                # Return updated settings
                return {
                    "success": True,
                    "settings": {
                        "shell_path": config.shell_path,
                        "rows": config.rows,
                        "cols": config.cols,
                        "permission_timeout_seconds": config.permission_timeout_seconds,
                        "idle_timeout_seconds": config.idle_timeout_seconds,
                        "max_sessions": config.max_sessions,
                        "working_dir": config.working_dir or os.path.expanduser("~")
                    }
                }
                
            else:
                return {
                    "success": False,
                    "error": f"Invalid action: {action}. Use 'get' or 'update'"
                }
                
        except Exception as e:
            PrintStyle(
                background_color="#C62828",
                font_color="white",
                bold=True,
                padding=True
            ).print(f"Failed to process terminal settings: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
