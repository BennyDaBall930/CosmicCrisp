"""
Terminal Stream API Handler
Streams terminal output via Server-Sent Events (SSE).
"""

from python.helpers.api import ApiHandler, Request, Response
from python.services.terminal_manager import get_manager
from python.helpers.print_style import PrintStyle
from flask import Response as FlaskResponse
import json
import asyncio


class TerminalStream(ApiHandler):
    @classmethod
    def requires_loopback(cls) -> bool:
        return True

    @classmethod
    def requires_csrf(cls) -> bool:
        return False
    async def process(self, input: dict, request: Request) -> dict | Response:
        """
        Stream terminal output via SSE.
        
        Expected input:
        {
            "session_id": "pty_123456"
        }
        
        Returns: SSE stream of terminal events
        """
        try:
            # Get terminal manager
            manager = get_manager()
            
            # Parse input parameters
            session_id = input.get("session_id")
            
            if not session_id:
                return {
                    "success": False,
                    "error": "session_id is required"
                }
            
            # Check if session exists
            session = manager.get_session(session_id)
            if not session:
                return {
                    "success": False,
                    "error": f"Session {session_id} not found"
                }
            
            PrintStyle(
                background_color="#1976D2",
                font_color="white",
                bold=True,
                padding=True
            ).print(f"Starting terminal stream for: {session_id}")
            
            # Create SSE generator
            async def generate():
                try:
                    async for event in manager.stream_session(session_id):
                        # Format as SSE
                        data = {
                            "type": event.type,
                            "data": event.data,
                            "timestamp": event.timestamp,
                            "session_id": event.session_id
                        }
                        
                        # Send SSE formatted message
                        yield f"data: {json.dumps(data)}\n\n"
                        
                        # Flush to ensure data is sent immediately
                        await asyncio.sleep(0)
                        
                except Exception as e:
                    # Send error event
                    error_data = {
                        "type": "error",
                        "data": str(e),
                        "session_id": session_id
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    
                finally:
                    # Send close event
                    close_data = {
                        "type": "close",
                        "data": "Stream closed",
                        "session_id": session_id
                    }
                    yield f"data: {json.dumps(close_data)}\n\n"
            
            # Return SSE response
            return FlaskResponse(
                generate(),
                mimetype="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive"
                }
            )
            
        except Exception as e:
            PrintStyle(
                background_color="#C62828",
                font_color="white",
                bold=True,
                padding=True
            ).print(f"Failed to stream terminal: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
