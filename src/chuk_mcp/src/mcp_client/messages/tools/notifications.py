# chuk_mcp/mcp_client/messages/tools/notifications.py
"""
Module for handling tool-related notifications in the Model Context Protocol.
"""
from typing import Dict, Any, Callable, Awaitable

async def handle_tools_list_changed_notification(
    callback: Callable[[], Awaitable[None]],
    notification: Dict[str, Any]
) -> None:
    """
    Handle a notification that the tools list has changed.
    
    Args:
        callback: Async function to call when the notification is received
        notification: The notification message
        
    Returns:
        None
    """
    # Verify it's the correct notification type
    if notification.get("method") != "notifications/tools/list_changed":
        return
    
    # Call the provided callback function
    await callback()