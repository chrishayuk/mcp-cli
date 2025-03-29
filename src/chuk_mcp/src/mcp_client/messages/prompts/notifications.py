# chuk_mcp/mcp_client/messages/prompts/notifications.py
"""
Module for handling prompt-related notifications in the Model Context Protocol.
"""
from typing import Dict, Any, Callable, Awaitable

# chuk_mcp imports
from mcp_client.messages.message_method import MessageMethod

async def handle_prompts_list_changed_notification(
    callback: Callable[[], Awaitable[None]],
    notification: Dict[str, Any]
) -> None:
    """
    Handle a notification that the prompts list has changed.
    
    Args:
        callback: Async function to call when the notification is received
        notification: The notification message
        
    Returns:
        None
    """
    # Verify it's the correct notification type
    if notification.get("method") != MessageMethod.NOTIFICATION_PROMPTS_LIST_CHANGED:
        return
    
    # Call the provided callback function
    await callback()