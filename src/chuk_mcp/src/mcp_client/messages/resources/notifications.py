# chuk_mcp/mcp_client/messages/resources/notifications.py
"""
Module for handling resource-related notifications in the Model Context Protocol.
"""
from typing import Dict, Any, Callable, Awaitable

async def handle_resources_list_changed_notification(
    callback: Callable[[], Awaitable[None]],
    notification: Dict[str, Any]
) -> None:
    """
    Handle a notification that the resources list has changed.
    
    Args:
        callback: Async function to call when the notification is received
        notification: The notification message
        
    Returns:
        None
    """
    # Verify it's the correct notification type
    if notification.get("method") != "notifications/resources/list_changed":
        return
    
    # Call the provided callback function
    await callback()


async def handle_resources_updated_notification(
    callback: Callable[[str], Awaitable[None]],
    notification: Dict[str, Any]
) -> None:
    """
    Handle a notification that a specific resource has been updated.
    
    Args:
        callback: Async function to call with the resource URI when the notification is received
        notification: The notification message
        
    Returns:
        None
    """
    # Verify it's the correct notification type
    if notification.get("method") != "notifications/resources/updated":
        return
    
    # Extract the URI from the notification parameters
    params = notification.get("params", {})
    uri = params.get("uri")
    
    if uri:
        # Call the provided callback function with the URI
        await callback(uri)