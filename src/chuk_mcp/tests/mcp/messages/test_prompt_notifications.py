# tests/mcp/test_prompt_notifications.py
import pytest
import anyio
from unittest.mock import AsyncMock

from chuk_mcp.mcp_client.messages.message_method import MessageMethod
from chuk_mcp.mcp_client.messages.prompts.notifications import handle_prompts_list_changed_notification

# Force asyncio only for all tests in this file
pytestmark = [pytest.mark.asyncio]


async def test_handle_prompts_list_changed_notification():
    """Test handling prompts/list_changed notification"""
    # Create a mock callback function
    mock_callback = AsyncMock()
    
    # Create a sample notification
    notification = {
        "jsonrpc": "2.0",
        "method": MessageMethod.NOTIFICATION_PROMPTS_LIST_CHANGED
    }
    
    # Call the handler
    await handle_prompts_list_changed_notification(mock_callback, notification)
    
    # Verify the callback was called
    mock_callback.assert_called_once()


async def test_handle_prompts_list_changed_notification_wrong_method():
    """Test the handler ignores notifications with wrong method"""
    # Create a mock callback function
    mock_callback = AsyncMock()
    
    # Create a notification with wrong method
    notification = {
        "jsonrpc": "2.0",
        "method": "some_other_notification"
    }
    
    # Call the handler
    await handle_prompts_list_changed_notification(mock_callback, notification)
    
    # Verify the callback was NOT called
    mock_callback.assert_not_called()


async def test_handle_prompts_list_changed_notification_client_integration():
    """Test integration of notification handler with client behavior"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    
    # Create a flag to track if notification was handled
    notification_handled = False
    
    # Define an async callback that updates the flag
    async def on_prompts_list_changed():
        nonlocal notification_handled
        notification_handled = True
    
    # Create a notification
    notification = {
        "jsonrpc": "2.0",
        "method": MessageMethod.NOTIFICATION_PROMPTS_LIST_CHANGED
    }
    
    # Send the notification
    await read_send.send(notification)
    
    # Process the notification
    received_notification = await read_receive.receive()
    await handle_prompts_list_changed_notification(on_prompts_list_changed, received_notification)
    
    # Verify the flag was set
    assert notification_handled is True