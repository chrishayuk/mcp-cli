# tests/mcp/test_tools_notification.py
import pytest
import anyio
from unittest.mock import AsyncMock

from mcp_client.messages.tools.notifications import handle_tools_list_changed_notification

# Force asyncio only for all tests in this file
pytestmark = [pytest.mark.asyncio]


async def test_handle_tools_list_changed_notification():
    """Test handling tools/list_changed notification"""
    # Create a mock callback function
    mock_callback = AsyncMock()
    
    # Create a sample notification
    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/tools/list_changed"
    }
    
    # Call the handler
    await handle_tools_list_changed_notification(mock_callback, notification)
    
    # Verify the callback was called
    mock_callback.assert_called_once()


async def test_handle_tools_list_changed_notification_wrong_method():
    """Test the handler ignores notifications with wrong method"""
    # Create a mock callback function
    mock_callback = AsyncMock()
    
    # Create a notification with wrong method
    notification = {
        "jsonrpc": "2.0",
        "method": "some_other_notification"
    }
    
    # Call the handler
    await handle_tools_list_changed_notification(mock_callback, notification)
    
    # Verify the callback was NOT called
    mock_callback.assert_not_called()


async def test_tools_notification_client_integration():
    """Test integration of notification handler with client behavior"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    
    # Track notifications received
    list_changed_received = False
    
    # Define async callback
    async def on_list_changed():
        nonlocal list_changed_received
        list_changed_received = True
    
    # Create notification
    list_changed_notification = {
        "jsonrpc": "2.0",
        "method": "notifications/tools/list_changed"
    }
    
    # Send notification
    await read_send.send(list_changed_notification)
    
    # Process notification
    received_notification = await read_receive.receive()
    await handle_tools_list_changed_notification(on_list_changed, received_notification)
    
    # Verify notification was handled correctly
    assert list_changed_received is True