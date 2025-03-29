# tests/mcp/test_resource_notifications.py
import pytest
import anyio
from unittest.mock import AsyncMock

from mcp_client.messages.resources.notifications import (
    handle_resources_list_changed_notification,
    handle_resources_updated_notification
)

# Force asyncio only for all tests in this file
pytestmark = [pytest.mark.asyncio]


async def test_handle_resources_list_changed_notification():
    """Test handling resources/list_changed notification"""
    # Create a mock callback function
    mock_callback = AsyncMock()
    
    # Create a sample notification
    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/resources/list_changed"
    }
    
    # Call the handler
    await handle_resources_list_changed_notification(mock_callback, notification)
    
    # Verify the callback was called
    mock_callback.assert_called_once()


async def test_handle_resources_list_changed_notification_wrong_method():
    """Test the handler ignores notifications with wrong method"""
    # Create a mock callback function
    mock_callback = AsyncMock()
    
    # Create a notification with wrong method
    notification = {
        "jsonrpc": "2.0",
        "method": "some_other_notification"
    }
    
    # Call the handler
    await handle_resources_list_changed_notification(mock_callback, notification)
    
    # Verify the callback was NOT called
    mock_callback.assert_not_called()


async def test_handle_resources_updated_notification():
    """Test handling resources/updated notification"""
    # Create a mock callback function
    mock_callback = AsyncMock()
    
    # Sample URI that was updated
    test_uri = "file:///project/src/main.rs"
    
    # Create a sample notification
    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/resources/updated",
        "params": {
            "uri": test_uri
        }
    }
    
    # Call the handler
    await handle_resources_updated_notification(mock_callback, notification)
    
    # Verify the callback was called with the correct URI
    mock_callback.assert_called_once_with(test_uri)


async def test_handle_resources_updated_notification_wrong_method():
    """Test the handler ignores notifications with wrong method"""
    # Create a mock callback function
    mock_callback = AsyncMock()
    
    # Create a notification with wrong method
    notification = {
        "jsonrpc": "2.0",
        "method": "some_other_notification",
        "params": {
            "uri": "file:///some/file.txt"
        }
    }
    
    # Call the handler
    await handle_resources_updated_notification(mock_callback, notification)
    
    # Verify the callback was NOT called
    mock_callback.assert_not_called()


async def test_handle_resources_updated_notification_missing_uri():
    """Test handling resources/updated notification with missing URI"""
    # Create a mock callback function
    mock_callback = AsyncMock()
    
    # Create a notification with missing URI
    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/resources/updated",
        "params": {}  # No URI provided
    }
    
    # Call the handler
    await handle_resources_updated_notification(mock_callback, notification)
    
    # Verify the callback was NOT called
    mock_callback.assert_not_called()


async def test_resources_notification_client_integration():
    """Test integration of notification handlers with client behavior"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    
    # Track notifications received
    list_changed_received = False
    updated_uri = None
    
    # Define async callbacks
    async def on_list_changed():
        nonlocal list_changed_received
        list_changed_received = True
    
    async def on_resource_updated(uri):
        nonlocal updated_uri
        updated_uri = uri
    
    # Create notifications
    list_changed_notification = {
        "jsonrpc": "2.0",
        "method": "notifications/resources/list_changed"
    }
    
    resource_updated_notification = {
        "jsonrpc": "2.0",
        "method": "notifications/resources/updated",
        "params": {
            "uri": "file:///updated_file.txt"
        }
    }
    
    # Send notifications
    await read_send.send(list_changed_notification)
    await read_send.send(resource_updated_notification)
    
    # Process notifications
    received_notification = await read_receive.receive()
    await handle_resources_list_changed_notification(on_list_changed, received_notification)
    
    received_notification = await read_receive.receive()
    await handle_resources_updated_notification(on_resource_updated, received_notification)
    
    # Verify notifications were handled correctly
    assert list_changed_received is True
    assert updated_uri == "file:///updated_file.txt"