import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import json
from io import StringIO

# Import the module to test
from mcp_cli.commands import resources

@pytest.fixture
def mock_stream_manager():
    """Create a mock StreamManager with predefined test data."""
    mock_manager = MagicMock()
    
    # Set up server info
    mock_manager.get_server_info.return_value = [
        {
            "id": 1, 
            "name": "TestServer1", 
            "tools": 3, 
            "status": "Connected",
            "tool_start_index": 0
        },
        {
            "id": 2, 
            "name": "TestServer2", 
            "tools": 2, 
            "status": "Connected",
            "tool_start_index": 3
        },
        {
            "id": 3, 
            "name": "FailedServer", 
            "tools": 0, 
            "status": "Failed to initialize",
            "tool_start_index": 5
        }
    ]
    
    # Set up server streams map
    mock_manager.server_streams_map = {
        "TestServer1": 0,
        "TestServer2": 1
    }
    
    # Set up streams
    mock_read_stream1 = AsyncMock()
    mock_write_stream1 = AsyncMock()
    mock_read_stream2 = AsyncMock()
    mock_write_stream2 = AsyncMock()
    
    mock_manager.streams = [
        (mock_read_stream1, mock_write_stream1),
        (mock_read_stream2, mock_write_stream2)
    ]
    
    return mock_manager

@pytest.mark.asyncio
async def test_resources_list_basic(mock_stream_manager, capsys):
    """Test the resources_list command fetches resources correctly."""
    
    # Mock the send_resources_list function
    async def mock_send_resources_list(r_stream, w_stream):
        if r_stream == mock_stream_manager.streams[0][0]:
            return {
                "resources": [
                    {"name": "resource1", "type": "file"},
                    {"name": "resource2", "type": "image"}
                ]
            }
        else:
            return {"resources": []}
    
    with patch("mcp_cli.commands.resources.send_resources_list", 
               new=mock_send_resources_list):
        # Run the command
        await resources.resources_list(mock_stream_manager)
        
        # Capture and check output
        captured = capsys.readouterr()
        
        # Check that all servers are mentioned in the output
        assert "TestServer1" in captured.out
        assert "TestServer2" in captured.out
        assert "FailedServer" in captured.out
        
        # Check that resource data is present
        assert "resource1" in captured.out
        assert "resource2" in captured.out
        assert "file" in captured.out
        assert "image" in captured.out
        
        # Check that the "No resources" message appears for the second server
        assert "No resources available" in captured.out

@pytest.mark.asyncio
async def test_resources_list_string_resources(mock_stream_manager, capsys):
    """Test the resources_list command with string resources instead of objects."""
    
    # Mock the send_resources_list function to return string resources
    async def mock_send_resources_list(r_stream, w_stream):
        return {
            "resources": ["resource1", "resource2", "resource3"]
        }
    
    with patch("mcp_cli.commands.resources.send_resources_list", 
               new=mock_send_resources_list):
        # Run the command
        await resources.resources_list(mock_stream_manager)
        
        # Capture and check output
        captured = capsys.readouterr()
        
        # Check that string resources are displayed correctly
        assert "- resource1" in captured.out
        assert "- resource2" in captured.out
        assert "- resource3" in captured.out

@pytest.mark.asyncio
async def test_resources_list_empty(mock_stream_manager, capsys):
    """Test the resources_list command when no resources are available."""
    
    # Mock the send_resources_list function to return empty resources
    async def mock_send_resources_list(r_stream, w_stream):
        return {"resources": []}
    
    with patch("mcp_cli.commands.resources.send_resources_list", 
               new=mock_send_resources_list):
        # Run the command
        await resources.resources_list(mock_stream_manager)
        
        # Capture and check output
        captured = capsys.readouterr()
        
        # Check that "No resources" message appears for all servers
        assert "No resources available" in captured.out
        assert captured.out.count("No resources available") >= 2  # At least for 2 connected servers

@pytest.mark.asyncio
async def test_resources_list_error_handling(mock_stream_manager, capsys):
    """Test the resources_list command handles errors gracefully."""
    
    # Mock the send_resources_list function to raise an exception
    async def mock_send_resources_list(r_stream, w_stream):
        if r_stream == mock_stream_manager.streams[0][0]:
            return {"resources": [{"name": "resource1"}]}
        else:
            raise Exception("Test error")
    
    with patch("mcp_cli.commands.resources.send_resources_list", 
               new=mock_send_resources_list):
        # Run the command (should not propagate the exception)
        await resources.resources_list(mock_stream_manager)
        
        # Capture and check output
        captured = capsys.readouterr()
        
        # Check that the first server's resources were displayed
        assert "resource1" in captured.out

@pytest.mark.asyncio
async def test_resources_list_with_custom_server_names(mock_stream_manager, capsys):
    """Test the resources_list command with custom server names."""
    
    # Create custom server_names dictionary
    server_names = {0: "CustomServer1", 1: "CustomServer2"}
    
    # Mock the send_resources_list function
    async def mock_send_resources_list(r_stream, w_stream):
        return {"resources": [{"name": "resource1"}]}
    
    with patch("mcp_cli.commands.resources.send_resources_list", 
               new=mock_send_resources_list):
        # Run the command with custom server names
        await resources.resources_list(mock_stream_manager, server_names)
        
        # Capture and check output
        captured = capsys.readouterr()
        
        # Custom names should NOT override the names from server_info
        # The StreamManager should always use the names from server_info
        assert "TestServer1" in captured.out
        assert "TestServer2" in captured.out
        assert "CustomServer1" not in captured.out
        assert "CustomServer2" not in captured.out