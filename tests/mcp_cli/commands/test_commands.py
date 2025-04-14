import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import json
from io import StringIO

# Import the modules to test
from mcp_cli.commands import resources, prompts, ping

# Create a pytest fixture for the StreamManager mock
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

# Test for resources_list command
@pytest.mark.asyncio
async def test_resources_list(mock_stream_manager, monkeypatch, capsys):
    """Test the resources_list command with the StreamManager."""
    
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
        
        # Check that the "No resources" message appears for the second server
        assert "No resources available" in captured.out

# Test for prompts_list command
@pytest.mark.asyncio
async def test_prompts_list(mock_stream_manager, monkeypatch, capsys):
    """Test the prompts_list command with the StreamManager."""
    
    # Mock the send_prompts_list function
    async def mock_send_prompts_list(r_stream, w_stream):
        if r_stream == mock_stream_manager.streams[0][0]:
            return {
                "prompts": ["prompt1", "prompt2", "prompt3"]
            }
        else:
            return {"prompts": []}
    
    with patch("mcp_cli.commands.prompts.send_prompts_list", 
               new=mock_send_prompts_list):
        # Run the command
        await prompts.prompts_list(mock_stream_manager)
        
        # Capture and check output
        captured = capsys.readouterr()
        
        # Check that all servers are mentioned in the output
        assert "TestServer1" in captured.out
        assert "TestServer2" in captured.out
        assert "FailedServer" in captured.out
        
        # Check that prompt data is present
        assert "prompt1" in captured.out
        assert "prompt2" in captured.out
        assert "prompt3" in captured.out
        
        # Check that the "No prompts" message appears for the second server
        assert "No prompts available" in captured.out

# Test for ping_run command
@pytest.mark.asyncio
async def test_ping_run(mock_stream_manager, monkeypatch, capsys):
    """Test the ping_run command with the StreamManager."""
    
    # Mock the send_ping function
    async def mock_send_ping(r_stream, w_stream):
        # First server responds successfully, second one fails
        return r_stream == mock_stream_manager.streams[0][0]
    
    with patch("mcp_cli.commands.ping.send_ping", 
               new=mock_send_ping):
        # Run the command
        await ping.ping_run(mock_stream_manager)
        
        # Capture and check output
        captured = capsys.readouterr()
        
        # Check that all servers are mentioned in the output
        assert "TestServer1" in captured.out
        assert "TestServer2" in captured.out
        assert "FailedServer" in captured.out
        
        # Check for success and failure messages
        assert "TestServer1 is up!" in captured.out
        assert "TestServer2 failed to respond" in captured.out
        assert "failed to initialize" in captured.out

# Test handling of server not found in streams map
@pytest.mark.asyncio
async def test_ping_server_not_in_map(mock_stream_manager, monkeypatch, capsys):
    """Test the ping_run command when a server is not found in the streams map."""
    
    # Add a server to server_info that's not in the streams map
    mock_stream_manager.get_server_info.return_value.append({
        "id": 4, 
        "name": "MissingServer", 
        "tools": 1, 
        "status": "Connected",
        "tool_start_index": 6
    })
    
    # Mock the send_ping function
    async def mock_send_ping(r_stream, w_stream):
        return True  # This should not be called for the missing server
    
    with patch("mcp_cli.commands.ping.send_ping", 
               new=mock_send_ping):
        # Run the command
        await ping.ping_run(mock_stream_manager)
        
        # Capture and check output
        captured = capsys.readouterr()
        
        # Check that the missing server is mentioned in the output
        assert "MissingServer" in captured.out
        assert "not found in stream map" in captured.out

# Test handling of failed streams in resources_list command
@pytest.mark.asyncio
async def test_resources_list_with_failed_stream(mock_stream_manager, monkeypatch, capsys):
    """Test the resources_list command with a stream that raises an exception."""
    
    # Mock the send_resources_list function to raise an exception for the second server
    async def mock_send_resources_list(r_stream, w_stream):
        if r_stream == mock_stream_manager.streams[0][0]:
            return {
                "resources": [{"name": "resource1", "type": "file"}]
            }
        else:
            raise Exception("Connection lost")
    
    with patch("mcp_cli.commands.resources.send_resources_list", 
               new=mock_send_resources_list):
        # Run the command (should not raise the exception)
        await resources.resources_list(mock_stream_manager)
        
        # Capture and check output
        captured = capsys.readouterr()
        
        # Check that the successful server data is still present
        assert "resource1" in captured.out
        assert "TestServer1" in captured.out