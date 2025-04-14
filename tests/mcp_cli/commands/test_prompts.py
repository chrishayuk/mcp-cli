import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import json
from io import StringIO

# Import the module to test
from mcp_cli.commands import prompts

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
async def test_prompts_list_basic(mock_stream_manager, capsys):
    """Test the prompts_list command fetches prompts correctly."""
    
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

@pytest.mark.asyncio
async def test_prompts_list_empty(mock_stream_manager, capsys):
    """Test the prompts_list command when no prompts are available."""
    
    # Mock the send_prompts_list function to return empty prompts
    async def mock_send_prompts_list(r_stream, w_stream):
        return {"prompts": []}
    
    with patch("mcp_cli.commands.prompts.send_prompts_list", 
               new=mock_send_prompts_list):
        # Run the command
        await prompts.prompts_list(mock_stream_manager)
        
        # Capture and check output
        captured = capsys.readouterr()
        
        # Check that "No prompts" message appears for all servers
        assert "No prompts available" in captured.out
        assert captured.out.count("No prompts available") >= 2  # At least for 2 connected servers

@pytest.mark.asyncio
async def test_prompts_list_error_handling(mock_stream_manager, capsys):
    """Test the prompts_list command handles errors gracefully."""
    
    # Mock the send_prompts_list function to raise an exception
    async def mock_send_prompts_list(r_stream, w_stream):
        if r_stream == mock_stream_manager.streams[0][0]:
            return {"prompts": ["prompt1", "prompt2"]}
        else:
            raise Exception("Test error")
    
    with patch("mcp_cli.commands.prompts.send_prompts_list", 
               new=mock_send_prompts_list):
        # Run the command (should not propagate the exception)
        await prompts.prompts_list(mock_stream_manager)
        
        # Capture and check output
        captured = capsys.readouterr()
        
        # Check that the first server's prompts were displayed
        assert "prompt1" in captured.out
        assert "prompt2" in captured.out

@pytest.mark.asyncio
async def test_prompts_list_missing_prompts_key(mock_stream_manager, capsys):
    """Test the prompts_list command when the 'prompts' key is missing in the response."""
    
    # Mock the send_prompts_list function to return a response without the 'prompts' key
    async def mock_send_prompts_list(r_stream, w_stream):
        return {"status": "ok"}  # No 'prompts' key
    
    with patch("mcp_cli.commands.prompts.send_prompts_list", 
               new=mock_send_prompts_list):
        # Run the command
        await prompts.prompts_list(mock_stream_manager)
        
        # Capture and check output
        captured = capsys.readouterr()
        
        # Check that "No prompts" message appears for all servers
        assert "No prompts available" in captured.out
        assert captured.out.count("No prompts available") >= 2  # At least for 2 connected servers

@pytest.mark.asyncio
async def test_prompts_list_none_response(mock_stream_manager, capsys):
    """Test the prompts_list command when send_prompts_list returns None."""
    
    # Mock the send_prompts_list function to return None
    async def mock_send_prompts_list(r_stream, w_stream):
        return None
    
    with patch("mcp_cli.commands.prompts.send_prompts_list", 
               new=mock_send_prompts_list):
        # Run the command
        await prompts.prompts_list(mock_stream_manager)
        
        # Capture and check output
        captured = capsys.readouterr()
        
        # Check that "No prompts" message appears for all servers
        assert "No prompts available" in captured.out
        assert captured.out.count("No prompts available") >= 2  # At least for 2 connected servers