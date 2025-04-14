import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import json
from io import StringIO

# Import the module to test
from mcp_cli.commands import ping

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
async def test_ping_basic(mock_stream_manager, capsys):
    """Test the ping_run command pings servers correctly."""
    
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

@pytest.mark.asyncio
async def test_ping_all_up(mock_stream_manager, capsys):
    """Test the ping_run command when all servers are up."""
    
    # Mock the send_ping function to return True for all servers
    async def mock_send_ping(r_stream, w_stream):
        return True
    
    with patch("mcp_cli.commands.ping.send_ping", 
               new=mock_send_ping):
        # Run the command
        await ping.ping_run(mock_stream_manager)
        
        # Capture and check output
        captured = capsys.readouterr()
        
        # Check that both servers show as up
        assert "TestServer1 is up!" in captured.out
        assert "TestServer2 is up!" in captured.out
        assert "failed to respond" not in captured.out

@pytest.mark.asyncio
async def test_ping_all_down(mock_stream_manager, capsys):
    """Test the ping_run command when all servers are down."""
    
    # Mock the send_ping function to return False for all servers
    async def mock_send_ping(r_stream, w_stream):
        return False
    
    with patch("mcp_cli.commands.ping.send_ping", 
               new=mock_send_ping):
        # Run the command
        await ping.ping_run(mock_stream_manager)
        
        # Capture and check output
        captured = capsys.readouterr()
        
        # Check that both servers show as down
        assert "TestServer1 failed to respond" in captured.out
        assert "TestServer2 failed to respond" in captured.out
        assert "is up!" not in captured.out

@pytest.mark.asyncio
async def test_ping_error_handling(mock_stream_manager, capsys):
    """Test the ping_run command handles errors gracefully."""
    
    # Mock the send_ping function to raise an exception
    async def mock_send_ping(r_stream, w_stream):
        if r_stream == mock_stream_manager.streams[0][0]:
            return True
        else:
            raise Exception("Test error")
    
    with patch("mcp_cli.commands.ping.send_ping", 
               new=mock_send_ping):
        # Run the command (should not propagate the exception)
        await ping.ping_run(mock_stream_manager)
        
        # Capture and check output
        captured = capsys.readouterr()
        
        # Check that the first server shows as up
        assert "TestServer1 is up!" in captured.out

@pytest.mark.asyncio
async def test_ping_no_servers(capsys):
    """Test the ping_run command when no servers are available."""
    
    # Create a StreamManager with no servers
    empty_manager = MagicMock()
    empty_manager.get_server_info.return_value = []
    empty_manager.server_streams_map = {}
    empty_manager.streams = []
    
    # Mock the send_ping function (should not be called)
    async def mock_send_ping(r_stream, w_stream):
        # This should never be called
        assert False, "send_ping should not be called with no servers"
        return False
    
    with patch("mcp_cli.commands.ping.send_ping", 
               new=mock_send_ping):
        # Run the command
        await ping.ping_run(empty_manager)
        
        # Capture and check output
        captured = capsys.readouterr()
        
        # Check that the pinging message is still displayed
        assert "Pinging Servers" in captured.out