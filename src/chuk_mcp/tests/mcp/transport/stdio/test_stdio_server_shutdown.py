# tests/mcp/transport/stdio/test_stdio_server_shutdown.py
import pytest
import anyio
import logging
from unittest.mock import AsyncMock, patch, MagicMock
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

from chuk_mcp.mcp_client.transport.stdio.stdio_server_shutdown import shutdown_stdio_server

# Force asyncio only for all tests in this file
pytestmark = [pytest.mark.asyncio]

# Skip all tests in this file if we can't import the required module
pytest.importorskip("mcp.transport.stdio.stdio_server_shutdown")


class MockProcess:
    """Mock implementation of anyio.abc.Process for testing."""
    
    def __init__(self, exit_on_close=True, exit_on_terminate=True, exit_on_kill=True):
        self.pid = 12345
        self.returncode = None
        self.stdin = AsyncMock()
        self.stdout = AsyncMock()
        self._exit_on_close = exit_on_close
        self._exit_on_terminate = exit_on_terminate
        self._exit_on_kill = exit_on_kill
        self._terminated = False
        self._killed = False
    
    async def wait(self):
        # Simulate different exit behaviors
        if self._exit_on_close and hasattr(self.stdin, '_closed') and self.stdin._closed:
            self.returncode = 0
        elif self._exit_on_terminate and self._terminated:
            self.returncode = 0
        elif self._exit_on_kill and self._killed:
            self.returncode = 9
        else:
            # If we get here and returncode is still None, it means the process
            # is not exiting and should trigger a timeout
            if self.returncode is None:
                await anyio.sleep(10)  # This will trigger timeout in tests
        return self.returncode or 0
    
    def terminate(self):
        self._terminated = True
    
    def kill(self):
        self._killed = True
        self.returncode = 9


async def test_shutdown_normal_exit():
    """Test normal graceful shutdown where process exits after stdin close."""
    # Create a mock process that exits when stdin is closed
    mock_process = MockProcess(exit_on_close=True)
    
    # Mock the stdin close method
    async def mock_aclose():
        mock_process.stdin._closed = True
    
    mock_process.stdin.aclose = AsyncMock(side_effect=mock_aclose)
    
    # Create mock streams
    read_send, read_stream = anyio.create_memory_object_stream(max_buffer_size=10)
    write_stream, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    
    # Call shutdown function
    with patch("logging.info") as mock_log_info:
        await shutdown_stdio_server(
            read_stream=read_stream,
            write_stream=write_stream,
            process=mock_process,
            timeout=1.0
        )
    
    # Verify stdin was closed
    mock_process.stdin.aclose.assert_called_once()
    
    # Verify process exited normally
    assert "Process exited normally" in mock_log_info.call_args_list[-2][0][0]
    assert "Stdio server shutdown complete" in mock_log_info.call_args_list[-1][0][0]
    
    # Verify terminate and kill were not called
    assert not mock_process._terminated
    assert not mock_process._killed


async def test_shutdown_terminate_required():
    """Test shutdown where SIGTERM is required."""
    # Create a mock process that doesn't exit when stdin is closed, but exits on terminate
    mock_process = MockProcess(exit_on_close=False, exit_on_terminate=True)
    
    # Mock the stdin close method
    mock_process.stdin.aclose = AsyncMock()
    
    # Create mock streams
    read_send, read_stream = anyio.create_memory_object_stream(max_buffer_size=10)
    write_stream, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    
    # Call shutdown function
    with patch("logging.info") as mock_log_info, patch("logging.warning") as mock_log_warning:
        await shutdown_stdio_server(
            read_stream=read_stream,
            write_stream=write_stream,
            process=mock_process,
            timeout=0.1  # Short timeout to speed up test
        )
    
    # Verify stdin was closed
    mock_process.stdin.aclose.assert_called_once()
    
    # Verify terminate was called
    assert mock_process._terminated
    assert "sending SIGTERM" in mock_log_warning.call_args_list[0][0][0]
    
    # Verify process exited after SIGTERM
    assert "Process exited after SIGTERM" in mock_log_info.call_args_list[-2][0][0]
    assert "Stdio server shutdown complete" in mock_log_info.call_args_list[-1][0][0]
    
    # Verify kill was not called
    assert not mock_process._killed


async def test_shutdown_kill_required():
    """Test shutdown where SIGKILL is required."""
    # Create a mock process that doesn't exit when stdin is closed or terminated
    mock_process = MockProcess(exit_on_close=False, exit_on_terminate=False, exit_on_kill=True)
    
    # Mock the stdin close method
    mock_process.stdin.aclose = AsyncMock()
    
    # Create mock streams
    read_send, read_stream = anyio.create_memory_object_stream(max_buffer_size=10)
    write_stream, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    
    # Call shutdown function
    with patch("logging.info") as mock_log_info, patch("logging.warning") as mock_log_warning:
        await shutdown_stdio_server(
            read_stream=read_stream,
            write_stream=write_stream,
            process=mock_process,
            timeout=0.1  # Short timeout to speed up test
        )
    
    # Verify stdin was closed
    mock_process.stdin.aclose.assert_called_once()
    
    # Verify terminate and kill were both called
    assert mock_process._terminated
    assert mock_process._killed
    assert "sending SIGTERM" in mock_log_warning.call_args_list[0][0][0]
    assert "sending SIGKILL" in mock_log_warning.call_args_list[1][0][0]
    
    # Verify process exited after SIGKILL
    assert "Process exited after SIGKILL" in mock_log_info.call_args_list[-2][0][0]
    assert "Stdio server shutdown complete" in mock_log_info.call_args_list[-1][0][0]


async def test_shutdown_exception_handling():
    """Test handling of exceptions during shutdown."""
    # Create a mock process that raises an exception during stdin close
    mock_process = MockProcess()
    
    # Mock the stdin close method to raise an exception
    mock_process.stdin.aclose = AsyncMock(side_effect=Exception("Test exception"))
    
    # Create mock streams
    read_send, read_stream = anyio.create_memory_object_stream(max_buffer_size=10)
    write_stream, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    
    # Call shutdown function
    with patch("logging.info") as mock_log_info, patch("logging.error") as mock_log_error:
        await shutdown_stdio_server(
            read_stream=read_stream,
            write_stream=write_stream,
            process=mock_process,
            timeout=0.1
        )
    
    # Verify the exception was caught and logged
    assert "Unexpected error during stdio server shutdown" in mock_log_error.call_args[0][0]
    assert "Test exception" in mock_log_error.call_args[0][0]
    
    # Verify process was forcibly terminated
    assert mock_process._killed
    assert "Process forcibly terminated" in mock_log_info.call_args_list[-2][0][0]
    assert "Stdio server shutdown complete" in mock_log_info.call_args_list[-1][0][0]


async def test_shutdown_with_null_streams():
    """Test shutdown with null read/write streams."""
    # Create a mock process
    mock_process = MockProcess(exit_on_close=True)
    
    # Mock the stdin close method
    async def mock_aclose():
        mock_process.stdin._closed = True
    
    mock_process.stdin.aclose = AsyncMock(side_effect=mock_aclose)
    
    # Call shutdown function with null streams
    with patch("logging.info") as mock_log_info:
        await shutdown_stdio_server(
            read_stream=None,
            write_stream=None,
            process=mock_process,
            timeout=1.0
        )
    
    # Verify stdin was still closed
    mock_process.stdin.aclose.assert_called_once()
    
    # Verify process exited normally
    assert "Process exited normally" in mock_log_info.call_args_list[-2][0][0]
    assert "Stdio server shutdown complete" in mock_log_info.call_args_list[-1][0][0]


async def test_shutdown_with_null_process():
    """Test shutdown with null process."""
    # Create mock streams
    read_send, read_stream = anyio.create_memory_object_stream(max_buffer_size=10)
    write_stream, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    
    # Call shutdown function with null process
    with patch("logging.info") as mock_log_info:
        await shutdown_stdio_server(
            read_stream=read_stream,
            write_stream=write_stream,
            process=None,
            timeout=1.0
        )
    
    # Verify shutdown completed
    assert "Stdio server shutdown complete" in mock_log_info.call_args[0][0]