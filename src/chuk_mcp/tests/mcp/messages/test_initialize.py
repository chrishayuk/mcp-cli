# tests/mcp/test_imitialize.py
import pytest
import anyio
import logging
from unittest.mock import patch

from chuk_mcp.mcp_client.messages.initialize.errors import VersionMismatchError
from chuk_mcp.mcp_client.messages.json_rpc_message import JSONRPCMessage
from chuk_mcp.mcp_client.messages.initialize.send_messages import (
    send_initialize,
    InitializeResult
)

# Force asyncio only for all tests in this file
pytestmark = [pytest.mark.asyncio]


async def test_send_initialize_success():
    """Test successful initialization with matching protocol version"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample server response
    server_response = {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "logging": {},
            "prompts": {
                "listChanged": True
            },
            "resources": {
                "subscribe": True,
                "listChanged": True
            },
            "tools": {
                "listChanged": True
            }
        },
        "serverInfo": {
            "name": "TestServer",
            "version": "1.0.0"
        }
    }

    # Define server behavior
    async def server_task():
        try:
            # Get the initialize request
            req = await write_receive.receive()
            
            # Verify it's an initialize method
            assert req.method == "initialize"
            
            # Check protocol version
            assert req.params.get("protocolVersion") == "2024-11-05"
            
            # Send success response
            response = JSONRPCMessage(id=req.id, result=server_response)
            await read_send.send(response)
            
            # Check for initialized notification
            notification = await write_receive.receive()
            assert notification.method == "notifications/initialized"
            
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request
        result = await send_initialize(
            read_stream=read_receive,
            write_stream=write_send
        )
    
    # Check if initialization was successful
    assert result is not None
    assert isinstance(result, InitializeResult)
    assert result.protocolVersion == "2024-11-05"
    assert result.serverInfo.name == "TestServer"
    assert "prompts" in result.capabilities.model_dump()
    assert result.capabilities.resources.get("subscribe") is True


async def test_send_initialize_version_mismatch():
    """Test initialization with protocol version mismatch"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample error response
    version_error = {
        "code": -32602,
        "message": "Unsupported protocol version",
        "data": {
            "supported": ["2024-05-20"],
            "requested": "2024-11-05"
        }
    }

    # Define server behavior
    async def server_task():
        try:
            # Get the initialize request
            req = await write_receive.receive()
            
            # Send error response
            response = JSONRPCMessage(id=req.id, error=version_error)
            await read_send.send(response)
            
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request should raise VersionMismatchError
        with pytest.raises(VersionMismatchError) as exc_info:
            await send_initialize(
                read_stream=read_receive,
                write_stream=write_send
            )
    
    # Verify error details
    assert exc_info.value.requested == "2024-11-05"
    assert exc_info.value.supported == ["2024-05-20"]


async def test_send_initialize_different_version():
    """Test initialization with different but supported protocol version"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample server response with different version
    server_response = {
        "protocolVersion": "2024-05-20",  # Different but supported version
        "capabilities": {
            "logging": {}
        },
        "serverInfo": {
            "name": "TestServer",
            "version": "1.0.0"
        }
    }

    # Define server behavior
    async def server_task():
        try:
            # Get the initialize request
            req = await write_receive.receive()
            
            # Send response with different version
            response = JSONRPCMessage(id=req.id, result=server_response)
            await read_send.send(response)
            
            # Will expect initialized notification if version is supported
            notification = await write_receive.receive()
            assert notification.method == "notifications/initialized"
            
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request with multiple supported versions
        result = await send_initialize(
            read_stream=read_receive,
            write_stream=write_send,
            supported_versions=["2024-11-05", "2024-05-20"]
        )
    
    # Check if initialization was successful with different version
    assert result is not None
    assert result.protocolVersion == "2024-05-20"


async def test_send_initialize_timeout():
    """Test initialization timeout"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # No response from server (timeout)
    async def server_task():
        try:
            # Get the initialize request but don't respond
            req = await write_receive.receive()
            # Ignore request and don't send response
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request should timeout
        with pytest.raises(TimeoutError):
            await send_initialize(
                read_stream=read_receive,
                write_stream=write_send,
                timeout=0.5  # Short timeout for test
            )


async def test_send_initialize_server_error():
    """Test initialization with server error"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample error response
    server_error = {
        "code": -32603,
        "message": "Internal server error during initialization"
    }

    # Define server behavior
    async def server_task():
        try:
            # Get the initialize request
            req = await write_receive.receive()
            
            # Send error response
            response = JSONRPCMessage(id=req.id, error=server_error)
            await read_send.send(response)
            
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request
        result = await send_initialize(
            read_stream=read_receive,
            write_stream=write_send
        )
    
    # Check if initialization failed
    assert result is None


# No test for shutdown as it's not part of the MCP protocol
# Shutdown is handled by the underlying transport mechanism