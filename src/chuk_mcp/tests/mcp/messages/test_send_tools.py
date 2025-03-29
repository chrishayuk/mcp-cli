# tests/mcp/test_send_tools.py
import pytest
import anyio
from mcp_client.messages.json_rpc_message import JSONRPCMessage
from mcp_client.messages.message_method import MessageMethod
from mcp_client.messages.tools.send_messages import send_tools_list, send_tools_call

# Force asyncio only for all tests in this file
pytestmark = [pytest.mark.asyncio]


async def test_send_tools_list():
    """Test tools/list request and response"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample tools list response
    sample_tools = {
        "tools": [
            {
                "name": "get_weather",
                "description": "Get current weather information for a location",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name or zip code"
                        }
                    },
                    "required": ["location"]
                }
            }
        ],
        "nextCursor": "next-page-cursor"
    }

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Verify it's a tools/list method
            assert req.method == MessageMethod.TOOLS_LIST
            
            # Check if cursor was included
            cursor = req.params.get("cursor") if req.params else None
            if cursor:
                assert cursor == "test-cursor"
            
            # Send response
            response = JSONRPCMessage(id=req.id, result=sample_tools)
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request
        result = await send_tools_list(
            read_stream=read_receive,
            write_stream=write_send,
            cursor="test-cursor"
        )
    
    # Check if response is correct
    assert result == sample_tools
    assert len(result["tools"]) == 1
    assert result["tools"][0]["name"] == "get_weather"
    assert result["nextCursor"] == "next-page-cursor"


async def test_send_tools_call():
    """Test tools/call request and response"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample tool call result
    sample_result = {
        "content": [
            {
                "type": "text",
                "text": "Current weather in New York:\nTemperature: 72°F\nConditions: Partly cloudy"
            }
        ],
        "isError": False
    }

    # Test tool and arguments
    test_tool = "get_weather"
    test_args = {"location": "New York"}

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Verify it's a tools/call method
            assert req.method == MessageMethod.TOOLS_CALL
            
            # Check the tool name and arguments
            assert req.params.get("name") == test_tool
            assert req.params.get("arguments") == test_args
            
            # Send response
            response = JSONRPCMessage(id=req.id, result=sample_result)
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request
        result = await send_tools_call(
            read_stream=read_receive,
            write_stream=write_send,
            name=test_tool,
            arguments=test_args
        )
    
    # Check if response is correct
    assert result == sample_result
    assert len(result["content"]) == 1
    assert result["content"][0]["type"] == "text"
    assert "New York" in result["content"][0]["text"]
    assert result["isError"] is False


async def test_send_tools_call_error():
    """Test tools/call with tool execution error"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample error result
    sample_error_result = {
        "content": [
            {
                "type": "text",
                "text": "Failed to fetch weather data: API rate limit exceeded"
            }
        ],
        "isError": True
    }

    # Test tool and arguments
    test_tool = "get_weather"
    test_args = {"location": "Invalid Location"}

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Send error response (but still in "result" field per spec)
            response = JSONRPCMessage(id=req.id, result=sample_error_result)
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request - should not raise exception since this is a "tool" error,
        # not a protocol error
        result = await send_tools_call(
            read_stream=read_receive,
            write_stream=write_send,
            name=test_tool,
            arguments=test_args
        )
    
    # Check if response indicates error
    assert result == sample_error_result
    assert result["isError"] is True
    assert "Failed" in result["content"][0]["text"]


async def test_send_tools_call_protocol_error():
    """Test tools/call with protocol error (unknown tool)"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Define server behavior that returns protocol error
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Send protocol error response
            response = JSONRPCMessage(
                id=req.id, 
                error={
                    "code": -32602,
                    "message": "Unknown tool: invalid_tool_name"
                }
            )
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request should raise an exception for protocol errors
        with pytest.raises(Exception) as exc_info:
            await send_tools_call(
                read_stream=read_receive,
                write_stream=write_send,
                name="invalid_tool_name",
                arguments={},
                retries=1  # Only retry once to avoid timeout errors in test
            )
    
    # Verify error message
    assert "Unknown tool" in str(exc_info.value)