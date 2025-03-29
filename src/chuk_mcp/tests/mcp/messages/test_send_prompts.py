# tests/mcp/test_send_prompts.py
import pytest
import anyio

# chuk_mcp imports
from mcp_client.messages.json_rpc_message import JSONRPCMessage
from mcp_client.messages.message_method import MessageMethod
from mcp_client.messages.prompts.send_messages import send_prompts_list, send_prompts_get

# Force asyncio only for all tests in this file
pytestmark = [pytest.mark.asyncio]


async def test_send_prompts_list():
    """Test prompts/list request and response"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample prompts list response that matches MCP spec
    sample_prompts = {
        "prompts": [
            {
                "name": "code_review",
                "description": "Asks the LLM to analyze code quality and suggest improvements",
                "arguments": [
                    {
                        "name": "code",
                        "description": "The code to review",
                        "required": True
                    }
                ]
            },
            {
                "name": "weather_info",
                "description": "Get weather information for a location",
                "arguments": [
                    {
                        "name": "location",
                        "description": "The location to get weather for",
                        "required": True
                    }
                ]
            }
        ],
        "nextCursor": "next-page-cursor"
    }

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Verify it's a prompts/list method
            assert req.method == MessageMethod.PROMPTS_LIST
            
            # Check if cursor was included
            cursor = req.params.get("cursor") if req.params else None
            if cursor:
                assert cursor == "test-cursor"
            
            # Send response
            response = JSONRPCMessage(id=req.id, result=sample_prompts)
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request
        result = await send_prompts_list(
            read_stream=read_receive,
            write_stream=write_send,
            cursor="test-cursor"
        )
    
    # Check if response is correct
    assert result == sample_prompts
    assert len(result["prompts"]) == 2
    assert result["prompts"][0]["name"] == "code_review"
    assert result["prompts"][1]["name"] == "weather_info"
    assert result["nextCursor"] == "next-page-cursor"
    
    # Check for correct structure in prompt definitions
    assert "arguments" in result["prompts"][0]
    assert result["prompts"][0]["arguments"][0]["name"] == "code"
    assert result["prompts"][0]["arguments"][0]["required"] is True


async def test_send_prompts_get():
    """Test prompts/get request and response"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample prompt detail response per MCP specification
    sample_prompt_result = {
        "description": "Weather Information Prompt",
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": "Provide weather information for New York"
                }
            }
        ]
    }

    # Test prompt name and arguments
    test_prompt_name = "weather_info"
    test_prompt_args = {"location": "New York"}

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Verify it's a prompts/get method
            assert req.method == MessageMethod.PROMPTS_GET
            
            # Check the prompt name and arguments
            assert req.params.get("name") == test_prompt_name
            assert req.params.get("arguments") == test_prompt_args
            
            # Send response
            response = JSONRPCMessage(id=req.id, result=sample_prompt_result)
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request
        result = await send_prompts_get(
            read_stream=read_receive,
            write_stream=write_send,
            name=test_prompt_name,
            arguments=test_prompt_args
        )
    
    # Check if response is correct
    assert result == sample_prompt_result
    assert result["description"] == "Weather Information Prompt"
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"]["type"] == "text"
    assert "New York" in result["messages"][0]["content"]["text"]


async def test_send_prompts_get_not_found():
    """Test prompts/get with not found error"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Define server behavior that returns not found error
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Send protocol error response
            response = JSONRPCMessage(
                id=req.id, 
                error={
                    "code": -32602,
                    "message": "Invalid params: Prompt not found: invalid_prompt_name"
                }
            )
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request should raise an exception for not found error
        with pytest.raises(Exception) as exc_info:
            await send_prompts_get(
                read_stream=read_receive,
                write_stream=write_send,
                name="invalid_prompt_name",
                retries=1  # Only retry once to avoid timeout errors in test
            )
    
    # Verify error message
    assert "Prompt not found" in str(exc_info.value)


async def test_send_prompts_list_empty():
    """Test prompts/list with empty result"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Empty prompts list response
    empty_prompts = {
        "prompts": [],
        "nextCursor": None
    }

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Send response with empty list
            response = JSONRPCMessage(id=req.id, result=empty_prompts)
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request
        result = await send_prompts_list(
            read_stream=read_receive,
            write_stream=write_send
        )
    
    # Check if response is correct
    assert result == empty_prompts
    assert len(result["prompts"]) == 0
    assert result["nextCursor"] is None