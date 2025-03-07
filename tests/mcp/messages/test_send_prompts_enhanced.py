# tests/mcp/test_send_prompts_enhanced.py
import pytest
import anyio
import base64
from mcp.messages.json_rpc_message import JSONRPCMessage
from mcp.messages.message_method import MessageMethod
from mcp.messages.prompts.send_messages import send_prompts_list, send_prompts_get

# Force asyncio only for all tests in this file
pytestmark = [pytest.mark.asyncio]


async def test_send_prompts_get_multi_message_types():
    """Test prompts/get request and response with multiple message types (text, image, resource)"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample base64 encoded image (very small 1x1 pixel transparent PNG)
    sample_base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    # Sample prompt with multiple content types
    complex_prompt_result = {
        "description": "Mixed content prompt example",
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": "Analyze this code sample and explain the image below:"
                }
            },
            {
                "role": "user",
                "content": {
                    "type": "resource",
                    "resource": {
                        "uri": "resource://code-samples/example.py",
                        "mimeType": "text/plain",
                        "text": "def hello_world():\n    print('Hello, world!')"
                    }
                }
            },
            {
                "role": "user",
                "content": {
                    "type": "image",
                    "data": sample_base64_image,
                    "mimeType": "image/png"
                }
            },
            {
                "role": "assistant",
                "content": {
                    "type": "text",
                    "text": "The code defines a simple function that prints 'Hello, world!'."
                }
            }
        ]
    }

    # Test prompt name and arguments
    test_prompt_name = "complex_prompt"
    test_prompt_args = {"format": "detailed", "language": "python"}

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
            response = JSONRPCMessage(id=req.id, result=complex_prompt_result)
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
    assert result == complex_prompt_result
    
    # Verify multiple message types
    assert len(result["messages"]) == 4
    
    # Check text content message
    assert result["messages"][0]["content"]["type"] == "text"
    
    # Check resource content
    assert result["messages"][1]["content"]["type"] == "resource"
    assert result["messages"][1]["content"]["resource"]["uri"] == "resource://code-samples/example.py"
    assert result["messages"][1]["content"]["resource"]["mimeType"] == "text/plain"
    
    # Check image content
    assert result["messages"][2]["content"]["type"] == "image"
    assert result["messages"][2]["content"]["data"] == sample_base64_image
    assert result["messages"][2]["content"]["mimeType"] == "image/png"
    
    # Check assistant message
    assert result["messages"][3]["role"] == "assistant"


async def test_send_prompts_get_missing_required_argument():
    """Test prompts/get with missing required argument error"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Define server behavior that returns error for missing required argument
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Send protocol error response
            response = JSONRPCMessage(
                id=req.id, 
                error={
                    "code": -32602,
                    "message": "Invalid params: Missing required argument 'code'"
                }
            )
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request should raise an exception for missing argument
        with pytest.raises(Exception) as exc_info:
            await send_prompts_get(
                read_stream=read_receive,
                write_stream=write_send,
                name="code_review",
                arguments={"language": "python"},  # Missing 'code' argument
                retries=1  # Only retry once to avoid timeout errors in test
            )
    
    # Verify error message
    assert "Missing required argument" in str(exc_info.value)


async def test_send_prompts_get_resource_blob():
    """Test prompts/get with a binary resource response"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample blob data (binary content as base64)
    sample_blob_data = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"  # 1x1 GIF

    # Sample prompt with binary resource
    resource_prompt_result = {
        "description": "Binary resource prompt example",
        "messages": [
            {
                "role": "user",
                "content": {
                    "type": "text",
                    "text": "Here's an example binary file:"
                }
            },
            {
                "role": "user",
                "content": {
                    "type": "resource",
                    "resource": {
                        "uri": "resource://examples/sample.bin",
                        "mimeType": "application/octet-stream",
                        "blob": sample_blob_data
                    }
                }
            }
        ]
    }

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Send response
            response = JSONRPCMessage(id=req.id, result=resource_prompt_result)
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
            name="binary_example"
        )
    
    # Check if response has the correct binary resource
    assert result["messages"][1]["content"]["type"] == "resource"
    assert result["messages"][1]["content"]["resource"]["mimeType"] == "application/octet-stream"
    assert result["messages"][1]["content"]["resource"]["blob"] == sample_blob_data


async def test_send_prompts_get_server_resource_error():
    """Test prompts/get with error accessing a resource"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample prompt with an error response
    resource_error_result = {
        "isError": True,
        "content": [
            {
                "type": "text", 
                "text": "Error accessing resource: Resource not found or access denied"
            }
        ]
    }

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Send response with resource error
            response = JSONRPCMessage(id=req.id, result=resource_error_result)
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
            name="restricted_resource",
            arguments={"resourceId": "secure-123"}
        )
    
    # Should return the error without raising an exception (tool error vs protocol error)
    assert result == resource_error_result
    assert result["isError"] is True
    assert "Error accessing resource" in result["content"][0]["text"]


async def test_send_prompts_list_extensive():
    """Test prompts/list with extensive prompt definitions"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample prompts list with extensive argument definitions
    extensive_prompts = {
        "prompts": [
            {
                "name": "code_review",
                "description": "Asks the LLM to analyze code quality and suggest improvements",
                "arguments": [
                    {
                        "name": "code",
                        "description": "The code to review",
                        "required": True
                    },
                    {
                        "name": "language",
                        "description": "Programming language of the code",
                        "required": False
                    },
                    {
                        "name": "focus",
                        "description": "Review focus (e.g., 'performance', 'security', 'readability')",
                        "required": False,
                        "enum": ["performance", "security", "readability", "all"]
                    }
                ]
            },
            {
                "name": "image_analysis",
                "description": "Analyzes images and provides descriptions",
                "arguments": [
                    {
                        "name": "imageData",
                        "description": "Base64-encoded image data",
                        "required": True
                    },
                    {
                        "name": "format",
                        "description": "Format of the analysis (brief or detailed)",
                        "required": False,
                        "default": "brief"
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
            
            # Send response
            response = JSONRPCMessage(id=req.id, result=extensive_prompts)
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
    
    # Verify response has extensive argument definitions
    assert result == extensive_prompts
    
    # Check code_review prompt
    code_review = result["prompts"][0]
    assert len(code_review["arguments"]) == 3
    assert code_review["arguments"][0]["required"] is True
    assert code_review["arguments"][1]["required"] is False
    assert "enum" in code_review["arguments"][2]
    assert len(code_review["arguments"][2]["enum"]) == 4
    
    # Check image_analysis prompt
    image_analysis = result["prompts"][1]
    assert len(image_analysis["arguments"]) == 2
    assert image_analysis["arguments"][0]["required"] is True
    assert "default" in image_analysis["arguments"][1]
    assert image_analysis["arguments"][1]["default"] == "brief"