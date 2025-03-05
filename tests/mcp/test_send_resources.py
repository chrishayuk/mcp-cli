import pytest
import anyio

# imports
from mcp.messages.message_types.json_rpc_message import JSONRPCMessage
from mcp.messages.message_types.message_method import MessageMethod
from mcp.messages.resources.send_messages import (
    send_resources_list,
    send_resources_read,
    send_resources_templates_list,
    send_resources_subscribe
)

# Force asyncio only for all tests in this file
pytestmark = [pytest.mark.asyncio]


async def test_send_resources_list():
    """Test resources/list request and response"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample resources list response
    sample_resources = {
        "resources": [
            {
                "uri": "file:///project/src/main.rs",
                "name": "main.rs",
                "description": "Primary application entry point",
                "mimeType": "text/x-rust"
            },
            {
                "uri": "file:///project/src/lib.rs",
                "name": "lib.rs",
                "mimeType": "text/x-rust"
            }
        ],
        "nextCursor": "next-page-token"
    }

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Verify it's a resources/list method
            assert req.method == MessageMethod.RESOURCES_LIST
            
            # Check if cursor was included
            cursor = req.params.get("cursor") if req.params else None
            if cursor:
                assert cursor == "test-cursor"
            
            # Send response
            response = JSONRPCMessage(id=req.id, result=sample_resources)
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request
        result = await send_resources_list(
            read_stream=read_receive,
            write_stream=write_send,
            cursor="test-cursor"
        )
    
    # Check if response is correct
    assert result == sample_resources
    assert len(result["resources"]) == 2
    assert result["resources"][0]["uri"] == "file:///project/src/main.rs"
    assert result["nextCursor"] == "next-page-token"


async def test_send_resources_read():
    """Test resources/read request and response"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample resource content
    sample_content = {
        "contents": [
            {
                "uri": "file:///project/src/main.rs",
                "mimeType": "text/x-rust",
                "text": "fn main() {\n    println!(\"Hello world!\");\n}"
            }
        ]
    }

    test_uri = "file:///project/src/main.rs"

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Verify it's a resources/read method
            assert req.method == MessageMethod.RESOURCES_READ
            
            # Check the URI parameter
            assert req.params.get("uri") == test_uri
            
            # Send response
            response = JSONRPCMessage(id=req.id, result=sample_content)
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request
        result = await send_resources_read(
            read_stream=read_receive,
            write_stream=write_send,
            uri=test_uri
        )
    
    # Check if response is correct
    assert result == sample_content
    assert len(result["contents"]) == 1
    assert "text" in result["contents"][0]
    assert result["contents"][0]["uri"] == test_uri


async def test_send_resources_templates_list():
    """Test resources/templates/list request and response"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample templates response
    sample_templates = {
        "resourceTemplates": [
            {
                "uriTemplate": "file:///{path}",
                "name": "Project Files",
                "description": "Access files in the project directory",
                "mimeType": "application/octet-stream"
            }
        ]
    }

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Verify it's a templates/list method
            assert req.method == MessageMethod.RESOURCES_TEMPLATES_LIST
            
            # Send response
            response = JSONRPCMessage(id=req.id, result=sample_templates)
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request
        result = await send_resources_templates_list(
            read_stream=read_receive,
            write_stream=write_send
        )
    
    # Check if response is correct
    assert result == sample_templates
    assert len(result["resourceTemplates"]) == 1
    assert result["resourceTemplates"][0]["uriTemplate"] == "file:///{path}"


async def test_send_resources_subscribe_success():
    """Test successful resource subscription"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    test_uri = "file:///project/src/main.rs"

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Verify it's a subscribe method
            assert req.method == MessageMethod.RESOURCES_SUBSCRIBE
            
            # Check the URI parameter
            assert req.params.get("uri") == test_uri
            
            # Send successful response
            response = JSONRPCMessage(id=req.id, result={"subscribed": True})
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request
        result = await send_resources_subscribe(
            read_stream=read_receive,
            write_stream=write_send,
            uri=test_uri
        )
    
    # Check if subscription was successful
    assert result is True


async def test_send_resources_subscribe_error():
    """Test failed resource subscription"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    test_uri = "file:///nonexistent.txt"

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Verify it's a subscribe method
            assert req.method == MessageMethod.RESOURCES_SUBSCRIBE
            
            # Send error response
            response = JSONRPCMessage(
                id=req.id, 
                error={
                    "code": -32002,
                    "message": "Resource not found",
                    "data": {"uri": test_uri}
                }
            )
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request
        result = await send_resources_subscribe(
            read_stream=read_receive,
            write_stream=write_send,
            uri=test_uri
        )
    
    # Check if subscription failure is reported
    assert result is False


async def test_resources_list_error_handling():
    """Test handling of error responses for resources/list"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Send error response
            response = JSONRPCMessage(
                id=req.id, 
                error={
                    "code": -32603,
                    "message": "Internal server error"
                }
            )
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side request should raise an exception
        with pytest.raises(Exception):
            await send_resources_list(
                read_stream=read_receive,
                write_stream=write_send
            )