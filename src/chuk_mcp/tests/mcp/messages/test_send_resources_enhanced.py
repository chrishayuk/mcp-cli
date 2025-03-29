# tests/mcp/test_send_resources_emhanced.py
import pytest
import anyio
import base64

from mcp_client.messages.json_rpc_message import JSONRPCMessage
from mcp_client.messages.message_method import MessageMethod
from mcp_client.messages.resources.send_messages import send_resources_read

# Force asyncio only for all tests in this file
pytestmark = [pytest.mark.asyncio]


async def test_send_resources_read_binary():
    """Test resources/read with binary content (blob)"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample base64 encoded binary content (very small 1x1 pixel transparent PNG)
    sample_blob = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    # Sample binary resource content response
    sample_content = {
        "contents": [
            {
                "uri": "file:///project/image.png",
                "mimeType": "image/png",
                "blob": sample_blob
            }
        ]
    }

    test_uri = "file:///project/image.png"

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Verify it's a resources/read method
            assert req.method == MessageMethod.RESOURCES_READ
            
            # Send response with binary content
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
    
    # Check if binary response is correct
    assert result == sample_content
    assert len(result["contents"]) == 1
    assert "blob" in result["contents"][0]
    assert result["contents"][0]["mimeType"] == "image/png"
    assert result["contents"][0]["blob"] == sample_blob

    # Verify blob is valid base64
    try:
        decoded = base64.b64decode(result["contents"][0]["blob"])
        assert len(decoded) > 0
    except Exception:
        pytest.fail("Could not decode base64 blob data")


async def test_send_resources_read_directory():
    """Test resources/read with a directory content type"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample directory content
    sample_content = {
        "contents": [
            {
                "uri": "file:///project/src/",
                "mimeType": "inode/directory",
                "text": "main.rs\nlib.rs\nutils/"
            }
        ]
    }

    test_uri = "file:///project/src/"

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Send response with directory content
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
    
    # Check if directory response is correct
    assert result == sample_content
    assert len(result["contents"]) == 1
    assert result["contents"][0]["mimeType"] == "inode/directory"
    assert "main.rs" in result["contents"][0]["text"]


async def test_send_resources_read_git():
    """Test resources/read with git URI scheme"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample git resource content
    sample_content = {
        "contents": [
            {
                "uri": "git:///project?ref=main&path=README.md",
                "mimeType": "text/markdown",
                "text": "# Project\n\nThis is a sample project."
            }
        ]
    }

    test_uri = "git:///project?ref=main&path=README.md"

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Send response with git content
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
    
    # Check if git resource response is correct
    assert result == sample_content
    assert result["contents"][0]["uri"].startswith("git://")
    assert result["contents"][0]["mimeType"] == "text/markdown"


async def test_send_resources_read_multiple_contents():
    """Test resources/read with multiple content items in response"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample multi-content response (e.g., for a template)
    sample_content = {
        "contents": [
            {
                "uri": "file:///project/src/main.rs",
                "mimeType": "text/x-rust",
                "text": "fn main() {\n    println!(\"Hello world!\");\n}"
            },
            {
                "uri": "file:///project/src/lib.rs",
                "mimeType": "text/x-rust",
                "text": "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}"
            }
        ]
    }

    # Test with template URI
    test_uri = "file:///{path}"

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Send response with multiple contents
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
    
    # Check if multi-content response is correct
    assert result == sample_content
    assert len(result["contents"]) == 2
    assert result["contents"][0]["uri"] == "file:///project/src/main.rs"
    assert result["contents"][1]["uri"] == "file:///project/src/lib.rs"


async def test_send_resources_read_https():
    """Test resources/read with https URI scheme"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample https resource content
    sample_content = {
        "contents": [
            {
                "uri": "https://example.com/api/data.json",
                "mimeType": "application/json",
                "text": "{\n  \"key\": \"value\"\n}"
            }
        ]
    }

    test_uri = "https://example.com/api/data.json"

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Send response with https content
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
    
    # Check if https resource response is correct
    assert result == sample_content
    assert result["contents"][0]["uri"].startswith("https://")
    assert result["contents"][0]["mimeType"] == "application/json"