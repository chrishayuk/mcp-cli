# tests/mcp/test_send_tools_enhanced.py
import pytest
import anyio
import base64

from mcp.messages.json_rpc_message import JSONRPCMessage
from mcp.messages.message_method import MessageMethod
from mcp.messages.tools.send_messages import send_tools_call

# Force asyncio only for all tests in this file
pytestmark = [pytest.mark.asyncio]


async def test_send_tools_call_with_image_result():
    """Test tools/call with an image result"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample base64 encoded image (very small 1x1 pixel transparent PNG)
    sample_base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

    # Sample tool call result with image content
    sample_result = {
        "content": [
            {
                "type": "text",
                "text": "Here's a visualization of the data:"
            },
            {
                "type": "image",
                "data": sample_base64_image,
                "mimeType": "image/png"
            }
        ],
        "isError": False
    }

    # Test tool and arguments
    test_tool = "generate_chart"
    test_args = {"chartType": "bar", "data": [10, 20, 30, 40]}

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Send response with image content
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
    
    # Check if image response is correct
    assert result == sample_result
    assert len(result["content"]) == 2
    assert result["content"][0]["type"] == "text"
    assert result["content"][1]["type"] == "image"
    assert result["content"][1]["data"] == sample_base64_image
    assert result["content"][1]["mimeType"] == "image/png"

    # Verify base64 image data is valid
    try:
        decoded = base64.b64decode(result["content"][1]["data"])
        assert len(decoded) > 0
    except Exception:
        pytest.fail("Could not decode base64 image data")


async def test_send_tools_call_with_resource_result():
    """Test tools/call with an embedded resource result"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample tool call result with embedded resource
    sample_result = {
        "content": [
            {
                "type": "text",
                "text": "Here are the query results:"
            },
            {
                "type": "resource",
                "resource": {
                    "uri": "resource://query-results/12345",
                    "mimeType": "application/json",
                    "text": '{\n  "results": [\n    {"id": 1, "name": "Item 1"},\n    {"id": 2, "name": "Item 2"}\n  ]\n}'
                }
            }
        ],
        "isError": False
    }

    # Test tool and arguments
    test_tool = "query_database"
    test_args = {"query": "SELECT * FROM items LIMIT 2"}

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Send response with resource content
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
    
    # Check if resource response is correct
    assert result == sample_result
    assert len(result["content"]) == 2
    assert result["content"][1]["type"] == "resource"
    assert result["content"][1]["resource"]["uri"] == "resource://query-results/12345"
    assert result["content"][1]["resource"]["mimeType"] == "application/json"
    assert "results" in result["content"][1]["resource"]["text"]


async def test_send_tools_call_with_mixed_content():
    """Test tools/call with mixed content types in result"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Sample base64 encoded binary data
    sample_blob = "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"  # 1x1 GIF

    # Sample tool call with mixed content types
    sample_result = {
        "content": [
            {
                "type": "text",
                "text": "Analysis complete. Here are the results:"
            },
            {
                "type": "image",
                "data": sample_blob,
                "mimeType": "image/gif"
            },
            {
                "type": "resource",
                "resource": {
                    "uri": "resource://analysis/full-report",
                    "mimeType": "text/html",
                    "text": "<html><body><h1>Full Report</h1><p>Details here...</p></body></html>"
                }
            }
        ],
        "isError": False
    }

    # Test tool and arguments
    test_tool = "analyze_data"
    test_args = {"dataset": "sales_2023", "type": "comprehensive"}

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Send response with mixed content
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
    
    # Check if mixed content response is correct
    assert result == sample_result
    assert len(result["content"]) == 3
    
    # Check text content
    assert result["content"][0]["type"] == "text"
    assert "Analysis complete" in result["content"][0]["text"]
    
    # Check image content
    assert result["content"][1]["type"] == "image"
    assert result["content"][1]["data"] == sample_blob
    
    # Check resource content
    assert result["content"][2]["type"] == "resource"
    assert "<h1>Full Report</h1>" in result["content"][2]["resource"]["text"]


async def test_send_tools_call_with_complex_arguments():
    """Test tools/call with complex nested arguments"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Complex nested arguments
    complex_args = {
        "query": {
            "filters": [
                {"field": "category", "operator": "eq", "value": "electronics"},
                {"field": "price", "operator": "lt", "value": 1000}
            ],
            "sort": {"field": "rating", "order": "desc"},
            "pagination": {"page": 1, "limit": 10}
        },
        "format": "detailed"
    }

    # Sample success result
    sample_result = {
        "content": [
            {
                "type": "text",
                "text": "Found 5 items matching your criteria."
            }
        ],
        "isError": False
    }

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Verify the complex arguments are passed correctly
            assert req.params.get("arguments") == complex_args
            
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
            name="search_products",
            arguments=complex_args
        )
    
    # Check if response is correct
    assert result == sample_result
    assert not result["isError"]