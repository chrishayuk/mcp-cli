# tests/mcp/test_send_message.py
import pytest
import anyio

# imports
from mcp.messages.json_rpc_message import JSONRPCMessage
from mcp.messages.send_message import send_message

# Force asyncio only for all tests in this file
pytestmark = [pytest.mark.asyncio]

async def test_send_message_success():
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Define server behavior
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            assert req.id == "unique123"
            
            # Send the response
            response = JSONRPCMessage(id=req.id, result={"foo": "bar"})
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side
        resp = await send_message(
            read_stream=read_receive,
            write_stream=write_send,
            method="test_method",
            params={"param1": "value1"},
            message_id="unique123",
            timeout=2,
            retries=1,
            retry_delay=0
        )
    
    # Check if response is correct
    assert resp == {"foo": "bar"}


async def test_send_message_timeout():
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Server that doesn't respond
    async def silent_server():
        try:
            # Receive but don't reply
            _ = await write_receive.receive()
            # Just wait longer than the timeout
            await anyio.sleep(3)
        except Exception:
            # Exception is expected when client times out and cancels
            pass

    # Create task group
    async with anyio.create_task_group() as tg:
        tg.start_soon(silent_server)
        
        # Client should timeout
        with pytest.raises(TimeoutError):
            await send_message(
                read_stream=read_receive,
                write_stream=write_send,
                method="timeout_method",
                params={"delay": True},
                timeout=1,
                retries=1,
                retry_delay=0
            )


async def test_send_message_error_response():
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    error_data = {"code": 500, "message": "Internal error"}

    # Server that sends an error response
    async def error_server():
        try:
            req = await write_receive.receive()
            assert req.id == "error123"
            
            # Send error response
            response = JSONRPCMessage(id=req.id, error=error_data)
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Error server failed: {e}")

    # Create task group
    async with anyio.create_task_group() as tg:
        tg.start_soon(error_server)
        
        # Client should receive error
        with pytest.raises(Exception) as exc_info:
            await send_message(
                read_stream=read_receive,
                write_stream=write_send,
                method="error_method",
                params={"cause_error": True},
                message_id="error123",
                timeout=2,
                retries=1,
                retry_delay=0
            )

    # Check error message
    assert error_data["message"] in str(exc_info.value)


async def test_message_id_generation():
    """Test that a unique ID is generated when message_id is not provided"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Variable to capture the generated ID
    captured_id = None

    async def check_id_server():
        nonlocal captured_id
        try:
            req = await write_receive.receive()
            # Capture the ID for later assertion
            captured_id = req.id
            
            # According to JSON-RPC 2.0, IDs must be strings or integers and must be unique
            assert isinstance(req.id, str)
            # We don't need to validate the exact format - just that it's a non-empty string
            assert len(req.id) > 0
            
            response = JSONRPCMessage(id=req.id, result={"success": True})
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"ID generation server failed: {e}")

    async with anyio.create_task_group() as tg:
        tg.start_soon(check_id_server)
        
        resp = await send_message(
            read_stream=read_receive,
            write_stream=write_send,
            method="test_method",
            params={"check": "id"},
            # No message_id provided
            timeout=2,
            retries=1,
            retry_delay=0
        )
    
    # Verify we got a proper response
    assert resp == {"success": True}
    # Verify ID was generated according to JSON-RPC 2.0 specs (string that's unique)
    assert captured_id is not None
    assert isinstance(captured_id, str)
    assert len(captured_id) > 0


async def test_send_message_retry():
    """Test that the function retries when timeout occurs"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Track number of requests received
    request_count = 0
    
    async def delayed_response_server():
        nonlocal request_count
        try:
            # Only respond to the second request
            while True:
                req = await write_receive.receive()
                request_count += 1
                
                if request_count == 2:
                    # Respond on second attempt
                    response = JSONRPCMessage(id=req.id, result={"retry": "success"})
                    await read_send.send(response)
                    break
                # First attempt will timeout
        except Exception:
            # Handle any exceptions from client disconnecting
            pass

    async with anyio.create_task_group() as tg:
        tg.start_soon(delayed_response_server)
        
        # Set timeout low but allow retries
        resp = await send_message(
            read_stream=read_receive,
            write_stream=write_send,
            method="retry_method",
            params={"test": "retry"},
            message_id="retry123",
            timeout=0.5,  # Short timeout to trigger retry
            retries=3,    # Multiple retries allowed
            retry_delay=0.1  # Short delay between retries for testing
        )
    
    # Verify we got response on second attempt
    assert resp == {"retry": "success"}
    assert request_count == 2


async def test_invalid_json_rpc_response():
    """Test handling of invalid JSON-RPC responses containing both result and error"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    error_data = {"code": 500, "message": "Error with result"}

    # Server that sends an invalid response with both error and result
    async def invalid_response_server():
        try:
            req = await write_receive.receive()
            # Create an invalid response with both result and error
            response = JSONRPCMessage(
                id=req.id, 
                error=error_data,
                result={"some": "data"}
            )
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Invalid response server failed: {e}")

    async with anyio.create_task_group() as tg:
        tg.start_soon(invalid_response_server)
        
        # Client should handle the error
        with pytest.raises(Exception) as exc_info:
            await send_message(
                read_stream=read_receive,
                write_stream=write_send,
                method="invalid_method",
                message_id="invalid123",
                timeout=2,
                retries=1,
                retry_delay=0
            )

    # Verify error is raised properly
    assert error_data["message"] in str(exc_info.value)