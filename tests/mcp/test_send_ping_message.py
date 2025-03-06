# tests/mcp/test_send_ping_message.py
import pytest
import anyio

from mcp.messages.json_rpc_message import JSONRPCMessage
from mcp.messages.message_method import MessageMethod
from mcp.messages.ping.send_ping_message import send_ping

# Force asyncio only for all tests in this file
pytestmark = [pytest.mark.asyncio]


async def test_send_ping_success():
    """Test successful ping response"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Define server behavior that responds to ping
    async def server_task():
        try:
            # Get the request
            req = await write_receive.receive()
            
            # Verify it's a ping method
            assert req.method == MessageMethod.PING
            
            # Send success response
            response = JSONRPCMessage(id=req.id, result={"status": "ok"})
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Server task failed: {e}")

    # Create task group and run both client and server
    async with anyio.create_task_group() as tg:
        tg.start_soon(server_task)
        
        # Client side ping
        result = await send_ping(
            read_stream=read_receive,
            write_stream=write_send
        )
    
    # Check if ping was successful
    assert result is True


async def test_send_ping_error_response():
    """Test ping with error response"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Server that responds with error
    async def error_server():
        try:
            req = await write_receive.receive()
            
            # Send error response
            response = JSONRPCMessage(
                id=req.id, 
                error={"code": 500, "message": "Internal server error"}
            )
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Error server failed: {e}")

    async with anyio.create_task_group() as tg:
        tg.start_soon(error_server)
        
        # Ping should return False with error response
        result = await send_ping(
            read_stream=read_receive,
            write_stream=write_send
        )
    
    # Verify failure is reported
    assert result is False


async def test_send_ping_timeout():
    """Test ping timeout"""
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

    async with anyio.create_task_group() as tg:
        tg.start_soon(silent_server)
        
        # Ping with short timeout
        result = await send_ping(
            read_stream=read_receive,
            write_stream=write_send,
            timeout=0.5,
            retries=1
        )
    
    # Verify timeout is reported as failure
    assert result is False


async def test_send_ping_retry_success():
    """Test ping succeeds on retry"""
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
                    response = JSONRPCMessage(id=req.id, result={"status": "ok"})
                    await read_send.send(response)
                    break
                # First attempt will timeout
        except Exception:
            # Handle any exceptions from client disconnecting
            pass

    async with anyio.create_task_group() as tg:
        tg.start_soon(delayed_response_server)
        
        # Set timeout low but allow retries
        result = await send_ping(
            read_stream=read_receive,
            write_stream=write_send,
            timeout=0.5,  # Short timeout to trigger retry
            retries=3     # Multiple retries allowed
        )
    
    # Verify eventual success
    assert result is True
    assert request_count == 2


async def test_send_ping_method_enum():
    """Test that ping uses the MessageMethod enum correctly"""
    read_send, read_receive = anyio.create_memory_object_stream(max_buffer_size=10)
    write_send, write_receive = anyio.create_memory_object_stream(max_buffer_size=10)

    # Variable to capture the method
    captured_method = None

    async def check_method_server():
        nonlocal captured_method
        try:
            req = await write_receive.receive()
            # Capture the method for later assertion
            captured_method = req.method
            
            # Respond to complete the ping
            response = JSONRPCMessage(id=req.id, result={"status": "ok"})
            await read_send.send(response)
        except Exception as e:
            pytest.fail(f"Method check server failed: {e}")

    async with anyio.create_task_group() as tg:
        tg.start_soon(check_method_server)
        
        # Perform ping
        await send_ping(
            read_stream=read_receive,
            write_stream=write_send
        )
    
    # Verify correct method was used
    assert captured_method == MessageMethod.PING
    assert captured_method == "ping"  # String comparison also works