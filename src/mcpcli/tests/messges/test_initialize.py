import pytest
from unittest.mock import AsyncMock
from mcpcli.messages.send_initialize_message import send_initialize
from mcpcli.messages.message_types.initialize_message import InitializeResult, ServerInfo
from mcpcli.messages.message_types.json_rpc_message import JSONRPCMessage


@pytest.mark.asyncio
async def test_send_initialize_success():
    # Mock a JSONRPCMessage response that includes a valid InitializeResult
    mock_response = JSONRPCMessage(
        id="init-1",
        jsonrpc="2.0",
        result={
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "logging": {},
                "prompts": None,
                "resources": None,
                "tools": None,
            },
            "serverInfo": {"name": "TestServer", "version": "1.0.0"},
        }
    )

    mock_write_stream = AsyncMock()
    mock_read_stream = AsyncMock()
    # The async for loop in send_initialize will iterate over these responses
    mock_read_stream.__aiter__.return_value = [mock_response]

    result = await send_initialize(read_stream=mock_read_stream, write_stream=mock_write_stream)
    assert result is not None
    assert isinstance(result, InitializeResult)
    assert result.protocolVersion == "2024-11-05"
    assert result.serverInfo == ServerInfo(name="TestServer", version="1.0.0")
    mock_write_stream.send.assert_awaited()


@pytest.mark.asyncio
async def test_send_initialize_error():
    # Mock a JSONRPCMessage with an error field
    mock_response = JSONRPCMessage(
        id="init-1",
        jsonrpc="2.0",
        error={"code": -32603, "message": "Internal server error"}
    )

    mock_write_stream = AsyncMock()
    mock_read_stream = AsyncMock()
    mock_read_stream.__aiter__.return_value = [mock_response]

    result = await send_initialize(read_stream=mock_read_stream, write_stream=mock_write_stream)
    assert result is None
    mock_write_stream.send.assert_awaited()


@pytest.mark.asyncio
async def test_send_initialize_no_response():
    # No messages returned from the read_stream
    mock_write_stream = AsyncMock()
    mock_read_stream = AsyncMock()
    mock_read_stream.__aiter__.return_value = []

    result = await send_initialize(read_stream=mock_read_stream, write_stream=mock_write_stream)
    assert result is None
    mock_write_stream.send.assert_awaited()


@pytest.mark.asyncio
async def test_send_initialize_exception():
    # The server sends an Exception object instead of a JSONRPCMessage
    mock_exception = Exception("Simulated server error")

    mock_write_stream = AsyncMock()
    mock_read_stream = AsyncMock()
    # Only one item (an Exception) in the response stream
    mock_read_stream.__aiter__.return_value = [mock_exception]

    result = await send_initialize(read_stream=mock_read_stream, write_stream=mock_write_stream)
    # Since the response was an exception and no valid response followed, result should be None
    assert result is None
    mock_write_stream.send.assert_awaited()
