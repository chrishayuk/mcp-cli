import pytest
from unittest.mock import patch, AsyncMock
from mcpcli.messages.send_ping import send_ping
from mcpcli.messages.message_types.ping_message import PingMessage

@pytest.mark.asyncio
async def test_send_ping_success():
    # Mock send_message to return a non-None response, simulating a successful ping
    mock_send_message = AsyncMock(return_value={"id": "ping-1", "result": {"status": "ok"}})

    with patch("mcpcli.messages.send_ping.send_message", new=mock_send_message):
        result = await send_ping(read_stream=None, write_stream=None)
        assert result is True
        mock_send_message.assert_awaited_once()

        # Extract the arguments that send_message was called with
        _, kwargs = mock_send_message.await_args
        # Check that a message was passed in and it is a PingMessage
        assert isinstance(kwargs["message"], PingMessage)
        assert kwargs["message"].method == "ping"
        # Assuming the ping message ID increments as expected
        assert kwargs["message"].id.startswith("ping-")

@pytest.mark.asyncio
async def test_send_ping_timeout():
    # Mock send_message to raise TimeoutError
    mock_send_message = AsyncMock(side_effect=TimeoutError("No response received"))

    with patch("mcpcli.messages.send_ping.send_message", new=mock_send_message):
        with pytest.raises(TimeoutError):
            await send_ping(read_stream=None, write_stream=None)
        mock_send_message.assert_awaited_once()

        # Verify message was passed correctly
        _, kwargs = mock_send_message.await_args
        assert isinstance(kwargs["message"], PingMessage)
        assert kwargs["message"].method == "ping"

@pytest.mark.asyncio
async def test_send_ping_error():
    # Mock send_message to raise a generic Exception
    mock_send_message = AsyncMock(side_effect=Exception("Server error"))

    with patch("mcpcli.messages.send_ping.send_message", new=mock_send_message):
        with pytest.raises(Exception, match="Server error"):
            await send_ping(read_stream=None, write_stream=None)
        mock_send_message.assert_awaited_once()

        # Verify message was passed correctly
        _, kwargs = mock_send_message.await_args
        assert isinstance(kwargs["message"], PingMessage)
        assert kwargs["message"].method == "ping"

@pytest.mark.asyncio
async def test_send_ping_no_response():
    # Mock send_message to return None
    mock_send_message = AsyncMock(return_value=None)

    with patch("mcpcli.messages.send_ping.send_message", new=mock_send_message):
        result = await send_ping(read_stream=None, write_stream=None)
        assert result is False
        mock_send_message.assert_awaited_once()

        # Verify message was passed correctly
        _, kwargs = mock_send_message.await_args
        assert isinstance(kwargs["message"], PingMessage)
        assert kwargs["message"].method == "ping"
