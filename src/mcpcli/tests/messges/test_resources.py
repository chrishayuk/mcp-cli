# tests/test_resources.py
import pytest
from unittest.mock import patch, AsyncMock
from mcpcli.messages.send_resources import send_resources_list
from mcpcli.messages.message_types.resources_messages import ResourcesListMessage

@pytest.mark.asyncio
async def test_send_resources_list_success():
    mock_response = {"id": "resources-list-1", "result": ["res1", "res2"]}
    mock_send_message = AsyncMock(return_value=mock_response)

    with patch("mcpcli.messages.send_resources.send_message", new=mock_send_message):
        result = await send_resources_list(read_stream=None, write_stream=None)
        assert result == ["res1", "res2"]
        mock_send_message.assert_awaited_once()

        args, kwargs = mock_send_message.call_args
        sent_msg = kwargs["message"]
        assert isinstance(sent_msg, ResourcesListMessage)
        assert sent_msg.id == "resources-list-1"

@pytest.mark.asyncio
async def test_send_resources_list_no_result():
    mock_response = {"id": "resources-list-1"}
    mock_send_message = AsyncMock(return_value=mock_response)

    with patch("mcpcli.messages.send_resources.send_message", new=mock_send_message):
        result = await send_resources_list(read_stream=None, write_stream=None)
        assert result == []
        mock_send_message.assert_awaited_once()

@pytest.mark.asyncio
async def test_send_resources_list_error():
    mock_send_message = AsyncMock(side_effect=Exception("Server error"))

    with patch("mcpcli.messages.send_resources.send_message", new=mock_send_message):
        with pytest.raises(Exception, match="Server error"):
            await send_resources_list(read_stream=None, write_stream=None)
        mock_send_message.assert_awaited_once()

@pytest.mark.asyncio
async def test_send_resources_list_increment_id():
    # Reset the counter for predictable test results
    ResourcesListMessage.load_counter(0)

    mock_response_1 = {"id": "resources-list-1", "result": ["resA"]}
    mock_send_message = AsyncMock(return_value=mock_response_1)

    with patch("mcpcli.messages.send_resources.send_message", new=mock_send_message):
        result = await send_resources_list(read_stream=None, write_stream=None)
        assert result == ["resA"]
        mock_send_message.assert_awaited_once()

        args, kwargs = mock_send_message.call_args
        sent_msg_1 = kwargs["message"]
        assert sent_msg_1.id == "resources-list-1"

    # Test increment
    mock_send_message.reset_mock()
    mock_response_2 = {"id": "resources-list-2", "result": ["resB"]}
    mock_send_message.return_value = mock_response_2

    with patch("mcpcli.messages.send_resources.send_message", new=mock_send_message):
        result = await send_resources_list(read_stream=None, write_stream=None)
        assert result == ["resB"]
        mock_send_message.assert_awaited_once()

        args, kwargs = mock_send_message.call_args
        sent_msg_2 = kwargs["message"]
        assert sent_msg_2.id == "resources-list-2"
