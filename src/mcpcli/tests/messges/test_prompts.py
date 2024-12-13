import pytest
from unittest.mock import patch, AsyncMock
from mcpcli.messages.send_prompts import send_prompts_list
from mcpcli.messages.message_types.prompts_messages import PromptsListMessage

@pytest.mark.asyncio
async def test_send_prompts_list_success():
    mock_response = {"id": "prompts-list-1", "result": ["prompt1", "prompt2"]}
    mock_send_message = AsyncMock(return_value=mock_response)

    # Patch the module where send_prompts_list is defined and where it imports send_message.
    with patch("mcpcli.messages.send_prompts.send_message", new=mock_send_message):
        result = await send_prompts_list(read_stream=None, write_stream=None)
        assert result == ["prompt1", "prompt2"]
        mock_send_message.assert_awaited_once()

        args, kwargs = mock_send_message.call_args
        sent_msg = kwargs["message"]
        assert isinstance(sent_msg, PromptsListMessage)
        assert sent_msg.id == "prompts-list-1"

@pytest.mark.asyncio
async def test_send_prompts_list_no_result():
    mock_response = {"id": "prompts-list-1"}
    mock_send_message = AsyncMock(return_value=mock_response)

    with patch("mcpcli.messages.send_prompts.send_message", new=mock_send_message):
        result = await send_prompts_list(read_stream=None, write_stream=None)
        assert result == []
        mock_send_message.assert_awaited_once()

@pytest.mark.asyncio
async def test_send_prompts_list_error():
    mock_send_message = AsyncMock(side_effect=Exception("Server error"))

    with patch("mcpcli.messages.send_prompts.send_message", new=mock_send_message):
        with pytest.raises(Exception, match="Server error"):
            await send_prompts_list(read_stream=None, write_stream=None)
        mock_send_message.assert_awaited_once()

@pytest.mark.asyncio
async def test_send_prompts_list_increment_id():
    # Reset the counter to ensure test consistency
    PromptsListMessage.load_counter(0)

    # First call
    mock_response_1 = {"id": "prompts-list-1", "result": ["promptA"]}
    mock_send_message = AsyncMock(return_value=mock_response_1)

    with patch("mcpcli.messages.send_prompts.send_message", new=mock_send_message):
        result = await send_prompts_list(read_stream=None, write_stream=None)
        assert result == ["promptA"]
        mock_send_message.assert_awaited_once()

        args, kwargs = mock_send_message.call_args
        sent_msg_1 = kwargs["message"]
        assert sent_msg_1.id == "prompts-list-1"

    # Second call, counter should have incremented
    mock_send_message.reset_mock()
    mock_response_2 = {"id": "prompts-list-2", "result": ["promptB"]}
    mock_send_message.return_value = mock_response_2

    with patch("mcpcli.messages.send_prompts.send_message", new=mock_send_message):
        result = await send_prompts_list(read_stream=None, write_stream=None)
        assert result == ["promptB"]
        mock_send_message.assert_awaited_once()

        args, kwargs = mock_send_message.call_args
        sent_msg_2 = kwargs["message"]
        assert sent_msg_2.id == "prompts-list-2"
