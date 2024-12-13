# tests/test_tools.py
import pytest
from unittest.mock import patch, AsyncMock
from mcpcli.messages.send_call_tool import send_call_tool
from mcpcli.messages.send_tools_list import send_tools_list
from mcpcli.messages.message_types.tools_messages import CallToolMessage, ToolsListMessage

@pytest.mark.asyncio
async def test_send_tools_list_success():
    mock_response = {"id": "tools-list-1", "result": ["toolA", "toolB"]}
    mock_send_message = AsyncMock(return_value=mock_response)

    with patch("mcpcli.messages.send_tools_list.send_message", new=mock_send_message):
        result = await send_tools_list(read_stream=None, write_stream=None)
        assert result == ["toolA", "toolB"]
        mock_send_message.assert_awaited_once()

        args, kwargs = mock_send_message.call_args
        sent_msg = kwargs["message"]
        assert isinstance(sent_msg, ToolsListMessage)
        assert sent_msg.id == "tools-list-1"

@pytest.mark.asyncio
async def test_send_tools_list_increment_id():
    ToolsListMessage.load_counter(0)

    mock_response_1 = {"id": "tools-list-1", "result": ["toolX"]}
    mock_send_message = AsyncMock(return_value=mock_response_1)

    with patch("mcpcli.messages.send_tools_list.send_message", new=mock_send_message):
        result = await send_tools_list(read_stream=None, write_stream=None)
        assert result == ["toolX"]
        mock_send_message.assert_awaited_once()

        args, kwargs = mock_send_message.call_args
        sent_msg_1 = kwargs["message"]
        assert sent_msg_1.id == "tools-list-1"

    mock_send_message.reset_mock()
    mock_response_2 = {"id": "tools-list-2", "result": ["toolY"]}
    mock_send_message.return_value = mock_response_2

    with patch("mcpcli.messages.send_tools_list.send_message", new=mock_send_message):
        result = await send_tools_list(read_stream=None, write_stream=None)
        assert result == ["toolY"]
        mock_send_message.assert_awaited_once()

        args, kwargs = mock_send_message.call_args
        sent_msg_2 = kwargs["message"]
        assert sent_msg_2.id == "tools-list-2"

@pytest.mark.asyncio
async def test_send_call_tool_success():
    mock_response = {"id": "tools-call-1", "result": {"output": "done"}}
    mock_send_message = AsyncMock(return_value=mock_response)

    with patch("mcpcli.messages.send_call_tool.send_message", new=mock_send_message):
        result = await send_call_tool("myTool", {"param": "value"}, read_stream=None, write_stream=None)
        assert result == {"output": "done"}
        mock_send_message.assert_awaited_once()

        args, kwargs = mock_send_message.call_args
        sent_msg = kwargs["message"]
        assert isinstance(sent_msg, CallToolMessage)
        assert sent_msg.id == "tools-call-1"
        assert sent_msg.params == {"name": "myTool", "arguments": {"param": "value"}}

@pytest.mark.asyncio
async def test_send_call_tool_increment_id():
    CallToolMessage.load_counter(0)

    mock_response_1 = {"id": "tools-call-1", "result": {"output": "first"}}
    mock_send_message = AsyncMock(return_value=mock_response_1)

    with patch("mcpcli.messages.send_call_tool.send_message", new=mock_send_message):
        result = await send_call_tool("toolA", {"arg": 1}, read_stream=None, write_stream=None)
        assert result == {"output": "first"}
        mock_send_message.assert_awaited_once()

        args, kwargs = mock_send_message.call_args
        sent_msg_1 = kwargs["message"]
        assert sent_msg_1.id == "tools-call-1"

    mock_send_message.reset_mock()
    mock_response_2 = {"id": "tools-call-2", "result": {"output": "second"}}
    mock_send_message.return_value = mock_response_2

    with patch("mcpcli.messages.send_call_tool.send_message", new=mock_send_message):
        result = await send_call_tool("toolB", {"arg": 2}, read_stream=None, write_stream=None)
        assert result == {"output": "second"}
        mock_send_message.assert_awaited_once()

        args, kwargs = mock_send_message.call_args
        sent_msg_2 = kwargs["message"]
        assert sent_msg_2.id == "tools-call-2"
