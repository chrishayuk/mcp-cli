"""Tests for the interrupt command definition."""

import pytest
from unittest.mock import MagicMock

from mcp_cli.commands.definitions.interrupt import InterruptCommand
from mcp_cli.commands.base import CommandMode


@pytest.fixture
def interrupt_command():
    """Create an interrupt command instance."""
    return InterruptCommand()


@pytest.fixture
def mock_chat_context():
    """Create a mock chat context."""
    context = MagicMock()
    context.is_streaming = False
    context.is_executing_tool = False
    return context


def test_interrupt_command_properties(interrupt_command):
    """Test interrupt command properties."""
    assert interrupt_command.name == "interrupt"
    assert interrupt_command.aliases == ["stop", "cancel"]
    assert interrupt_command.description == "Interrupt currently running operations"
    assert interrupt_command.modes == CommandMode.CHAT
    assert interrupt_command.requires_context is True
    assert "Interrupt currently running" in interrupt_command.help_text
    assert "streaming-aware" in interrupt_command.help_text


@pytest.mark.asyncio
async def test_interrupt_without_context(interrupt_command):
    """Test interrupt command without chat context."""
    result = await interrupt_command.execute()

    assert result.success is False
    assert result.error == "Interrupt command requires chat context."


@pytest.mark.asyncio
async def test_interrupt_streaming(interrupt_command, mock_chat_context):
    """Test interrupting streaming response."""
    mock_chat_context.is_streaming = True
    mock_chat_context.interrupt_streaming = MagicMock()

    result = await interrupt_command.execute(chat_context=mock_chat_context)

    assert result.success is True
    assert result.output == "Streaming response interrupted."
    mock_chat_context.interrupt_streaming.assert_called_once()


@pytest.mark.asyncio
async def test_interrupt_tool_execution(interrupt_command, mock_chat_context):
    """Test interrupting tool execution."""
    mock_chat_context.is_executing_tool = True
    mock_chat_context.interrupt_tool_execution = MagicMock()

    result = await interrupt_command.execute(chat_context=mock_chat_context)

    assert result.success is True
    assert result.output == "Tool execution interrupted."
    mock_chat_context.interrupt_tool_execution.assert_called_once()


@pytest.mark.asyncio
async def test_interrupt_general_operation(interrupt_command, mock_chat_context):
    """Test general operation cancellation."""
    mock_chat_context.cancel_current_operation = MagicMock()

    result = await interrupt_command.execute(chat_context=mock_chat_context)

    assert result.success is True
    assert result.output == "Current operation cancelled."
    mock_chat_context.cancel_current_operation.assert_called_once()


@pytest.mark.asyncio
async def test_interrupt_nothing_to_interrupt(interrupt_command, mock_chat_context):
    """Test interrupt when nothing is running."""
    # Context has no interrupt methods and nothing is running
    # But mock context has cancel_current_operation by default
    mock_chat_context.cancel_current_operation = MagicMock()
    result = await interrupt_command.execute(chat_context=mock_chat_context)

    assert result.success is True
    assert result.output == "Current operation cancelled."


@pytest.mark.asyncio
async def test_interrupt_streaming_priority(interrupt_command, mock_chat_context):
    """Test that streaming interrupt has priority over tool execution."""
    # Both streaming and tool execution are active
    mock_chat_context.is_streaming = True
    mock_chat_context.is_executing_tool = True
    mock_chat_context.interrupt_streaming = MagicMock()
    mock_chat_context.interrupt_tool_execution = MagicMock()

    result = await interrupt_command.execute(chat_context=mock_chat_context)

    # Should interrupt streaming first
    assert result.success is True
    assert result.output == "Streaming response interrupted."
    mock_chat_context.interrupt_streaming.assert_called_once()
    mock_chat_context.interrupt_tool_execution.assert_not_called()


@pytest.mark.asyncio
async def test_interrupt_without_methods(interrupt_command):
    """Test interrupt with context that lacks interrupt methods."""
    # Context exists but has no interrupt-related attributes
    bare_context = object()

    result = await interrupt_command.execute(chat_context=bare_context)

    assert result.success is True
    assert result.output == "Nothing to interrupt."


@pytest.mark.asyncio
async def test_interrupt_really_nothing_to_interrupt(interrupt_command):
    """Test interrupt when truly nothing is running and no methods exist."""
    # Create a mock context that explicitly has no interrupt-related attributes
    mock_context = MagicMock(spec=["some_other_attr"])
    mock_context.is_streaming = False
    mock_context.is_executing_tool = False

    result = await interrupt_command.execute(chat_context=mock_context)

    assert result.success is True
    assert result.output == "Nothing to interrupt."
