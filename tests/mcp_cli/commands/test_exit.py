# tests/commands/test_exit.py
import pytest
from unittest.mock import Mock

from mcp_cli.interactive.commands.exit import ExitCommand
import mcp_cli.commands.exit as exit_module


@pytest.mark.asyncio
async def test_exit_command_prints_and_returns_true(monkeypatch):
    # Arrange: mock the console and its print method
    mock_console = Mock()
    printed_messages = []
    
    def capture_print(message):
        printed_messages.append(message)
    
    mock_console.print = capture_print
    
    # Patch get_console to return our mock
    monkeypatch.setattr(exit_module, "get_console", lambda: mock_console)
    monkeypatch.setattr(exit_module, "restore_terminal", lambda: None)

    # Act
    cmd = ExitCommand()
    result = await cmd.execute([], tool_manager=None)

    # Assert
    assert result is True
    assert len(printed_messages) == 1
    assert "Exiting… Goodbye!" in printed_messages[0]