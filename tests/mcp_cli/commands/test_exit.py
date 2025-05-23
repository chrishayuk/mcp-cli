# tests/commands/test_exit.py
import pytest

from mcp_cli.interactive.commands.exit import ExitCommand
import mcp_cli.commands.exit as exit_module  # patch the shared module

@pytest.mark.asyncio
async def test_exit_command_prints_and_returns_true(monkeypatch):
    # Arrange: capture print calls from exit_action()
    printed = []
    def fake_print(*args, **kwargs):
        printed.append(args[0])

    # exit_action() does `from rich import print`, so patch here:
    monkeypatch.setattr(exit_module, "print", fake_print)
    monkeypatch.setattr(exit_module, "restore_terminal", lambda: None)

    cmd = ExitCommand()
    result = await cmd.execute([], tool_manager=None)

    assert result is True
    assert any("Exiting… Goodbye!" in str(p) for p in printed)


