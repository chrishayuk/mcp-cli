# tests/commands/test_clear.py
import pytest

from mcp_cli.interactive.commands.clear import ClearCommand
import mcp_cli.commands.clear as clear_module  # patch here

@pytest.mark.asyncio
async def test_clear_command_calls_clear_screen(monkeypatch):
    called = {"count": 0}
    def fake_clear_screen():
        called["count"] += 1

    # Now patch the shared clear_screen that clear_action() calls
    monkeypatch.setattr(clear_module, "clear_screen", fake_clear_screen)

    cmd = ClearCommand()
    result = await cmd.execute([], tool_manager=None)

    assert called["count"] == 1, "clear_screen should be called exactly once"
    assert result is None
