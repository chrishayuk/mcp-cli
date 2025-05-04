# tests/test_cli_interactive_command.py
import pytest
import asyncio

from mcp_cli.cli.commands.interactive import InteractiveCommand
from mcp_cli.tools.manager import ToolManager

class DummyTM(ToolManager):
    pass

@pytest.mark.asyncio
async def test_execute_forwards_defaults(monkeypatch):
    """If no params passed, execute() should call interactive_mode with defaults."""
    captured = {}
    async def fake_im(tool_manager, provider, model, server_names=None):
        captured['tm'] = tool_manager
        captured['provider'] = provider
        captured['model'] = model
        captured['servers'] = server_names
        return "RESULT"

    # Patch the real interactive_mode in its defining module
    monkeypatch.setattr(
        "mcp_cli.interactive.shell.interactive_mode",
        fake_im
    )

    cmd = InteractiveCommand()
    tm = DummyTM(config_file="", servers=[])

    result = await cmd.execute(tool_manager=tm)
    assert result == "RESULT"
    assert captured['tm'] is tm
    assert captured['provider'] == "openai"
    assert captured['model'] == "gpt-4o-mini"
    assert captured['servers'] is None

@pytest.mark.asyncio
async def test_execute_forwards_explicit_params(monkeypatch):
    """If provider/model/server_names passed in, execute() should forward them."""
    captured = {}
    async def fake_im(tool_manager, provider, model, server_names=None):
        captured['tm'] = tool_manager
        captured['provider'] = provider
        captured['model'] = model
        captured['servers'] = server_names
        return "OK"

    monkeypatch.setattr(
        "mcp_cli.interactive.shell.interactive_mode",
        fake_im
    )

    cmd = InteractiveCommand()
    tm = DummyTM(config_file="", servers=[])

    params = {
        "provider": "myprov",
        "model": "my-model",
        "server_names": {0: "one", 1: "two"}
    }
    result = await cmd.execute(tool_manager=tm, **params)
    assert result == "OK"
    assert captured['tm'] is tm
    assert captured['provider'] == "myprov"
    assert captured['model'] == "my-model"
    assert captured['servers'] == {0: "one", 1: "two"}
