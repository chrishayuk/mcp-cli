# tests/test_cli_chat_command.py

import pytest
from typing import Any
from mcp_cli.cli.commands.chat import ChatCommand
from mcp_cli.tools.manager import ToolManager

class DummyTM(ToolManager):
    pass

@pytest.mark.asyncio
async def test_chat_execute_forwards_defaults(monkeypatch):
    """When no override params are passed, execute() should call handle_chat_mode with defaults."""
    captured: dict[str, Any] = {}
    async def fake_handle(tm, provider, model):
        captured['tm'] = tm
        captured['provider'] = provider
        captured['model'] = model
        return "CHAT_DONE"

    # Patch the real handle_chat_mode in its module
    monkeypatch.setattr(
        "mcp_cli.chat.chat_handler.handle_chat_mode",
        fake_handle
    )

    cmd = ChatCommand()
    tm = DummyTM(config_file="", servers=[])

    # Call execute without params → uses default provider/model
    result = await cmd.execute(tool_manager=tm)
    assert result == "CHAT_DONE"
    assert captured['tm'] is tm
    assert captured['provider'] == "openai"
    assert captured['model'] == "gpt-4o-mini"

@pytest.mark.asyncio
async def test_chat_execute_forwards_explicit(monkeypatch):
    """When provider/model overrides are passed, execute() should forward them."""
    captured: dict[str, Any] = {}
    async def fake_handle(tm, provider, model):
        captured['tm'] = tm
        captured['provider'] = provider
        captured['model'] = model
        return "OK"

    monkeypatch.setattr(
        "mcp_cli.chat.chat_handler.handle_chat_mode",
        fake_handle
    )

    cmd = ChatCommand()
    tm = DummyTM(config_file="", servers=[])

    result = await cmd.execute(
        tool_manager=tm,
        provider="myProv",
        model="myModel"
    )
    assert result == "OK"
    assert captured['tm'] is tm
    assert captured['provider'] == "myProv"
    assert captured['model'] == "myModel"
