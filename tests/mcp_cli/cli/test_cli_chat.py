# tests/test_cli_chat_command.py

import pytest
from typing import Any
from unittest.mock import Mock
from mcp_cli.cli.commands.chat import ChatCommand


class DummyToolManager:
    """A simple mock ToolManager that doesn't inherit from the real class."""
    def __init__(self, config_file="", servers=None):
        self.config_file = config_file
        self.servers = servers or []


@pytest.mark.asyncio
async def test_chat_execute_forwards_defaults(monkeypatch):
    """When no override params are passed, execute() should call handle_chat_mode with defaults."""
    captured: dict[str, Any] = {}
    
    # Fix: The real function signature includes all parameters from the actual function
    async def fake_handle(tool_manager, provider, model, api_base=None, api_key=None):
        captured['tool_manager'] = tool_manager
        captured['provider'] = provider
        captured['model'] = model
        captured['api_base'] = api_base
        captured['api_key'] = api_key
        return "CHAT_DONE"

    # Patch the real handle_chat_mode in its module
    monkeypatch.setattr(
        "mcp_cli.chat.chat_handler.handle_chat_mode",
        fake_handle
    )

    cmd = ChatCommand()
    tm = DummyToolManager(config_file="", servers=[])

    # Call execute without params → ChatCommand passes None values, handler applies defaults
    result = await cmd.execute(tool_manager=tm)
    assert result == "CHAT_DONE"
    assert captured['tool_manager'] is tm
    # The ChatCommand passes None values, not defaults - the handler applies defaults
    assert captured['provider'] is None
    assert captured['model'] is None


@pytest.mark.asyncio
async def test_chat_execute_forwards_explicit(monkeypatch):
    """When provider/model overrides are passed, execute() should forward them."""
    captured: dict[str, Any] = {}
    
    # Fix: The real function signature includes all parameters from the actual function
    async def fake_handle(tool_manager, provider, model, api_base=None, api_key=None):
        captured['tool_manager'] = tool_manager
        captured['provider'] = provider
        captured['model'] = model
        captured['api_base'] = api_base
        captured['api_key'] = api_key
        return "OK"

    monkeypatch.setattr(
        "mcp_cli.chat.chat_handler.handle_chat_mode",
        fake_handle
    )

    cmd = ChatCommand()
    tm = DummyToolManager(config_file="", servers=[])

    result = await cmd.execute(
        tool_manager=tm,
        provider="myProv",
        model="myModel"
    )
    assert result == "OK"
    assert captured['tool_manager'] is tm
    assert captured['provider'] == "myProv"
    assert captured['model'] == "myModel"


@pytest.mark.asyncio
async def test_chat_execute_with_partial_params(monkeypatch):
    """Test that partial parameter overrides work correctly."""
    captured: dict[str, Any] = {}
    
    # Fix: The real function signature includes all parameters from the actual function
    async def fake_handle(tool_manager, provider, model, api_base=None, api_key=None):
        captured['tool_manager'] = tool_manager
        captured['provider'] = provider
        captured['model'] = model
        captured['api_base'] = api_base
        captured['api_key'] = api_key
        return "PARTIAL_OK"

    monkeypatch.setattr(
        "mcp_cli.chat.chat_handler.handle_chat_mode",
        fake_handle
    )

    cmd = ChatCommand()
    tm = DummyToolManager(config_file="test.json", servers=["server1"])

    # Test with only provider override
    result = await cmd.execute(
        tool_manager=tm,
        provider="custom_provider"
        # model should use default
    )
    assert result == "PARTIAL_OK"
    assert captured['tool_manager'] is tm
    assert captured['provider'] == "custom_provider"
    # The ChatCommand passes None for model when not specified
    assert captured['model'] is None


@pytest.mark.asyncio
async def test_chat_execute_with_model_only(monkeypatch):
    """Test that model-only override works correctly."""
    captured: dict[str, Any] = {}
    
    # Fix: The real function signature includes all parameters from the actual function
    async def fake_handle(tool_manager, provider, model, api_base=None, api_key=None):
        captured['tool_manager'] = tool_manager
        captured['provider'] = provider
        captured['model'] = model
        captured['api_base'] = api_base
        captured['api_key'] = api_key
        return "MODEL_OK"

    monkeypatch.setattr(
        "mcp_cli.chat.chat_handler.handle_chat_mode",
        fake_handle
    )

    cmd = ChatCommand()
    tm = DummyToolManager()

    # Test with only model override
    result = await cmd.execute(
        tool_manager=tm,
        model="custom_model"
        # provider should use default
    )
    assert result == "MODEL_OK"
    assert captured['tool_manager'] is tm
    # The ChatCommand passes None for provider when not specified
    assert captured['provider'] is None
    assert captured['model'] == "custom_model"