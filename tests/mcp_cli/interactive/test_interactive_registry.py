# mcp_cli/interactive/test_interactice_registry.py
import pytest
from typing import List, Any
from mcp_cli.interactive.registry import InteractiveCommandRegistry
from mcp_cli.interactive.commands.base import InteractiveCommand


class DummyCommand(InteractiveCommand):
    def __init__(self, name: str, aliases: List[str] = None):
        super().__init__(name=name, help_text=f"help for {name}", aliases=aliases or [])

    async def execute(self, args: List[str], tool_manager: Any = None, **kwargs):
        return f"{self.name}-ran"


@pytest.fixture(autouse=True)
def clear_registry():
    # Ensure registry is empty before each test
    InteractiveCommandRegistry._commands.clear()
    InteractiveCommandRegistry._aliases.clear()
    yield
    InteractiveCommandRegistry._commands.clear()
    InteractiveCommandRegistry._aliases.clear()


def test_register_and_get_by_name():
    cmd = DummyCommand("foo")
    InteractiveCommandRegistry.register(cmd)

    # get_command by its name
    got = InteractiveCommandRegistry.get_command("foo")
    assert got is cmd

    # get_all_commands includes it
    all_cmds = InteractiveCommandRegistry.get_all_commands()
    assert "foo" in all_cmds and all_cmds["foo"] is cmd


def test_register_with_aliases():
    cmd = DummyCommand("bar", aliases=["b", "baz"])
    InteractiveCommandRegistry.register(cmd)

    # direct by name
    assert InteractiveCommandRegistry.get_command("bar") is cmd
    # lookup via each alias
    assert InteractiveCommandRegistry.get_command("b") is cmd
    assert InteractiveCommandRegistry.get_command("baz") is cmd

    # _aliases maps aliases back to "bar"
    assert InteractiveCommandRegistry._aliases["b"] == "bar"
    assert InteractiveCommandRegistry._aliases["baz"] == "bar"


def test_get_missing_command_returns_none():
    # No registration at all
    assert InteractiveCommandRegistry.get_command("nonexistent") is None
    # Alias also not found
    assert InteractiveCommandRegistry.get_command("x") is None
