# mcp_cli/interactive/test_interactice_registry.py
import sys
import types
import pytest
from typing import List, Any

# ─── Stub out mcp_cli.interactive.commands so registry can import safely ───
# Create a dummy package and module for `mcp_cli.interactive.commands`
dummy_pkg = types.ModuleType("mcp_cli.interactive.commands")
dummy_base = types.ModuleType("mcp_cli.interactive.commands.base")
# Give the dummy_base a minimal InteractiveCommand class
class InteractiveCommand:
    def __init__(self, name, help_text="", aliases=None):
        self.name = name
        self.aliases = aliases or []
dummy_base.InteractiveCommand = InteractiveCommand

# Insert into sys.modules so that `import mcp_cli.interactive.commands.base` works
sys.modules["mcp_cli.interactive.commands"] = dummy_pkg
sys.modules["mcp_cli.interactive.commands.base"] = dummy_base

# ─── Now import the registry itself ────────────────────────────────────────
from mcp_cli.interactive.registry import InteractiveCommandRegistry


class DummyCommand(InteractiveCommand):
    """
    Minimal command-like object for testing the registry.
    Inherits from the stubbed InteractiveCommand to satisfy any isinstance checks.
    """
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
    assert "foo" in all_cmds
    assert all_cmds["foo"] is cmd


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
