# tests/interactive/test_interactive_shell.py
import pytest
import builtins
import rich
from rich.console import Console

from mcp_cli.interactive.shell import interactive_mode
from mcp_cli.interactive.registry import InteractiveCommandRegistry

class DummyTM:
    pass

@pytest.fixture(autouse=True)
def clear_registry():
    InteractiveCommandRegistry._commands.clear()
    InteractiveCommandRegistry._aliases.clear()
    yield
    InteractiveCommandRegistry._commands.clear()
    InteractiveCommandRegistry._aliases.clear()

@pytest.mark.asyncio
async def test_exit_immediately(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt="": "exit")
    printed = []
    monkeypatch.setattr(Console, "print", lambda self, msg, **kw: printed.append(msg))
    monkeypatch.setattr(rich, "print", lambda *args, **kwargs: printed.append("".join(map(str,args))))

    result = await interactive_mode(tool_manager=DummyTM())
    assert result is True
    assert any("Exiting interactive mode" in s for s in printed if isinstance(s, str))

@pytest.mark.asyncio
async def test_unknown_then_exit(monkeypatch):
    inputs = iter(["foo", "exit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

    printed = []
    monkeypatch.setattr(Console, "print", lambda self, msg, **kw: printed.append(msg))
    monkeypatch.setattr(rich, "print", lambda *args, **kwargs: printed.append("".join(map(str,args))))

    result = await interactive_mode(tool_manager=DummyTM())
    assert result is True

    # filter to strings
    strings = [s for s in printed if isinstance(s, str)]

    # Look for the colored "Unknown command" line
    assert any("Unknown command" in s and "foo" in s for s in strings), f"No unknown-command in {strings}"

    # Also check exit message
    assert any("Exiting interactive mode" in s for s in strings)
