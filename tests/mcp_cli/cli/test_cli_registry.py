# tests/test_cli_registry.py

import pytest
import logging
import typer
from typing import Any, Callable

from mcp_cli.cli.registry import CommandRegistry
from mcp_cli.cli.commands.base import BaseCommand, FunctionCommand

# Silence registry logs below WARNING
logging.getLogger("mcp_cli.cli.registry").setLevel(logging.DEBUG)


class DummyCommand(BaseCommand):
    """Concrete stub of BaseCommand for testing."""

    def __init__(self, name: str, help_text: str = ""):
        super().__init__(name, help_text)

    async def execute(self, *args, **kwargs) -> Any:
        """No-op execute."""
        return None


def test_register_and_get():
    CommandRegistry._commands.clear()

    cmd = DummyCommand("foo", "foo help")
    CommandRegistry.register(cmd)

    # get_command returns the same object
    assert CommandRegistry.get_command("foo") is cmd

    # get_all_commands includes it
    all_cmds = CommandRegistry.get_all_commands()
    assert cmd in all_cmds and len(all_cmds) == 1


def test_register_function():
    CommandRegistry._commands.clear()

    def sample_func(ctx: Any):
        """sample help"""
        pass

    CommandRegistry.register_function("bar", sample_func, help_text="bar help")
    cmd = CommandRegistry.get_command("bar")

    # Should be a FunctionCommand wrapping our function
    assert isinstance(cmd, FunctionCommand)
    assert cmd.name == "bar"
    assert cmd.help == "bar help"
    # Underlying function is stored on .func
    assert getattr(cmd, "func", None) is sample_func


def test_register_with_typer():
    CommandRegistry._commands.clear()

    # Register two dummy commands
    cmd1 = DummyCommand("one", "help1")
    cmd2 = DummyCommand("two", "help2")
    CommandRegistry.register(cmd1)
    CommandRegistry.register(cmd2)

    # Fake Typer app supporting .command()
    class DummyApp:
        def __init__(self):
            self.registered = {}
        def command(self, name: str):
            def decorator(fn):
                self.registered[name] = fn
                return fn
            return decorator

    app = DummyApp()
    def runner(fn, config_file, servers, extra_params=None):
        pass

    # Should not raise
    CommandRegistry.register_with_typer(app, runner)

    # Both commands should be wired into app.registered
    assert "one" in app.registered
    assert "two" in app.registered


def test_create_subcommand_group_logs_warning(caplog):
    CommandRegistry._commands.clear()
    # Capture warnings from the registry module
    caplog.set_level(logging.WARNING, logger="mcp_cli.cli.registry")

    # Only register 'tools list'
    cmd_list = DummyCommand("tools list", "list help")
    CommandRegistry.register(cmd_list)

    # Fake app only needs add_typer()
    class FakeApp:
        def add_typer(self, subapp, name):
            pass
    fake_app = FakeApp()

    def runner(fn, config_file, servers, extra_params=None):
        pass

    # Create subcommand group with missing 'tools call'
    CommandRegistry.create_subcommand_group(
        app=fake_app,
        group_name="tools",
        sub_commands=["list", "call"],
        run_command_func=runner
    )

    # Assert a warning was logged about missing 'tools call'
    assert any(
        "Command 'tools call' not found in registry" in rec.message
        for rec in caplog.records
    )
