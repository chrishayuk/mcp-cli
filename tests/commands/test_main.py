# tests/commands/test_main.py
"""Tests for the main Typer app in src/mcp_cli/main.py."""

from typer.testing import CliRunner

from mcp_cli.main import app

runner = CliRunner()


def test_ping_command_exists():
    """Test that 'mcp-cli ping' command is registered."""
    result = runner.invoke(app, ["ping", "--help"])
    assert result.exit_code == 0
    assert "Test connectivity to MCP servers" in result.stdout

def test_all_direct_commands_are_registered():
    """
    Test that all commands intended for direct registration are present.
    This prevents commands from being accidentally removed.
    """
    # This list should be kept in sync with the commands registered in main.py
    expected_commands = [
        "chat",
        "interactive",
        "provider",
        "providers",
        "tools",
        "servers",
        "resources",
        "prompts",
        "models",
        "theme",
        "token",
        "tokens",
        "cmd",
        "ping",
    ]

    registered_commands = [cmd.name for cmd in app.registered_commands if cmd.name]

    for cmd_name in expected_commands:
        assert cmd_name in registered_commands, f"Command '{cmd_name}' is not registered in main.py"
