import pytest
from unittest.mock import MagicMock, patch, call
import typer

from mcp_cli.commands import register_commands

@pytest.fixture
def mock_typer_app():
    """Create a mock Typer app for testing."""
    app = MagicMock(spec=typer.Typer)
    app.add_typer = MagicMock()
    app.command = MagicMock()
    
    # Mock the behavior of app.command() to return a decorator
    mock_decorator = MagicMock()
    app.command.return_value = mock_decorator
    
    return app

@pytest.fixture
def mock_process_options():
    """Create a mock process_options function."""
    mock_func = MagicMock()
    mock_func.return_value = (["server1"], ["server1"], {"0": "Server1"})
    return mock_func

@pytest.fixture
def mock_run_command():
    """Create a mock run_command function."""
    return MagicMock()

def test_register_commands_function(mock_typer_app, mock_process_options, mock_run_command):
    """Test the register_commands function."""
    # Call the function
    register_commands.register_commands(mock_typer_app, mock_process_options, mock_run_command)
    
    # Verify that app.command was called for each top-level command
    assert mock_typer_app.command.call_count == 4
    mock_typer_app.command.assert_any_call("ping")
    mock_typer_app.command.assert_any_call("chat")
    mock_typer_app.command.assert_any_call("interactive")
    mock_typer_app.command.assert_any_call("cmd")
    
    # Verify that add_typer was called for each sub-command app
    assert mock_typer_app.add_typer.call_count == 3
    # Check that add_typer was called with the right names
    added_names = [call_args.kwargs.get('name') for call_args in mock_typer_app.add_typer.call_args_list]
    assert "prompts" in added_names
    assert "tools" in added_names
    assert "resources" in added_names