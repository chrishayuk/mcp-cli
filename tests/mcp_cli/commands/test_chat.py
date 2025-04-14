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

def test_ping_command():
    """Test the ping_command function."""
    # Here's the key change: patch the module where process_options is actually imported from
    with patch("mcp_cli.cli_options.process_options") as mock_process_options, \
         patch("mcp_cli.commands.register_commands.run_command") as mock_run_command:
        
        # Setup mock return value
        mock_process_options.return_value = (["server1"], ["server1"], {"0": "Server1"})
        
        # Call the function
        result = register_commands.ping_command(
            config_file="test_config.json",
            server="server1",
            provider="test-provider",
            model="test-model",
            disable_filesystem=True
        )
        
        # Verify process_options was called with the right arguments
        mock_process_options.assert_called_with(
            "server1", True, "test-provider", "test-model", "test_config.json"
        )
        
        # Verify run_command was called with the right arguments
        mock_run_command.assert_called_once()
        assert mock_run_command.call_args[0][0] == register_commands.ping.ping_run
        assert mock_run_command.call_args[0][1] == "test_config.json"
        assert mock_run_command.call_args[0][2] == ["server1"]
        assert mock_run_command.call_args[0][3] == ["server1"]
        assert mock_run_command.call_args[0][4] == {"server_names": {"0": "Server1"}}
        
        # Verify the return value
        assert result == 0

def test_chat_command():
    """Test the chat_command function."""
    with patch("mcp_cli.cli_options.process_options") as mock_process_options, \
         patch("mcp_cli.commands.register_commands.run_command") as mock_run_command:
        
        # Setup mock return value
        mock_process_options.return_value = (["server1"], ["server1"], {"0": "Server1"})
        
        # Call the function
        result = register_commands.chat_command(
            config_file="test_config.json",
            server="server1",
            provider="test-provider",
            model="test-model",
            disable_filesystem=True
        )
        
        # Verify process_options was called with the right arguments
        mock_process_options.assert_called_with(
            "server1", True, "test-provider", "test-model", "test_config.json"
        )
        
        # Verify run_command was called with the right arguments
        mock_run_command.assert_called_once()
        assert mock_run_command.call_args[0][0] == register_commands.chat.chat_run
        assert mock_run_command.call_args[0][1] == "test_config.json"
        assert mock_run_command.call_args[0][2] == ["server1"]
        assert mock_run_command.call_args[0][3] == ["server1"]
        assert mock_run_command.call_args[0][4] == {"server_names": {"0": "Server1"}}
        
        # Verify the return value
        assert result == 0

def test_interactive_command():
    """Test the interactive_command function."""
    with patch("mcp_cli.cli_options.process_options") as mock_process_options, \
         patch("mcp_cli.commands.register_commands.run_command") as mock_run_command:
        
        # Setup mock return value
        mock_process_options.return_value = (["server1"], ["server1"], {"0": "Server1"})
        
        # Call the function
        result = register_commands.interactive_command(
            config_file="test_config.json",
            server="server1",
            provider="test-provider",
            model="test-model",
            disable_filesystem=True
        )
        
        # Verify process_options was called with the right arguments
        mock_process_options.assert_called_with(
            "server1", True, "test-provider", "test-model", "test_config.json"
        )
        
        # Verify run_command was called with the right arguments
        # Note: interactive_command doesn't pass server_names
        mock_run_command.assert_called_once()
        assert mock_run_command.call_args[0][0] == register_commands.interactive.interactive_mode
        assert mock_run_command.call_args[0][1] == "test_config.json"
        assert mock_run_command.call_args[0][2] == ["server1"]
        assert mock_run_command.call_args[0][3] == ["server1"]
        assert len(mock_run_command.call_args[0]) == 4  # No extra params
        
        # Verify the return value
        assert result == 0

def test_prompts_list_command():
    """Test the prompts_list_command function."""
    with patch("mcp_cli.cli_options.process_options") as mock_process_options, \
         patch("mcp_cli.commands.register_commands.run_command") as mock_run_command:
        
        # Setup mock return value
        mock_process_options.return_value = (["server1"], ["server1"], {"0": "Server1"})
        
        # Call the function
        result = register_commands.prompts_list_command(
            config_file="test_config.json",
            server="server1",
            provider="test-provider",
            model="test-model",
            disable_filesystem=True
        )
        
        # Verify process_options was called with the right arguments
        mock_process_options.assert_called_with(
            "server1", True, "test-provider", "test-model", "test_config.json"
        )
        
        # Verify run_command was called with the right arguments
        mock_run_command.assert_called_once()
        assert mock_run_command.call_args[0][0] == register_commands.prompts.prompts_list
        assert mock_run_command.call_args[0][1] == "test_config.json"
        assert mock_run_command.call_args[0][2] == ["server1"]
        assert mock_run_command.call_args[0][3] == ["server1"]
        assert mock_run_command.call_args[0][4] == {"server_names": {"0": "Server1"}}
        
        # Verify the return value
        assert result == 0

def test_tools_list_command():
    """Test the tools_list_command function."""
    with patch("mcp_cli.cli_options.process_options") as mock_process_options, \
         patch("mcp_cli.commands.register_commands.run_command") as mock_run_command:
        
        # Setup mock return value
        mock_process_options.return_value = (["server1"], ["server1"], {"0": "Server1"})
        
        # Call the function
        result = register_commands.tools_list_command(
            config_file="test_config.json",
            server="server1",
            provider="test-provider",
            model="test-model",
            disable_filesystem=True
        )
        
        # Verify process_options was called with the right arguments
        mock_process_options.assert_called_with(
            "server1", True, "test-provider", "test-model", "test_config.json"
        )
        
        # Verify run_command was called with the right arguments
        mock_run_command.assert_called_once()
        assert mock_run_command.call_args[0][0] == register_commands.tools.tools_list
        assert mock_run_command.call_args[0][1] == "test_config.json"
        assert mock_run_command.call_args[0][2] == ["server1"]
        assert mock_run_command.call_args[0][3] == ["server1"]
        assert mock_run_command.call_args[0][4] == {"server_names": {"0": "Server1"}}
        
        # Verify the return value
        assert result == 0

def test_tools_call_command():
    """Test the tools_call_command function."""
    with patch("mcp_cli.cli_options.process_options") as mock_process_options, \
         patch("mcp_cli.commands.register_commands.run_command") as mock_run_command:
        
        # Setup mock return value
        mock_process_options.return_value = (["server1"], ["server1"], {"0": "Server1"})
        
        # Call the function
        result = register_commands.tools_call_command(
            config_file="test_config.json",
            server="server1",
            provider="test-provider",
            model="test-model",
            disable_filesystem=True
        )
        
        # Verify process_options was called with the right arguments
        mock_process_options.assert_called_with(
            "server1", True, "test-provider", "test-model", "test_config.json"
        )
        
        # Verify run_command was called with the right arguments
        mock_run_command.assert_called_once()
        assert mock_run_command.call_args[0][0] == register_commands.tools.tools_call
        assert mock_run_command.call_args[0][1] == "test_config.json"
        assert mock_run_command.call_args[0][2] == ["server1"]
        assert mock_run_command.call_args[0][3] == ["server1"]
        assert mock_run_command.call_args[0][4] == {"server_names": {"0": "Server1"}}
        
        # Verify the return value
        assert result == 0

def test_resources_list_command():
    """Test the resources_list_command function."""
    with patch("mcp_cli.cli_options.process_options") as mock_process_options, \
         patch("mcp_cli.commands.register_commands.run_command") as mock_run_command:
        
        # Setup mock return value
        mock_process_options.return_value = (["server1"], ["server1"], {"0": "Server1"})
        
        # Call the function
        result = register_commands.resources_list_command(
            config_file="test_config.json",
            server="server1",
            provider="test-provider",
            model="test-model",
            disable_filesystem=True
        )
        
        # Verify process_options was called with the right arguments
        mock_process_options.assert_called_with(
            "server1", True, "test-provider", "test-model", "test_config.json"
        )
        
        # Verify run_command was called with the right arguments
        mock_run_command.assert_called_once()
        assert mock_run_command.call_args[0][0] == register_commands.resources.resources_list
        assert mock_run_command.call_args[0][1] == "test_config.json"
        assert mock_run_command.call_args[0][2] == ["server1"]
        assert mock_run_command.call_args[0][3] == ["server1"]
        assert mock_run_command.call_args[0][4] == {"server_names": {"0": "Server1"}}
        
        # Verify the return value
        assert result == 0

def test_cmd_command():
    """Test the cmd_command function."""
    with patch("mcp_cli.cli_options.process_options") as mock_process_options, \
         patch("mcp_cli.commands.register_commands.run_command") as mock_run_command:
        
        # Setup mock return value
        mock_process_options.return_value = (["server1"], ["server1"], {"0": "Server1"})
        
        # Call the function with all parameters
        result = register_commands.cmd_command(
            config_file="test_config.json",
            server="server1",
            provider="test-provider",
            model="test-model",
            disable_filesystem=True,
            input="test_input.txt",
            prompt="test prompt",
            output="test_output.txt",
            raw=True,
            tool="test_tool",
            tool_args='{"param": "value"}',
            system_prompt="test system prompt"
        )
        
        # Verify process_options was called with the right arguments
        mock_process_options.assert_called_with(
            "server1", True, "test-provider", "test-model", "test_config.json"
        )
        
        # Verify run_command was called with the right arguments
        mock_run_command.assert_called_once()
        assert mock_run_command.call_args[0][0] == register_commands.cmd.cmd_run
        assert mock_run_command.call_args[0][1] == "test_config.json"
        assert mock_run_command.call_args[0][2] == ["server1"]
        assert mock_run_command.call_args[0][3] == ["server1"]
        
        # Check that all parameters were passed in extra_params
        extra_params = mock_run_command.call_args[0][4]
        assert extra_params["input"] == "test_input.txt"
        assert extra_params["prompt"] == "test prompt"
        assert extra_params["output"] == "test_output.txt"
        assert extra_params["raw"] is True
        assert extra_params["tool"] == "test_tool"
        assert extra_params["tool_args"] == '{"param": "value"}'
        assert extra_params["system_prompt"] == "test system prompt"
        assert extra_params["server_names"] == {"0": "Server1"}
        
        # Verify the return value
        assert result == 0

def test_cmd_command_defaults():
    """Test the cmd_command function with default parameters."""
    with patch("mcp_cli.cli_options.process_options") as mock_process_options, \
         patch("mcp_cli.commands.register_commands.run_command") as mock_run_command:
        
        # Setup mock return value
        mock_process_options.return_value = (["server1"], ["server1"], {"0": "Server1"})
        
        # Call the function with only required parameters
        result = register_commands.cmd_command()
        
        # Verify process_options was called with default arguments
        mock_process_options.assert_called_with(
            None, False, "openai", None, "server_config.json"
        )
        
        # Verify run_command was called with the right arguments
        mock_run_command.assert_called_once()
        
        # Check that all parameters were passed in extra_params with default values
        extra_params = mock_run_command.call_args[0][4]
        assert extra_params["input"] is None
        assert extra_params["prompt"] is None
        assert extra_params["output"] is None
        assert extra_params["raw"] is False
        assert extra_params["tool"] is None
        assert extra_params["tool_args"] is None
        assert extra_params["system_prompt"] is None
        assert extra_params["server_names"] == {"0": "Server1"}
        
        # Verify the return value
        assert result == 0