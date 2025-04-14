import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call
import sys
from io import StringIO

# Import the module to test
from mcp_cli.commands import interactive

@pytest.fixture
def mock_stream_manager():
    """Create a mock StreamManager with predefined test data."""
    mock_manager = MagicMock()
    
    # Set up server info
    mock_manager.get_server_info.return_value = [
        {
            "id": 1, 
            "name": "TestServer1", 
            "tools": 3, 
            "status": "Connected",
            "tool_start_index": 0
        },
        {
            "id": 2, 
            "name": "TestServer2", 
            "tools": 2, 
            "status": "Connected",
            "tool_start_index": 3
        }
    ]
    
    # Set up tools
    mock_manager.get_all_tools.return_value = [
        {"name": "tool1", "description": "Test tool 1"},
        {"name": "tool2", "description": "Test tool 2"},
        {"name": "tool3", "description": "Test tool 3"},
        {"name": "tool4", "description": "Test tool 4"},
        {"name": "tool5", "description": "Test tool 5"}
    ]
    
    # Set up tool to server map
    mock_manager.tool_to_server_map = {
        "tool1": "TestServer1",
        "tool2": "TestServer1",
        "tool3": "TestServer1",
        "tool4": "TestServer2",
        "tool5": "TestServer2"
    }
    
    # Set up server streams map
    mock_manager.server_streams_map = {
        "TestServer1": 0,
        "TestServer2": 1
    }
    
    # Set up streams
    mock_read_stream1 = AsyncMock()
    mock_write_stream1 = AsyncMock()
    mock_read_stream2 = AsyncMock()
    mock_write_stream2 = AsyncMock()
    
    mock_manager.streams = [
        (mock_read_stream1, mock_write_stream1),
        (mock_read_stream2, mock_write_stream2)
    ]
    
    return mock_manager

@pytest.mark.asyncio
async def test_display_interactive_banner(mock_stream_manager, capsys):
    """Test that the interactive banner is displayed correctly."""
    # Create a test context
    context = {
        "provider": "test-provider",
        "model": "test-model",
        "tools": mock_stream_manager.get_all_tools(),
        "server_info": mock_stream_manager.get_server_info(),
        "tool_to_server_map": mock_stream_manager.tool_to_server_map,
        "stream_manager": mock_stream_manager
    }
    
    # Call the function
    interactive.display_interactive_banner(context)
    
    # Check output
    captured = capsys.readouterr()
    assert "Interactive Mode" in captured.out
    assert "test-provider" in captured.out
    assert "test-model" in captured.out
    assert "Loaded 5 tools successfully" in captured.out
    assert "Type '/servers' to see connected servers" in captured.out

@pytest.mark.asyncio
async def test_display_servers_info(mock_stream_manager, capsys):
    """Test that server information is displayed correctly."""
    # Create a test context
    context = {
        "server_info": mock_stream_manager.get_server_info(),
        "tools": mock_stream_manager.get_all_tools()
    }
    
    # Call the function
    interactive.display_servers_info(context)
    
    # Check output
    captured = capsys.readouterr()
    assert "Connected Servers" in captured.out
    assert "TestServer1" in captured.out
    assert "TestServer2" in captured.out
    assert "Use the /tools command" in captured.out

@pytest.mark.asyncio
async def test_display_servers_info_no_servers(capsys):
    """Test server info display with no servers."""
    # Create a test context with no servers
    context = {
        "server_info": [],
        "tools": []
    }
    
    # Call the function
    interactive.display_servers_info(context)
    
    # Check output
    captured = capsys.readouterr()
    assert "No servers connected" in captured.out

@pytest.mark.asyncio
async def test_show_help(capsys):
    """Test that the help message is displayed correctly."""
    # Call the function
    interactive.show_help()
    
    # Check output
    captured = capsys.readouterr()
    assert "Available Commands" in captured.out
    assert "/ping" in captured.out
    assert "/prompts" in captured.out
    assert "/tools" in captured.out
    assert "/resources" in captured.out
    assert "/servers" in captured.out
    assert "/chat" in captured.out
    assert "/exit" in captured.out

@pytest.mark.asyncio
async def test_clear_screen_cmd(monkeypatch, capsys):
    """Test the clear screen command."""
    # Mock the clear_screen function
    monkeypatch.setattr(interactive, "clear_screen", lambda: None)
    
    # Call the function with welcome banner
    interactive.clear_screen_cmd(with_welcome=True)
    
    # Check output (should include the welcome banner)
    captured = capsys.readouterr()
    assert "Interactive Mode" in captured.out
    
    # Call the function without welcome banner
    interactive.clear_screen_cmd(with_welcome=False)
    
    # Check output (should be empty)
    captured = capsys.readouterr()
    assert captured.out == ""

@pytest.mark.asyncio
async def test_interactive_mode_exit_command(mock_stream_manager, monkeypatch):
    """Test that the exit command works correctly."""
    # Mock Prompt.ask to return exit
    mock_session = MagicMock(return_value="exit")
    monkeypatch.setattr(interactive, "Prompt", MagicMock(ask=mock_session))
    
    # Run interactive mode
    result = await interactive.interactive_mode(mock_stream_manager)
    
    # Check that it returns True (clean exit)
    assert result is True

@pytest.mark.asyncio
async def test_interactive_mode_quit_command(mock_stream_manager, monkeypatch):
    """Test that the quit command works correctly."""
    # Mock Prompt.ask to return quit
    mock_session = MagicMock(return_value="quit")
    monkeypatch.setattr(interactive, "Prompt", MagicMock(ask=mock_session))
    
    # Run interactive mode
    result = await interactive.interactive_mode(mock_stream_manager)
    
    # Check that it returns True (clean exit)
    assert result is True

@pytest.mark.asyncio
async def test_interactive_mode_slash_exit(mock_stream_manager, monkeypatch):
    """Test that the /exit command works correctly."""
    # Mock Prompt.ask to return /exit
    mock_session = MagicMock(return_value="/exit")
    monkeypatch.setattr(interactive, "Prompt", MagicMock(ask=mock_session))
    
    # Run interactive mode
    result = await interactive.interactive_mode(mock_stream_manager)
    
    # Check that it returns True (clean exit)
    assert result is True

@pytest.mark.asyncio
async def test_interactive_mode_help_command(mock_stream_manager, monkeypatch, capsys):
    """Test that the /help command works correctly."""
    # Set up mock to return /help and then exit
    responses = ["/help", "exit"]
    mock_session = MagicMock(side_effect=responses)
    monkeypatch.setattr(interactive, "Prompt", MagicMock(ask=mock_session))
    
    # Mock show_help to track calls
    mock_show_help = MagicMock()
    monkeypatch.setattr(interactive, "show_help", mock_show_help)
    
    # Run interactive mode
    await interactive.interactive_mode(mock_stream_manager)
    
    # Check that show_help was called
    mock_show_help.assert_called_once()

@pytest.mark.asyncio
async def test_interactive_mode_ping_command(mock_stream_manager, monkeypatch):
    """Test that the /ping command works correctly."""
    # Set up mock to return /ping and then exit
    responses = ["/ping", "exit"]
    mock_session = MagicMock(side_effect=responses)
    monkeypatch.setattr(interactive, "Prompt", MagicMock(ask=mock_session))
    
    # Mock ping.ping_run
    mock_ping_run = AsyncMock()
    monkeypatch.setattr(interactive.ping, "ping_run", mock_ping_run)
    
    # Run interactive mode
    await interactive.interactive_mode(mock_stream_manager)
    
    # Check that ping_run was called with the stream_manager
    mock_ping_run.assert_called_once()
    assert mock_ping_run.call_args[1]["stream_manager"] == mock_stream_manager

@pytest.mark.asyncio
async def test_interactive_mode_prompts_command(mock_stream_manager, monkeypatch):
    """Test that the /prompts command works correctly."""
    # Set up mock to return /prompts and then exit
    responses = ["/prompts", "exit"]
    mock_session = MagicMock(side_effect=responses)
    monkeypatch.setattr(interactive, "Prompt", MagicMock(ask=mock_session))
    
    # Mock prompts.prompts_list
    mock_prompts_list = AsyncMock()
    monkeypatch.setattr(interactive.prompts, "prompts_list", mock_prompts_list)
    
    # Run interactive mode
    await interactive.interactive_mode(mock_stream_manager)
    
    # Check that prompts_list was called with the stream_manager
    mock_prompts_list.assert_called_once()
    assert mock_prompts_list.call_args[1]["stream_manager"] == mock_stream_manager

@pytest.mark.asyncio
async def test_interactive_mode_resources_command(mock_stream_manager, monkeypatch):
    """Test that the /resources command works correctly."""
    # Set up mock to return /resources and then exit
    responses = ["/resources", "exit"]
    mock_session = MagicMock(side_effect=responses)
    monkeypatch.setattr(interactive, "Prompt", MagicMock(ask=mock_session))
    
    # Mock resources.resources_list
    mock_resources_list = AsyncMock()
    monkeypatch.setattr(interactive.resources, "resources_list", mock_resources_list)
    
    # Run interactive mode
    await interactive.interactive_mode(mock_stream_manager)
    
    # Check that resources_list was called with the stream_manager
    mock_resources_list.assert_called_once()
    assert mock_resources_list.call_args[1]["stream_manager"] == mock_stream_manager

@pytest.mark.asyncio
async def test_interactive_mode_servers_command(mock_stream_manager, monkeypatch):
    """Test that the /servers command works correctly."""
    # Set up mock to return /servers and then exit
    responses = ["/servers", "exit"]
    mock_session = MagicMock(side_effect=responses)
    monkeypatch.setattr(interactive, "Prompt", MagicMock(ask=mock_session))
    
    # Mock display_servers_info to track calls
    mock_display_servers = MagicMock()
    monkeypatch.setattr(interactive, "display_servers_info", mock_display_servers)
    
    # Run interactive mode
    await interactive.interactive_mode(mock_stream_manager)
    
    # Check that display_servers_info was called with context containing stream_manager
    mock_display_servers.assert_called_once()
    context = mock_display_servers.call_args[0][0]
    assert context["stream_manager"] == mock_stream_manager

@pytest.mark.asyncio
async def test_interactive_mode_servers_alias(mock_stream_manager, monkeypatch):
    """Test that the /s alias for /servers works correctly."""
    # Set up mock to return /s and then exit
    responses = ["/s", "exit"]
    mock_session = MagicMock(side_effect=responses)
    monkeypatch.setattr(interactive, "Prompt", MagicMock(ask=mock_session))
    
    # Mock display_servers_info to track calls
    mock_display_servers = MagicMock()
    monkeypatch.setattr(interactive, "display_servers_info", mock_display_servers)
    
    # Run interactive mode
    await interactive.interactive_mode(mock_stream_manager)
    
    # Check that display_servers_info was called
    mock_display_servers.assert_called_once()

@pytest.mark.asyncio
async def test_interactive_mode_chat_command(mock_stream_manager, monkeypatch):
    """Test that the /chat command works correctly."""
    # Set up mock to return /chat and then exit
    responses = ["/chat", "exit"]
    mock_session = MagicMock(side_effect=responses)
    monkeypatch.setattr(interactive, "Prompt", MagicMock(ask=mock_session))
    
    # Mock chat.chat_run
    mock_chat_run = AsyncMock()
    monkeypatch.setattr(interactive.chat, "chat_run", mock_chat_run)
    
    # Run interactive mode
    await interactive.interactive_mode(mock_stream_manager)
    
    # Check that chat_run was called with the stream_manager
    mock_chat_run.assert_called_once()
    assert mock_chat_run.call_args[1]["stream_manager"] == mock_stream_manager

@pytest.mark.asyncio
async def test_interactive_mode_tools_command(mock_stream_manager, monkeypatch):
    """Test that the /tools command works correctly."""
    # Set up mock to return /tools and then exit
    responses = ["/tools", "exit"]
    mock_session = MagicMock(side_effect=responses)
    monkeypatch.setattr(interactive, "Prompt", MagicMock(ask=mock_session))
    
    # Mock tools_command
    mock_tools_command = AsyncMock()
    monkeypatch.setattr(interactive, "tools_command", mock_tools_command)
    
    # Run interactive mode
    await interactive.interactive_mode(mock_stream_manager)
    
    # Check that tools_command was called with empty args and context
    mock_tools_command.assert_called_once()
    args, context = mock_tools_command.call_args[0]
    assert args == []
    assert context["stream_manager"] == mock_stream_manager

@pytest.mark.asyncio
async def test_interactive_mode_tools_all_command(mock_stream_manager, monkeypatch):
    """Test that the /tools-all command works correctly."""
    # Set up mock to return /tools-all and then exit
    responses = ["/tools-all", "exit"]
    mock_session = MagicMock(side_effect=responses)
    monkeypatch.setattr(interactive, "Prompt", MagicMock(ask=mock_session))
    
    # Mock tools_command
    mock_tools_command = AsyncMock()
    monkeypatch.setattr(interactive, "tools_command", mock_tools_command)
    
    # Run interactive mode
    await interactive.interactive_mode(mock_stream_manager)
    
    # Check that tools_command was called with --all flag
    mock_tools_command.assert_called_once()
    args, context = mock_tools_command.call_args[0]
    assert args == ["--all"]

@pytest.mark.asyncio
async def test_interactive_mode_tools_raw_command(mock_stream_manager, monkeypatch):
    """Test that the /tools-raw command works correctly."""
    # Set up mock to return /tools-raw and then exit
    responses = ["/tools-raw", "exit"]
    mock_session = MagicMock(side_effect=responses)
    monkeypatch.setattr(interactive, "Prompt", MagicMock(ask=mock_session))
    
    # Mock tools_command
    mock_tools_command = AsyncMock()
    monkeypatch.setattr(interactive, "tools_command", mock_tools_command)
    
    # Run interactive mode
    await interactive.interactive_mode(mock_stream_manager)
    
    # Check that tools_command was called with --raw flag
    mock_tools_command.assert_called_once()
    args, context = mock_tools_command.call_args[0]
    assert args == ["--raw"]

@pytest.mark.asyncio
async def test_interactive_mode_unknown_command(mock_stream_manager, monkeypatch, capsys):
    """Test that unknown commands are handled correctly."""
    # Set up mock to return an unknown command and then exit
    responses = ["/unknown", "exit"]
    mock_session = MagicMock(side_effect=responses)
    monkeypatch.setattr(interactive, "Prompt", MagicMock(ask=mock_session))
    
    # Run interactive mode
    await interactive.interactive_mode(mock_stream_manager)
    
    # Check output
    captured = capsys.readouterr()
    assert "Unknown command: /unknown" in captured.out
    assert "Type '/help' for available commands" in captured.out

@pytest.mark.asyncio
async def test_interactive_mode_error_handling(mock_stream_manager, monkeypatch, capsys):
    """Test that errors in command handlers are caught and reported."""
    # Set up mock to return /ping (which will raise an error) and then exit
    responses = ["/ping", "exit"]
    mock_session = MagicMock(side_effect=responses)
    monkeypatch.setattr(interactive, "Prompt", MagicMock(ask=mock_session))
    
    # Mock ping.ping_run to raise an exception
    mock_ping_run = AsyncMock(side_effect=Exception("Test error"))
    monkeypatch.setattr(interactive.ping, "ping_run", mock_ping_run)
    
    # Run interactive mode
    await interactive.interactive_mode(mock_stream_manager)
    
    # Check output
    captured = capsys.readouterr()
    assert "Error:" in captured.out
    assert "Test error" in captured.out

@pytest.mark.asyncio
async def test_interactive_mode_keyboard_interrupt(mock_stream_manager, monkeypatch, capsys):
    """Test that keyboard interrupts are handled gracefully."""
    # Set up mock to raise KeyboardInterrupt and then return exit
    mock_session = MagicMock(side_effect=[KeyboardInterrupt, "exit"])
    monkeypatch.setattr(interactive, "Prompt", MagicMock(ask=mock_session))
    
    # Run interactive mode
    await interactive.interactive_mode(mock_stream_manager)
    
    # Check output
    captured = capsys.readouterr()
    assert "Command interrupted" in captured.out
    assert "Type '/exit' to quit" in captured.out

@pytest.mark.asyncio
async def test_interactive_mode_empty_input(mock_stream_manager, monkeypatch):
    """Test that empty input is handled correctly."""
    # Set up mock to return empty string, then exit
    responses = ["", "exit"]
    mock_session = MagicMock(side_effect=responses)
    monkeypatch.setattr(interactive, "Prompt", MagicMock(ask=mock_session))
    
    # Run interactive mode
    result = await interactive.interactive_mode(mock_stream_manager)
    
    # Check that it returns True (clean exit)
    assert result is True

@pytest.mark.asyncio
async def test_run_interactive_command():
    """Test the Typer command for interactive mode."""
    # Call the function
    result = interactive.run_interactive()
    
    # It should return 0 (success)
    assert result == 0