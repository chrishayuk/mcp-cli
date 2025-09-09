"""Comprehensive tests for servers action."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_cli.commands.actions.servers import (
    _get_server_icon,
    _format_capabilities,
    _format_performance,
    _get_server_status,
    _list_servers,
    _add_server,
    _remove_server,
    _enable_disable_server,
    _show_server_details,
    servers_action_async,
    servers_action,
)
from mcp_cli.config.config_manager import ServerConfig


class TestHelperFunctions:
    """Test helper functions."""

    def test_get_server_icon(self):
        """Test _get_server_icon function."""
        # Full-featured server (resources + prompts)
        capabilities = {"resources": True, "prompts": True, "tools": True}
        assert _get_server_icon(capabilities, 5) == "🎯"

        # Resources only
        capabilities = {"resources": True}
        assert _get_server_icon(capabilities, 0) == "📁"

        # Prompts only
        capabilities = {"prompts": True}
        assert _get_server_icon(capabilities, 0) == "💬"

        # Tool-heavy server (>15 tools)
        assert _get_server_icon({}, 20) == "🔧"

        # Basic tool server (1-15 tools)
        assert _get_server_icon({}, 5) == "⚙️"

        # Minimal server (no tools)
        assert _get_server_icon({}, 0) == "📦"

    def test_format_performance(self):
        """Test _format_performance function."""
        # Excellent performance (<10ms)
        icon, text = _format_performance(5.5)
        assert icon == "🚀"
        assert text == "5.5ms"

        # Good performance (<50ms)
        icon, text = _format_performance(25.0)
        assert icon == "✅"
        assert text == "25.0ms"

        # OK performance (<100ms)
        icon, text = _format_performance(75.0)
        assert icon == "⚠️"
        assert text == "75.0ms"

        # Poor performance (>=100ms)
        icon, text = _format_performance(150.0)
        assert icon == "🔴"
        assert text == "150.0ms"

        # Unknown performance
        icon, text = _format_performance(None)
        assert icon == "❓"
        assert text == "Unknown"

    def test_format_capabilities(self):
        """Test _format_capabilities function."""
        # Multiple capabilities
        capabilities = {"tools": True, "resources": True, "prompts": True}
        result = _format_capabilities(capabilities)
        assert "Tools" in result
        assert "Resources" in result
        assert "Prompts" in result

        # Single capability
        capabilities = {"tools": True}
        assert _format_capabilities(capabilities) == "Tools"

        # Experimental capabilities
        capabilities = {"experimental": {"events": True, "streaming": True}}
        result = _format_capabilities(capabilities)
        assert "Events*" in result
        assert "Streaming*" in result

        # No capabilities
        assert _format_capabilities({}) == "None"

    def test_get_server_status_with_dict(self):
        """Test _get_server_status with dictionary."""
        # Disabled server
        server_config = {"disabled": True}
        icon, text, reason = _get_server_status(server_config)
        assert icon == "⏸️"
        assert text == "Disabled"
        assert reason == "Server is disabled"

        # Connected server
        server_config = {"command": "test"}
        icon, text, reason = _get_server_status(server_config, connected=True)
        assert icon == "✅"
        assert text == "Connected"
        assert reason == "Server is active"

        # Not configured
        server_config = {}
        icon, text, reason = _get_server_status(server_config)
        assert icon == "❌"
        assert text == "Not Configured"
        assert reason == "No command or URL specified"

        # HTTP server
        server_config = {"url": "http://example.com", "transport": "http"}
        icon, text, reason = _get_server_status(server_config)
        assert icon == "🌐"
        assert text == "HTTP"
        assert reason == "URL: http://example.com"

        # STDIO server
        server_config = {"command": "test-command"}
        icon, text, reason = _get_server_status(server_config)
        assert icon == "📡"
        assert text == "STDIO"
        assert reason == "Command: test-command"

    def test_get_server_status_with_server_config(self):
        """Test _get_server_status with ServerConfig object."""
        # Disabled server
        server_config = ServerConfig(name="test", disabled=True)
        icon, text, reason = _get_server_status(server_config)
        assert icon == "⏸️"
        assert text == "Disabled"
        assert reason == "Server is disabled"

        # Connected server
        server_config = ServerConfig(name="test", command="test-cmd")
        icon, text, reason = _get_server_status(server_config, connected=True)
        assert icon == "✅"
        assert text == "Connected"
        assert reason == "Server is active"

        # HTTP server
        server_config = ServerConfig(name="test", url="http://example.com")
        icon, text, reason = _get_server_status(server_config)
        assert icon == "🌐"
        assert text == "HTTP"
        assert "URL:" in reason

        # STDIO server
        server_config = ServerConfig(name="test", command="test-command")
        icon, text, reason = _get_server_status(server_config)
        assert icon == "📡"
        assert text == "STDIO"
        assert "Command:" in reason


class TestListServers:
    """Test _list_servers function."""

    @pytest.mark.asyncio
    async def test_list_servers_basic(self):
        """Test basic server listing."""
        with patch("mcp_cli.commands.actions.servers.get_context") as mock_context:
            with patch(
                "mcp_cli.commands.actions.servers.ConfigManager"
            ) as mock_config_mgr:
                with patch(
                    "mcp_cli.commands.actions.servers.get_preference_manager"
                ) as mock_pref_mgr:
                    with patch(
                        "mcp_cli.commands.actions.servers.output"
                    ) as mock_output:
                        # Setup mocks
                        mock_ctx = MagicMock()
                        mock_ctx.tool_manager = None
                        mock_context.return_value = mock_ctx

                        # Mock config
                        mock_config = MagicMock()
                        test_server = ServerConfig(name="test", command="test-cmd")
                        mock_config.servers = {"test": test_server}
                        mock_config_mgr.return_value.get_config.return_value = (
                            mock_config
                        )

                        # Mock preferences
                        mock_prefs = MagicMock()
                        mock_prefs.get_runtime_servers.return_value = {}
                        mock_prefs.is_server_disabled.return_value = False
                        mock_pref_mgr.return_value = mock_prefs

                        # Call function
                        await _list_servers()

                        # Verify output was called
                        assert mock_output.rule.called
                        assert mock_output.print_table.called

    @pytest.mark.asyncio
    async def test_list_servers_with_runtime(self):
        """Test listing with runtime servers."""
        with patch("mcp_cli.commands.actions.servers.get_context") as mock_context:
            with patch(
                "mcp_cli.commands.actions.servers.ConfigManager"
            ) as mock_config_mgr:
                with patch(
                    "mcp_cli.commands.actions.servers.get_preference_manager"
                ) as mock_pref_mgr:
                    with patch("mcp_cli.commands.actions.servers.output"):
                        # Setup mocks
                        mock_ctx = MagicMock()
                        mock_ctx.tool_manager = None
                        mock_context.return_value = mock_ctx

                        # Mock config
                        mock_config = MagicMock()
                        mock_config.servers = {}
                        mock_config_mgr.return_value.get_config.return_value = (
                            mock_config
                        )

                        # Mock preferences with runtime server
                        mock_prefs = MagicMock()
                        mock_prefs.get_runtime_servers.return_value = {
                            "runtime-test": {
                                "command": "runtime-cmd",
                                "transport": "stdio",
                            }
                        }
                        mock_prefs.is_server_disabled.return_value = False
                        mock_pref_mgr.return_value = mock_prefs

                        # Call function
                        await _list_servers(show_all=True)

                        # Verify runtime servers were included
                        mock_prefs.get_runtime_servers.assert_called_once()


class TestAddServer:
    """Test _add_server function."""

    @pytest.mark.asyncio
    async def test_add_stdio_server(self):
        """Test adding a STDIO server."""
        with patch("mcp_cli.commands.actions.servers.ConfigManager") as mock_config_mgr:
            with patch(
                "mcp_cli.commands.actions.servers.get_preference_manager"
            ) as mock_pref_mgr:
                with patch("mcp_cli.commands.actions.servers.output") as mock_output:
                    # Mock config check
                    mock_config = MagicMock()
                    mock_config.servers = {}
                    mock_config_mgr.return_value.get_config.return_value = mock_config

                    # Mock preferences
                    mock_prefs = MagicMock()
                    mock_prefs.get_runtime_server.return_value = None
                    mock_pref_mgr.return_value = mock_prefs

                    # Call function
                    await _add_server(
                        name="test-server",
                        transport="stdio",
                        config_args=["test-command", "arg1", "arg2"],
                        env_vars={"TEST": "value"},
                    )

                    # Verify server was added
                    mock_prefs.add_runtime_server.assert_called_once()
                    call_args = mock_prefs.add_runtime_server.call_args[0]
                    assert call_args[0] == "test-server"
                    server_config = call_args[1]
                    assert server_config["command"] == "test-command"
                    assert server_config["args"] == ["arg1", "arg2"]
                    assert server_config["env"] == {"TEST": "value"}

                    # Verify success message
                    mock_output.success.assert_called()

    @pytest.mark.asyncio
    async def test_add_http_server(self):
        """Test adding an HTTP server."""
        with patch("mcp_cli.commands.actions.servers.ConfigManager") as mock_config_mgr:
            with patch(
                "mcp_cli.commands.actions.servers.get_preference_manager"
            ) as mock_pref_mgr:
                with patch("mcp_cli.commands.actions.servers.output"):
                    # Mock config check
                    mock_config = MagicMock()
                    mock_config.servers = {}
                    mock_config_mgr.return_value.get_config.return_value = mock_config

                    # Mock preferences
                    mock_prefs = MagicMock()
                    mock_prefs.get_runtime_server.return_value = None
                    mock_pref_mgr.return_value = mock_prefs

                    # Call function
                    await _add_server(
                        name="api-server",
                        transport="http",
                        config_args=["https://api.example.com"],
                        headers={"Authorization": "Bearer token"},
                    )

                    # Verify server was added
                    mock_prefs.add_runtime_server.assert_called_once()
                    call_args = mock_prefs.add_runtime_server.call_args[0]
                    assert call_args[0] == "api-server"
                    server_config = call_args[1]
                    assert server_config["url"] == "https://api.example.com"
                    assert server_config["transport"] == "http"
                    assert server_config["headers"] == {"Authorization": "Bearer token"}

    @pytest.mark.asyncio
    async def test_add_duplicate_server(self):
        """Test adding a duplicate server."""
        with patch(
            "mcp_cli.commands.actions.servers.get_preference_manager"
        ) as mock_pref_mgr:
            with patch("mcp_cli.commands.actions.servers.output") as mock_output:
                # Mock preferences - server already exists
                mock_prefs = MagicMock()
                mock_prefs.get_runtime_server.return_value = {"command": "existing"}
                mock_pref_mgr.return_value = mock_prefs

                # Call function
                await _add_server(
                    name="existing-server",
                    transport="stdio",
                    config_args=["test-command"],
                )

                # Verify error message
                mock_output.error.assert_called()
                assert "already exists" in str(mock_output.error.call_args)

                # Verify server was NOT added
                mock_prefs.add_runtime_server.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_sse_server(self):
        """Test adding an SSE server."""
        with patch("mcp_cli.commands.actions.servers.ConfigManager") as mock_config_mgr:
            with patch(
                "mcp_cli.commands.actions.servers.get_preference_manager"
            ) as mock_pref_mgr:
                with patch("mcp_cli.commands.actions.servers.output") as mock_output:
                    # Mock config check
                    mock_config = MagicMock()
                    mock_config.servers = {}
                    mock_config_mgr.return_value.get_config.return_value = mock_config

                    # Mock preferences
                    mock_prefs = MagicMock()
                    mock_prefs.get_runtime_server.return_value = None
                    mock_pref_mgr.return_value = mock_prefs

                    # Call function
                    await _add_server(
                        name="sse-server",
                        transport="sse",
                        config_args=["https://sse.example.com/events"],
                    )

                    # Verify server was added
                    mock_prefs.add_runtime_server.assert_called_once()
                    call_args = mock_prefs.add_runtime_server.call_args[0]
                    assert call_args[0] == "sse-server"
                    server_config = call_args[1]
                    assert server_config["url"] == "https://sse.example.com/events"
                    assert server_config["transport"] == "sse"
                    mock_output.success.assert_called()

    @pytest.mark.asyncio
    async def test_add_stdio_server_missing_command(self):
        """Test adding STDIO server without command."""
        with patch("mcp_cli.commands.actions.servers.output") as mock_output:
            await _add_server(
                name="test-server",
                transport="stdio",
                config_args=[],  # No command
            )

            # Should show error
            mock_output.error.assert_called_with("STDIO server requires a command")

    @pytest.mark.asyncio
    async def test_add_http_server_missing_url(self):
        """Test adding HTTP server without URL."""
        with patch("mcp_cli.commands.actions.servers.output") as mock_output:
            await _add_server(
                name="test-server",
                transport="http",
                config_args=[],  # No URL
            )

            # Should show error
            mock_output.error.assert_called_with("HTTP server requires a URL")

    @pytest.mark.asyncio
    async def test_add_invalid_transport(self):
        """Test adding server with invalid transport."""
        with patch("mcp_cli.commands.actions.servers.output") as mock_output:
            await _add_server(
                name="test-server", transport="invalid", config_args=["command"]
            )

            # Should show error
            mock_output.error.assert_called_with("Unknown transport type: invalid")


class TestRemoveServer:
    """Test _remove_server function."""

    @pytest.mark.asyncio
    async def test_remove_existing_server(self):
        """Test removing an existing server."""
        with patch(
            "mcp_cli.commands.actions.servers.get_preference_manager"
        ) as mock_pref_mgr:
            with patch("mcp_cli.commands.actions.servers.output") as mock_output:
                # Mock preferences
                mock_prefs = MagicMock()
                mock_prefs.get_runtime_server.return_value = {"command": "test"}
                mock_prefs.remove_runtime_server.return_value = True
                mock_pref_mgr.return_value = mock_prefs

                # Call function
                await _remove_server("test-server")

                # Verify server was removed
                mock_prefs.remove_runtime_server.assert_called_once_with("test-server")
                mock_output.success.assert_called()

    @pytest.mark.asyncio
    async def test_remove_nonexistent_server(self):
        """Test removing a non-existent server."""
        with patch(
            "mcp_cli.commands.actions.servers.get_preference_manager"
        ) as mock_pref_mgr:
            with patch(
                "mcp_cli.commands.actions.servers.ConfigManager"
            ) as mock_config_mgr:
                with patch("mcp_cli.commands.actions.servers.output") as mock_output:
                    # Mock preferences - server doesn't exist
                    mock_prefs = MagicMock()
                    mock_prefs.remove_runtime_server.return_value = (
                        False  # Server not found
                    )
                    mock_prefs.get_runtime_servers.return_value = {}
                    mock_pref_mgr.return_value = mock_prefs

                    # Mock config - server doesn't exist there either
                    mock_config = MagicMock()
                    mock_config.servers = {}
                    mock_config_mgr.return_value.get_config.return_value = mock_config

                    # Call function
                    await _remove_server("nonexistent")

                    # Verify error message
                    mock_output.error.assert_called()
                    assert "not found" in str(mock_output.error.call_args)

    @pytest.mark.asyncio
    async def test_remove_config_server(self):
        """Test removing a server from project config."""
        with patch("mcp_cli.commands.actions.servers.ConfigManager") as mock_config_mgr:
            with patch(
                "mcp_cli.commands.actions.servers.get_preference_manager"
            ) as mock_pref_mgr:
                with patch("mcp_cli.commands.actions.servers.output") as mock_output:
                    # Mock preferences - server not in runtime (returns False)
                    mock_prefs = MagicMock()
                    mock_prefs.remove_runtime_server.return_value = False
                    mock_pref_mgr.return_value = mock_prefs

                    # Mock config - server exists in config
                    mock_config = MagicMock()
                    test_server = ServerConfig(name="config-server", command="test")
                    mock_config.servers = {"config-server": test_server}
                    mock_config_mgr.return_value.get_config.return_value = mock_config

                    # Call function
                    await _remove_server("config-server")

                    # Verify warning message
                    mock_output.warning.assert_called()
                    assert "project configuration" in str(mock_output.warning.call_args)


class TestEnableDisableServer:
    """Test _enable_disable_server function."""

    @pytest.mark.asyncio
    async def test_enable_server(self):
        """Test enabling a server."""
        with patch(
            "mcp_cli.commands.actions.servers.get_preference_manager"
        ) as mock_pref_mgr:
            with patch(
                "mcp_cli.commands.actions.servers.ConfigManager"
            ) as mock_config_mgr:
                with patch("mcp_cli.commands.actions.servers.output") as mock_output:
                    # Mock preferences
                    mock_prefs = MagicMock()
                    mock_prefs.get_runtime_server.return_value = {"command": "test"}
                    mock_pref_mgr.return_value = mock_prefs

                    # Mock config
                    mock_config = MagicMock()
                    mock_config.get_server.return_value = None
                    mock_config_mgr.return_value.get_config.return_value = mock_config

                    # Call function to enable
                    await _enable_disable_server("test-server", enable=True)

                    # Verify server was enabled
                    mock_prefs.enable_server.assert_called_once_with("test-server")
                    mock_output.success.assert_called()

    @pytest.mark.asyncio
    async def test_disable_server(self):
        """Test disabling a server."""
        with patch(
            "mcp_cli.commands.actions.servers.get_preference_manager"
        ) as mock_pref_mgr:
            with patch(
                "mcp_cli.commands.actions.servers.ConfigManager"
            ) as mock_config_mgr:
                with patch("mcp_cli.commands.actions.servers.output") as mock_output:
                    # Mock preferences
                    mock_prefs = MagicMock()
                    mock_prefs.get_runtime_server.return_value = {"command": "test"}
                    mock_pref_mgr.return_value = mock_prefs

                    # Mock config
                    mock_config = MagicMock()
                    mock_config.get_server.return_value = None
                    mock_config_mgr.return_value.get_config.return_value = mock_config

                    # Call function to disable
                    await _enable_disable_server("test-server", enable=False)

                    # Verify server was disabled
                    mock_prefs.disable_server.assert_called_once_with("test-server")
                    mock_output.success.assert_called()


class TestShowServerDetails:
    """Test _show_server_details function."""

    @pytest.mark.asyncio
    async def test_show_server_details_project_server(self):
        """Test showing details for a project server."""
        with patch("mcp_cli.commands.actions.servers.ConfigManager") as mock_config_mgr:
            with patch(
                "mcp_cli.commands.actions.servers.get_preference_manager"
            ) as mock_pref_mgr:
                with patch("mcp_cli.commands.actions.servers.output") as mock_output:
                    # Mock config with server
                    test_server = ServerConfig(
                        name="test-server",
                        command="uvx",
                        args=["test-server"],
                        env={"DEBUG": "true"},
                        disabled=False,
                    )
                    mock_config = MagicMock()
                    mock_config.servers = {"test-server": test_server}
                    mock_config.get_server.return_value = (
                        test_server  # Mock get_server method
                    )
                    mock_config_mgr.return_value.get_config.return_value = mock_config

                    # Mock preferences - not found as runtime server
                    mock_prefs = MagicMock()
                    mock_prefs.get_runtime_server.return_value = None  # Not in runtime
                    mock_pref_mgr.return_value = mock_prefs

                    # Call function
                    await _show_server_details("test-server")

                    # Verify output
                    mock_output.rule.assert_called()
                    # Details are printed with output.print
                    assert mock_output.print.called

    @pytest.mark.asyncio
    async def test_show_server_details_runtime_server(self):
        """Test showing details for a runtime server."""
        with patch("mcp_cli.commands.actions.servers.ConfigManager") as mock_config_mgr:
            with patch(
                "mcp_cli.commands.actions.servers.get_preference_manager"
            ) as mock_pref_mgr:
                with patch("mcp_cli.commands.actions.servers.output") as mock_output:
                    # Mock config without server
                    mock_config = MagicMock()
                    mock_config.servers = {}
                    mock_config.get_server.return_value = None
                    mock_config_mgr.return_value.get_config.return_value = mock_config

                    # Mock preferences with runtime server
                    mock_prefs = MagicMock()
                    mock_prefs.get_runtime_server.return_value = {
                        "transport": "http",
                        "url": "http://api.example.com",
                        "headers": {"Auth": "Bearer token"},
                    }
                    mock_pref_mgr.return_value = mock_prefs

                    # Call function
                    await _show_server_details("runtime-server")

                    # Verify output
                    mock_output.rule.assert_called()
                    # Should print various info lines
                    assert mock_output.print.called

    @pytest.mark.asyncio
    async def test_show_server_details_not_found(self):
        """Test showing details for non-existent server."""
        with patch("mcp_cli.commands.actions.servers.ConfigManager") as mock_config_mgr:
            with patch(
                "mcp_cli.commands.actions.servers.get_preference_manager"
            ) as mock_pref_mgr:
                with patch(
                    "mcp_cli.commands.actions.servers.get_context"
                ) as mock_get_context:
                    with patch(
                        "mcp_cli.commands.actions.servers.output"
                    ) as mock_output:
                        # Mock empty config
                        mock_config = MagicMock()
                        mock_config.servers = {}
                        mock_config.get_server.return_value = None
                        mock_config_mgr.return_value.get_config.return_value = (
                            mock_config
                        )

                        # Mock empty preferences
                        mock_prefs = MagicMock()
                        mock_prefs.get_runtime_server.return_value = None
                        mock_pref_mgr.return_value = mock_prefs

                        # Mock context with no matching server
                        mock_context = MagicMock()
                        mock_tm = AsyncMock()
                        mock_tm.get_server_info.return_value = []  # No servers
                        mock_context.tool_manager = mock_tm
                        mock_get_context.return_value = mock_context

                        # Call function
                        await _show_server_details("nonexistent")

                        # Verify error
                        mock_output.error.assert_called_with(
                            "Server 'nonexistent' not found"
                        )


class TestServersActionAsync:
    """Test servers_action_async function."""

    @pytest.mark.asyncio
    async def test_no_args_no_tool_manager(self):
        """Test that no args with no tool manager shows error."""
        with patch("mcp_cli.commands.actions.servers.get_context") as mock_get_context:
            with patch("mcp_cli.commands.actions.servers.output") as mock_output:
                # Mock context with no tool manager
                mock_context = MagicMock()
                mock_context.tool_manager = None
                mock_get_context.return_value = mock_context

                result = await servers_action_async()

                # Should show error about no tool manager
                mock_output.error.assert_called_with("No tool manager available")
                assert result == []

    @pytest.mark.asyncio
    async def test_no_args_with_servers(self):
        """Test that no args lists connected servers."""
        with patch("mcp_cli.commands.actions.servers.get_context") as mock_get_context:
            with patch("mcp_cli.commands.actions.servers.output") as mock_output:
                # Mock context with tool manager
                mock_context = MagicMock()
                mock_tm = AsyncMock()

                # Mock server info
                mock_server = MagicMock()
                mock_server.name = "test-server"
                mock_server.transport = "stdio"
                mock_server.tool_count = 5
                mock_server.capabilities = {"tools": True}

                mock_tm.get_server_info.return_value = [mock_server]
                mock_context.tool_manager = mock_tm
                mock_get_context.return_value = mock_context

                result = await servers_action_async()

                # Should display server info
                assert mock_output.print_table.called
                assert len(result) == 1
                assert result[0]["name"] == "test-server"

    @pytest.mark.asyncio
    async def test_list_command(self):
        """Test list command."""
        with patch("mcp_cli.commands.actions.servers._list_servers") as mock_list:
            mock_list.return_value = None

            result = await servers_action_async(args=["list"])

            mock_list.assert_called_once_with(False)
            assert result == []

    @pytest.mark.asyncio
    async def test_list_all_command(self):
        """Test list all command."""
        with patch("mcp_cli.commands.actions.servers._list_servers") as mock_list:
            mock_list.return_value = None

            await servers_action_async(args=["list", "all"])

            mock_list.assert_called_once_with(True)

    @pytest.mark.asyncio
    async def test_add_stdio_command(self):
        """Test add STDIO server command."""
        with patch("mcp_cli.commands.actions.servers._add_server") as mock_add:
            mock_add.return_value = None

            await servers_action_async(args=["add", "test", "stdio", "command", "arg1"])

            mock_add.assert_called_once_with(
                "test", "stdio", ["command", "arg1"], {}, {}
            )

    @pytest.mark.asyncio
    async def test_add_http_with_options(self):
        """Test add HTTP server with options."""
        with patch("mcp_cli.commands.actions.servers._add_server") as mock_add:
            mock_add.return_value = None

            await servers_action_async(
                args=[
                    "add",
                    "api",
                    "--transport",
                    "http",
                    "--header",
                    "Auth: Bearer token",
                    "--env",
                    "KEY=value",
                    "--",
                    "https://api.example.com",
                ]
            )

            mock_add.assert_called_once()
            call_args = mock_add.call_args[0]  # Use positional args
            assert call_args[0] == "api"  # name
            assert call_args[1] == "http"  # transport
            assert call_args[2] == ["https://api.example.com"]  # config_args
            assert call_args[3] == {"KEY": "value"}  # env_vars
            assert call_args[4] == {"Auth": "Bearer token"}  # headers

    @pytest.mark.asyncio
    async def test_remove_command(self):
        """Test remove server command."""
        with patch("mcp_cli.commands.actions.servers._remove_server") as mock_remove:
            mock_remove.return_value = None

            await servers_action_async(args=["remove", "test-server"])

            mock_remove.assert_called_once_with("test-server")

    @pytest.mark.asyncio
    async def test_enable_command(self):
        """Test enable server command."""
        with patch(
            "mcp_cli.commands.actions.servers._enable_disable_server"
        ) as mock_enable:
            mock_enable.return_value = None

            await servers_action_async(args=["enable", "test-server"])

            mock_enable.assert_called_once_with("test-server", True)

    @pytest.mark.asyncio
    async def test_disable_command(self):
        """Test disable server command."""
        with patch(
            "mcp_cli.commands.actions.servers._enable_disable_server"
        ) as mock_disable:
            mock_disable.return_value = None

            await servers_action_async(args=["disable", "test-server"])

            mock_disable.assert_called_once_with("test-server", False)

    @pytest.mark.asyncio
    async def test_show_server_details(self):
        """Test showing server details."""
        with patch(
            "mcp_cli.commands.actions.servers._show_server_details"
        ) as mock_show:
            mock_show.return_value = None

            await servers_action_async(args=["test-server"])

            mock_show.assert_called_once_with("test-server")


class TestServersAction:
    """Test synchronous servers_action wrapper."""

    def test_servers_action_wrapper(self):
        """Test that servers_action properly wraps async function."""
        with patch("mcp_cli.commands.actions.servers.run_blocking") as mock_run:
            mock_run.return_value = [{"name": "test"}]

            result = servers_action(args=["list"])

            mock_run.assert_called_once()
            assert result == [{"name": "test"}]
