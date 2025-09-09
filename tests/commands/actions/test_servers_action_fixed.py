"""Tests for servers action - fixed version."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from mcp_cli.commands.actions.servers import (
    servers_action_async,
    servers_action,
    server_details_async,
    _get_server_icon,
    _format_performance,
    _format_capabilities,
)


def test_get_server_icon():
    """Test _get_server_icon helper function."""
    # Full-featured server
    capabilities = {"resources": True, "prompts": True}
    assert _get_server_icon(capabilities, 5) == "🎯"

    # Resource-capable server
    capabilities = {"resources": True}
    assert _get_server_icon(capabilities, 5) == "📁"

    # Prompt-capable server
    capabilities = {"prompts": True}
    assert _get_server_icon(capabilities, 5) == "💬"

    # Tool-heavy server (>15 tools)
    capabilities = {}
    assert _get_server_icon(capabilities, 20) == "🔧"

    # Basic tool server (1-15 tools)
    capabilities = {}
    assert _get_server_icon(capabilities, 5) == "⚙️"

    # Minimal server (no tools)
    capabilities = {}
    assert _get_server_icon(capabilities, 0) == "📦"


def test_format_performance():
    """Test _format_performance helper function."""
    # Unknown performance
    icon, text = _format_performance(None)
    assert icon == "❓"
    assert text == "Unknown"

    # Very fast (<10ms)
    icon, text = _format_performance(5.2)
    assert icon == "🚀"
    assert text == "5.2ms"

    # Fast (10-50ms)
    icon, text = _format_performance(25.7)
    assert icon == "✅"
    assert text == "25.7ms"

    # Moderate (50-100ms)
    icon, text = _format_performance(75.3)
    assert icon == "⚠️"
    assert text == "75.3ms"

    # Slow (>100ms)
    icon, text = _format_performance(150.8)
    assert icon == "🔴"
    assert text == "150.8ms"


def test_format_capabilities():
    """Test _format_capabilities helper function."""
    # No capabilities
    capabilities = {}
    assert _format_capabilities(capabilities) == "None"

    # Standard capabilities
    capabilities = {"tools": True, "prompts": True, "resources": True}
    result = _format_capabilities(capabilities)
    assert "Tools" in result
    assert "Prompts" in result
    assert "Resources" in result

    # Experimental capabilities
    capabilities = {"experimental": {"events": True, "streaming": True}}
    result = _format_capabilities(capabilities)
    assert "Events*" in result
    assert "Streaming*" in result


@pytest.fixture
def mock_context():
    """Create a mock context with tool manager."""
    context = MagicMock()
    tool_manager = MagicMock()
    context.tool_manager = tool_manager
    return context, tool_manager


@pytest.fixture
def sample_server_info():
    """Create sample server info data."""
    server1 = MagicMock()
    server1.name = "test_server"
    server1.transport = "stdio"
    server1.capabilities = {"tools": True}
    server1.tool_count = 5
    server1.display_status = "connected"

    server2 = MagicMock()
    server2.name = "another_server"
    server2.transport = "http"
    server2.capabilities = {"prompts": True}
    server2.tool_count = 0
    server2.display_status = "connecting"

    return [server1, server2]


@pytest.mark.asyncio
async def test_servers_action_async_basic(mock_context, sample_server_info):
    """Test basic servers action async."""
    context, tool_manager = mock_context
    tool_manager.get_server_info = AsyncMock(return_value=sample_server_info)

    with (
        patch("mcp_cli.commands.actions.servers.get_context", return_value=context),
        patch("mcp_cli.commands.actions.servers.output") as mock_output,
        patch("mcp_cli.commands.actions.servers.format_table") as mock_format_table,
    ):
        mock_table = MagicMock()
        mock_format_table.return_value = mock_table

        result = await servers_action_async()

        # Verify table creation and output
        mock_format_table.assert_called_once()
        mock_output.print_table.assert_called_once_with(mock_table)
        mock_output.rule.assert_called_once()

        # Verify result structure
        assert len(result) == 2
        assert result[0]["name"] == "test_server"
        assert result[1]["name"] == "another_server"


@pytest.mark.asyncio
async def test_servers_action_async_no_tool_manager():
    """Test servers action when no tool manager available."""
    context = MagicMock()
    context.tool_manager = None

    with (
        patch("mcp_cli.commands.actions.servers.get_context", return_value=context),
        patch("mcp_cli.commands.actions.servers.output") as mock_output,
    ):
        result = await servers_action_async()

        mock_output.error.assert_called_once_with("No tool manager available")
        assert result == []


@pytest.mark.asyncio
async def test_servers_action_async_no_servers(mock_context):
    """Test servers action when no servers available."""
    context, tool_manager = mock_context
    tool_manager.get_server_info = AsyncMock(return_value=[])

    with (
        patch("mcp_cli.commands.actions.servers.get_context", return_value=context),
        patch("mcp_cli.commands.actions.servers.output") as mock_output,
    ):
        result = await servers_action_async()

        mock_output.info.assert_called_once_with("No servers connected.")
        assert result == []


@pytest.mark.asyncio
async def test_servers_action_async_exception(mock_context):
    """Test servers action when exception occurs."""
    context, tool_manager = mock_context
    tool_manager.get_server_info = AsyncMock(
        side_effect=Exception("Server info failed")
    )

    with (
        patch("mcp_cli.commands.actions.servers.get_context", return_value=context),
        patch("mcp_cli.commands.actions.servers.output") as mock_output,
    ):
        result = await servers_action_async()

        mock_output.error.assert_called_once_with(
            "Failed to get server info: Server info failed"
        )
        assert result == []


@pytest.mark.asyncio
async def test_servers_action_async_json_output(mock_context, sample_server_info):
    """Test servers action with JSON output."""
    context, tool_manager = mock_context
    tool_manager.get_server_info = AsyncMock(return_value=sample_server_info)

    with (
        patch("mcp_cli.commands.actions.servers.get_context", return_value=context),
        patch("mcp_cli.commands.actions.servers.output") as mock_output,
    ):
        result = await servers_action_async(output_format="json")

        # Should call print with JSON instead of table
        mock_output.print.assert_called_once()

        assert len(result) == 2


@pytest.mark.asyncio
async def test_servers_action_async_with_ping(mock_context, sample_server_info):
    """Test servers action with ping enabled."""
    context, tool_manager = mock_context
    tool_manager.get_server_info = AsyncMock(return_value=sample_server_info)
    tool_manager.ping_server = AsyncMock()

    with (
        patch("mcp_cli.commands.actions.servers.get_context", return_value=context),
        patch("mcp_cli.commands.actions.servers.output"),
        patch("mcp_cli.commands.actions.servers.format_table") as mock_format_table,
    ):
        mock_format_table.return_value = MagicMock()

        result = await servers_action_async(ping_servers=True)

        # Verify ping was attempted
        assert tool_manager.ping_server.call_count == 2  # For both servers

        # Verify Ping column is included
        columns = mock_format_table.call_args[1]["columns"]
        assert "Ping" in columns

        assert len(result) == 2


@pytest.mark.asyncio
async def test_server_details_async(mock_context, sample_server_info):
    """Test server_details_async function."""
    context, tool_manager = mock_context
    tool_manager.get_server_info = AsyncMock(return_value=sample_server_info)
    tool_manager.ping_server = AsyncMock()
    tool_manager.get_tools_for_server = AsyncMock(return_value=[])

    with (
        patch("mcp_cli.commands.actions.servers.get_context", return_value=context),
        patch("mcp_cli.commands.actions.servers.output") as mock_output,
    ):
        await server_details_async("test_server")

        # Verify output was called for server details
        mock_output.rule.assert_called_once()
        assert mock_output.print.call_count >= 4  # Multiple print calls for details


@pytest.mark.asyncio
async def test_server_details_async_not_found(mock_context, sample_server_info):
    """Test server_details_async with non-existent server."""
    context, tool_manager = mock_context
    tool_manager.get_server_info = AsyncMock(return_value=sample_server_info)

    with (
        patch("mcp_cli.commands.actions.servers.get_context", return_value=context),
        patch("mcp_cli.commands.actions.servers.output") as mock_output,
    ):
        await server_details_async("nonexistent_server")

        mock_output.error.assert_called_once_with(
            "Server not found: nonexistent_server"
        )
        mock_output.hint.assert_called_once()


@pytest.mark.asyncio
async def test_server_details_async_no_tool_manager():
    """Test server_details_async when no tool manager available."""
    context = MagicMock()
    context.tool_manager = None

    with (
        patch("mcp_cli.commands.actions.servers.get_context", return_value=context),
        patch("mcp_cli.commands.actions.servers.output") as mock_output,
    ):
        await server_details_async("test_server")

        mock_output.error.assert_called_once_with("No tool manager available")


def test_servers_action_sync_wrapper():
    """Test the sync wrapper function."""
    with patch("mcp_cli.commands.actions.servers.run_blocking") as mock_run_blocking:
        mock_run_blocking.return_value = []

        result = servers_action(detailed=True)

        mock_run_blocking.assert_called_once()
        assert result == []


@pytest.mark.asyncio
async def test_servers_action_async_no_get_server_info_method(mock_context):
    """Test when tool manager doesn't have get_server_info method."""
    context, tool_manager = mock_context
    # Remove the get_server_info method
    del tool_manager.get_server_info

    with (
        patch("mcp_cli.commands.actions.servers.get_context", return_value=context),
        patch("mcp_cli.commands.actions.servers.output") as mock_output,
    ):
        result = await servers_action_async()

        mock_output.info.assert_called_once_with("No servers connected.")
        assert result == []


@pytest.mark.asyncio
async def test_server_details_async_with_tools(mock_context):
    """Test server_details_async with tools listed."""
    # Create server with some tools
    server = MagicMock()
    server.name = "test_server"
    server.transport = "stdio"
    server.capabilities = {"tools": True}
    server.tool_count = 3
    server.display_status = "connected"

    tool1 = MagicMock()
    tool1.name = "tool1"
    tool2 = MagicMock()
    tool2.name = "tool2"
    tools = [tool1, tool2]

    context, tool_manager = mock_context
    tool_manager.get_server_info = AsyncMock(return_value=[server])
    tool_manager.ping_server = AsyncMock()
    tool_manager.get_tools_for_server = AsyncMock(return_value=tools)

    with (
        patch("mcp_cli.commands.actions.servers.get_context", return_value=context),
        patch("mcp_cli.commands.actions.servers.output") as mock_output,
    ):
        await server_details_async("test_server")

        # Should show tools since count <= 10
        assert "Available tools:" in str(mock_output.print.call_args_list)
