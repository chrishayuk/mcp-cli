# commands/test_servers_command.py

import pytest
from unittest.mock import patch, MagicMock
from typing import List

from mcp_cli.commands.servers import servers_action_async
from mcp_cli.tools.models import ServerInfo


class DummyToolManagerNoServers:
    """Mock tool manager with no servers."""

    async def get_server_info(self) -> List[ServerInfo]:
        return []


class DummyToolManagerWithServers:
    """Mock tool manager with servers."""

    def __init__(self, infos: List[ServerInfo]):
        self._infos = infos

    async def get_server_info(self) -> List[ServerInfo]:
        return self._infos


def make_info(id: int, name: str, tools: int, status: str) -> ServerInfo:
    """Helper to create ServerInfo objects."""
    return ServerInfo(id=id, name=name, tool_count=tools, status=status, namespace="ns")


@pytest.mark.asyncio
async def test_servers_action_no_servers():
    """Test servers_action when no servers are connected."""
    tm = DummyToolManagerNoServers()

    # Mock the output from chuk_term
    with patch("mcp_cli.commands.servers.output") as mock_output:
        mock_output.print = MagicMock()

        result = await servers_action_async(tm)

        # Should return empty list
        assert result == []

        # Should show message about no servers
        mock_output.print.assert_called()
        # Check that at least one call contains "No servers"
        calls = mock_output.print.call_args_list
        assert any("No servers" in str(call) for call in calls)


@pytest.mark.asyncio
async def test_servers_action_with_servers():
    """Test servers_action with connected servers."""
    infos = [
        make_info(0, "alpha", 3, "online"),
        make_info(1, "beta", 5, "offline"),
    ]
    tm = DummyToolManagerWithServers(infos)

    # Mock the output
    with patch("mcp_cli.commands.servers.output") as mock_output:
        mock_output.print = MagicMock()

        result = await servers_action_async(tm)

        # Should return enhanced server info (not the original)
        assert len(result) == 2

        # Should have printed something
        mock_output.print.assert_called()


@pytest.mark.asyncio
async def test_servers_action_with_detailed_flag():
    """Test servers_action with detailed output."""
    infos = [
        make_info(0, "server1", 10, "ready"),
    ]
    tm = DummyToolManagerWithServers(infos)

    with patch("mcp_cli.commands.servers.output") as mock_output:
        mock_output.print = MagicMock()

        # Call with detailed flag
        result = await servers_action_async(tm, detailed=True)

        # Should return server infos
        assert len(result) == 1

        # Should have printed output
        mock_output.print.assert_called()


@pytest.mark.asyncio
async def test_servers_action_with_capabilities():
    """Test servers_action with capabilities display."""
    # Create server without extra fields (they're added during enhancement)
    server_info = make_info(0, "test-server", 5, "ready")
    tm = DummyToolManagerWithServers([server_info])

    with patch("mcp_cli.commands.servers.output") as mock_output:
        mock_output.print = MagicMock()

        # Call with capabilities flag
        result = await servers_action_async(tm, show_capabilities=True)

        # Should return enhanced server info
        assert len(result) == 1
        assert result[0]["name"] == "test-server"

        # Should have capabilities in enhanced result
        assert "capabilities" in result[0]


@pytest.mark.asyncio
async def test_servers_action_with_transport_info():
    """Test servers_action with transport information."""
    # Create server without transport (it's added during enhancement)
    server_info = make_info(0, "http-server", 2, "connected")
    tm = DummyToolManagerWithServers([server_info])

    with patch("mcp_cli.commands.servers.output") as mock_output:
        mock_output.print = MagicMock()

        # Call with transport flag
        result = await servers_action_async(tm, show_transport=True)

        # Should return enhanced server info
        assert len(result) == 1

        # Enhanced result should have server_info with transport
        assert "server_info" in result[0]
        assert "transport" in result[0]["server_info"]


@pytest.mark.asyncio
async def test_servers_action_output_formats():
    """Test different output formats."""
    infos = [
        make_info(0, "server1", 3, "ready"),
    ]
    tm = DummyToolManagerWithServers(infos)

    # Test JSON format
    with patch("mcp_cli.commands.servers.output") as mock_output:
        mock_output.print = MagicMock()

        result = await servers_action_async(tm, output_format="json")

        # Should return server infos
        assert len(result) == 1

        # Check that JSON was printed (json.dumps is called internally)
        mock_output.print.assert_called()
        # Verify JSON-like output was printed
        calls = str(mock_output.print.call_args_list)
        # The output should contain JSON structure markers
        assert "{" in calls or "id" in calls or "name" in calls

    # Test tree format
    with patch("mcp_cli.commands.servers.output") as mock_output:
        mock_output.print = MagicMock()

        result = await servers_action_async(tm, output_format="tree")

        # Should return server infos
        assert len(result) == 1

        # Output should be used for output
        mock_output.print.assert_called()


@pytest.mark.asyncio
async def test_servers_action_error_handling():
    """Test error handling in servers_action."""

    class ErrorToolManager:
        async def get_server_info(self):
            raise Exception("Connection error")

    tm = ErrorToolManager()

    with patch("mcp_cli.commands.servers.output") as mock_output:
        mock_output.print = MagicMock()

        # Should handle the error gracefully
        result = await servers_action_async(tm)

        # Should return empty list on error
        assert result == []

        # Should print error message
        mock_output.print.assert_called()
        calls = mock_output.print.call_args_list
        assert any("Error" in str(call) for call in calls)


@pytest.mark.asyncio
async def test_servers_action_mixed_statuses():
    """Test servers with various statuses."""
    infos = [
        make_info(0, "server1", 10, "ready"),
        make_info(1, "server2", 0, "error"),
        make_info(2, "server3", 5, "connecting"),
        make_info(3, "server4", 15, "ready"),
    ]
    tm = DummyToolManagerWithServers(infos)

    with patch("mcp_cli.commands.servers.output") as mock_output:
        mock_output.print = MagicMock()

        result = await servers_action_async(tm)

        # Should return enhanced info for all servers
        assert len(result) == 4

        # Check that names are preserved
        server_names = [r["name"] for r in result]
        assert "server1" in server_names
        assert "server2" in server_names
        assert "server3" in server_names
        assert "server4" in server_names

        # Should have printed output
        mock_output.print.assert_called()


@pytest.mark.asyncio
async def test_servers_action_with_json_format():
    """Test that JSON format works correctly."""
    infos = [
        make_info(0, "server1", 3, "ready"),
    ]
    tm = DummyToolManagerWithServers(infos)

    with patch("mcp_cli.commands.servers.output") as mock_output:
        mock_output.print = MagicMock()

        # JSON format should work without issues
        result = await servers_action_async(tm, output_format="json")

        # Should return server infos
        assert len(result) == 1

        # Output should have printed something
        mock_output.print.assert_called()

        # The printed output should be valid JSON-like structure
        # (checking that the enhanced server data was formatted)
        assert result[0]["name"] == "server1"
        assert "server_info" in result[0]


@pytest.mark.asyncio
async def test_servers_action_display_error_handling():
    """Test that display errors are handled gracefully."""
    infos = [
        make_info(0, "server1", 3, "ready"),
    ]
    tm = DummyToolManagerWithServers(infos)

    with patch("mcp_cli.commands.servers.output") as mock_output:
        mock_output.print = MagicMock()

        # Make output.print raise an exception on first call, then work normally
        mock_output.print.side_effect = [Exception("Output error"), None, None, None]

        result = await servers_action_async(tm)

        # Should still return results despite display error
        assert len(result) == 1

        # Output print should have been called multiple times (retries)
        assert mock_output.print.call_count >= 1
