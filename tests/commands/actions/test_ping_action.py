"""Tests for the ping action command."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_cli.commands.actions.ping import (
    ping_action_async,
    _ping_server,
    _get_server_name,
    _matches_target,
    _display_results,
    ping_action,
)


@pytest.fixture
def mock_tool_manager():
    """Create a mock tool manager."""
    tm = MagicMock()
    # Create mock streams
    read_stream1 = MagicMock()
    write_stream1 = MagicMock()
    read_stream2 = MagicMock()
    write_stream2 = MagicMock()

    tm.get_streams.return_value = [
        (read_stream1, write_stream1),
        (read_stream2, write_stream2),
    ]

    # Mock server info
    server_info1 = MagicMock()
    server_info1.name = "server1"
    server_info2 = MagicMock()
    server_info2.name = "server2"

    tm.get_server_info = AsyncMock(return_value=[server_info1, server_info2])

    return tm


@pytest.mark.asyncio
async def test_ping_action_async_all_servers(mock_tool_manager):
    """Test pinging all servers."""
    with patch(
        "mcp_cli.commands.actions.ping._ping_server", new_callable=AsyncMock
    ) as mock_ping:
        mock_ping.return_value = ("server1", True, 10.5)

        with patch("mcp_cli.commands.actions.ping._display_results") as mock_display:
            result = await ping_action_async(mock_tool_manager)

            assert result is True
            assert mock_ping.call_count == 2
            mock_display.assert_called_once()


@pytest.mark.asyncio
async def test_ping_action_async_with_targets(mock_tool_manager):
    """Test pinging specific target servers."""
    with patch(
        "mcp_cli.commands.actions.ping._ping_server", new_callable=AsyncMock
    ) as mock_ping:
        mock_ping.return_value = ("server1", True, 10.5)

        with patch("mcp_cli.commands.actions.ping._display_results") as mock_display:
            result = await ping_action_async(mock_tool_manager, targets=["server1"])

            assert result is True
            assert mock_ping.call_count == 1
            mock_display.assert_called_once()


@pytest.mark.asyncio
async def test_ping_action_async_no_matching_servers(mock_tool_manager):
    """Test when no servers match the target."""
    with patch("mcp_cli.commands.actions.ping.output") as mock_output:
        result = await ping_action_async(mock_tool_manager, targets=["nonexistent"])

        assert result is False
        mock_output.error.assert_called_with("No matching servers found")
        mock_output.hint.assert_called_with(
            "Use 'servers' command to list available servers"
        )


@pytest.mark.asyncio
async def test_ping_action_async_with_server_names(mock_tool_manager):
    """Test pinging with explicit server names."""
    server_names = {0: "custom-name-1", 1: "custom-name-2"}

    with patch(
        "mcp_cli.commands.actions.ping._ping_server", new_callable=AsyncMock
    ) as mock_ping:
        mock_ping.side_effect = [
            ("custom-name-1", True, 10.5),
            ("custom-name-2", False, 100.0),
        ]

        with patch("mcp_cli.commands.actions.ping._display_results"):
            result = await ping_action_async(
                mock_tool_manager, server_names=server_names
            )

            assert result is True
            assert mock_ping.call_count == 2
            # Verify custom names were used
            first_call = mock_ping.call_args_list[0]
            assert first_call[0][1] == "custom-name-1"


@pytest.mark.asyncio
async def test_ping_server_success():
    """Test successful server ping."""
    read_stream = MagicMock()
    write_stream = MagicMock()

    with patch(
        "mcp_cli.commands.actions.ping.send_ping", new_callable=AsyncMock
    ) as mock_send:
        mock_send.return_value = True

        name, success, latency = await _ping_server(
            0, "test-server", read_stream, write_stream
        )

        assert name == "test-server"
        assert success is True
        assert latency > 0


@pytest.mark.asyncio
async def test_ping_server_timeout():
    """Test server ping timeout."""
    read_stream = MagicMock()
    write_stream = MagicMock()

    with patch(
        "mcp_cli.commands.actions.ping.send_ping", new_callable=AsyncMock
    ) as mock_send:
        mock_send.side_effect = asyncio.TimeoutError()

        name, success, latency = await _ping_server(
            0, "test-server", read_stream, write_stream, timeout=0.1
        )

        assert name == "test-server"
        assert success is False
        assert latency > 0


@pytest.mark.asyncio
async def test_ping_server_exception():
    """Test server ping with exception."""
    read_stream = MagicMock()
    write_stream = MagicMock()

    with patch(
        "mcp_cli.commands.actions.ping.send_ping", new_callable=AsyncMock
    ) as mock_send:
        mock_send.side_effect = Exception("Connection error")

        name, success, latency = await _ping_server(
            0, "test-server", read_stream, write_stream
        )

        assert name == "test-server"
        assert success is False
        assert latency > 0


def test_get_server_name_explicit():
    """Test getting server name with explicit names."""
    explicit_names = {0: "custom-name", 1: "another-name"}
    server_infos = []

    name = _get_server_name(0, explicit_names, server_infos)
    assert name == "custom-name"


def test_get_server_name_from_info():
    """Test getting server name from server info."""
    server_info = MagicMock()
    server_info.name = "info-name"
    server_infos = [server_info]

    name = _get_server_name(0, None, server_infos)
    assert name == "info-name"


def test_get_server_name_fallback():
    """Test getting server name with fallback."""
    name = _get_server_name(5, None, [])
    assert name == "server-5"


def test_matches_target_by_index():
    """Test matching target by server index."""
    assert _matches_target(0, "server-name", ["0"]) is True
    assert _matches_target(1, "server-name", ["0"]) is False


def test_matches_target_by_name():
    """Test matching target by server name."""
    assert _matches_target(0, "test-server", ["test-server"]) is True
    assert _matches_target(0, "test-server", ["TEST-SERVER"]) is True
    assert _matches_target(0, "test-server", ["other-server"]) is False


def test_matches_target_multiple():
    """Test matching with multiple targets."""
    assert _matches_target(0, "test-server", ["0", "test-server"]) is True
    assert _matches_target(1, "other-server", ["test-server", "other-server"]) is True
    assert _matches_target(2, "third-server", ["test-server", "other-server"]) is False


def test_display_results_all_online():
    """Test displaying results with all servers online."""
    results = [
        ("server1", True, 10.5),
        ("server2", True, 20.3),
        ("server3", True, 15.7),
    ]

    with patch("mcp_cli.commands.actions.ping.output") as mock_output:
        with patch("mcp_cli.commands.actions.ping.format_table") as mock_format_table:
            mock_format_table.return_value = "formatted_table"

            _display_results(results)

            mock_format_table.assert_called_once()
            mock_output.print_table.assert_called_with("formatted_table")
            mock_output.success.assert_called_with("3/3 servers online")
            mock_output.info.assert_called()  # Average latency

            # Check table data
            table_data = mock_format_table.call_args[0][0]
            assert len(table_data) == 3
            assert all(row["Status"] == "✓ Online" for row in table_data)


def test_display_results_mixed():
    """Test displaying results with mixed online/offline servers."""
    results = [
        ("server1", True, 10.5),
        ("server2", False, 100.0),
        ("server3", True, 15.7),
    ]

    with patch("mcp_cli.commands.actions.ping.output") as mock_output:
        with patch("mcp_cli.commands.actions.ping.format_table") as mock_format_table:
            mock_format_table.return_value = "formatted_table"

            _display_results(results)

            mock_output.success.assert_called_with("2/3 servers online")

            # Check table data
            table_data = mock_format_table.call_args[0][0]
            assert table_data[1]["Status"] == "✗ Offline"
            assert table_data[1]["Latency"] == "-"


def test_display_results_all_offline():
    """Test displaying results with all servers offline."""
    results = [("server1", False, 100.0), ("server2", False, 100.0)]

    with patch("mcp_cli.commands.actions.ping.output") as mock_output:
        with patch("mcp_cli.commands.actions.ping.format_table") as mock_format_table:
            mock_format_table.return_value = "formatted_table"

            _display_results(results)

            mock_output.error.assert_called_with("All servers are offline")
            # Should not call success or info
            mock_output.success.assert_not_called()
            mock_output.info.assert_not_called()


def test_display_results_sorted():
    """Test that results are sorted by server name."""
    results = [("zebra", True, 10.0), ("alpha", True, 20.0), ("beta", True, 15.0)]

    with patch("mcp_cli.commands.actions.ping.output"):
        with patch("mcp_cli.commands.actions.ping.format_table") as mock_format_table:
            mock_format_table.return_value = "formatted_table"

            _display_results(results)

            # Check that table data is sorted
            table_data = mock_format_table.call_args[0][0]
            server_names = [row["Server"] for row in table_data]
            assert server_names == ["alpha", "beta", "zebra"]


def test_ping_action_sync():
    """Test synchronous wrapper for ping action."""
    tm = MagicMock()
    server_names = {0: "test"}
    targets = ["test"]

    with patch("mcp_cli.commands.actions.ping.run_blocking") as mock_run:
        with patch(
            "mcp_cli.commands.actions.ping.ping_action_async", new_callable=AsyncMock
        ) as mock_async:
            ping_action(tm, server_names=server_names, targets=targets)

            mock_async.assert_called_with(
                tm, server_names=server_names, targets=targets
            )
            mock_run.assert_called_once()
