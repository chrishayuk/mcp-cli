"""Tests for the servers command."""

import pytest
from unittest.mock import patch
from mcp_cli.commands.definitions.servers import ServersCommand


class TestServersCommand:
    """Test the ServersCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a ServersCommand instance."""
        return ServersCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "servers"
        assert command.aliases == []
        assert command.description == "List connected MCP servers and their status"
        assert "List connected MCP servers" in command.help_text

        # Check parameters
        params = {p.name for p in command.parameters}
        assert "detailed" in params
        assert "format" in params
        assert "ping" in params

    @pytest.mark.asyncio
    async def test_execute_basic(self, command):
        """Test basic execution without parameters."""
        with patch(
            "mcp_cli.commands.actions.servers.servers_action_async"
        ) as mock_action:
            # Mock the action to return server data
            mock_action.return_value = {
                "servers": [
                    {
                        "name": "test-server",
                        "status": "connected",
                        "tools": 5,
                        "resources": 2,
                    }
                ]
            }

            result = await command.execute()

            # Verify the action was called
            mock_action.assert_called_once_with(
                args=[],
                detailed=False,
                show_capabilities=False,
                show_transport=False,
                output_format="table",
                ping_servers=False,
            )

            # Check result
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_detailed(self, command):
        """Test execution with detailed flag."""
        with patch(
            "mcp_cli.commands.actions.servers.servers_action_async"
        ) as mock_action:
            mock_action.return_value = {
                "servers": [
                    {
                        "name": "test-server",
                        "status": "connected",
                        "tools": 5,
                        "resources": 2,
                        "capabilities": ["tools", "resources"],
                    }
                ]
            }

            result = await command.execute(detailed=True)

            # Verify the action was called with detailed=True
            mock_action.assert_called_once_with(
                args=[],
                detailed=True,
                show_capabilities=True,
                show_transport=True,
                output_format="table",
                ping_servers=False,
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_format(self, command):
        """Test execution with different output formats."""
        with patch(
            "mcp_cli.commands.actions.servers.servers_action_async"
        ) as mock_action:
            mock_action.return_value = {"servers": []}

            # Test with raw/json format
            result = await command.execute(raw=True)

            mock_action.assert_called_with(
                args=[],
                detailed=False,
                show_capabilities=False,
                show_transport=False,
                output_format="json",
                ping_servers=False,
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, command):
        """Test error handling during execution."""
        with patch(
            "mcp_cli.commands.actions.servers.servers_action_async"
        ) as mock_action:
            mock_action.side_effect = Exception("Connection failed")

            result = await command.execute()

            assert result.success is False
            assert "Connection failed" in result.error or result.output

    @pytest.mark.asyncio
    async def test_execute_no_servers(self, command):
        """Test execution when no servers are connected."""
        with patch(
            "mcp_cli.commands.actions.servers.servers_action_async"
        ) as mock_action:
            mock_action.return_value = {"servers": []}

            result = await command.execute()

            assert result.success is True
            # The result should indicate no servers are connected

    def test_parameter_validation(self, command):
        """Test parameter validation."""
        # Test with valid format
        error = command.validate_parameters(format="table")
        assert error is None

        error = command.validate_parameters(format="json")
        assert error is None

        # Test with invalid format
        error = command.validate_parameters(format="invalid")
        assert error is not None
        assert "Invalid choice" in error
