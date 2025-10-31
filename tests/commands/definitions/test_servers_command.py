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

            # Verify the action was called with ServerActionParams
            mock_action.assert_called_once()
            call_args = mock_action.call_args[0][0]
            assert call_args.args == []
            assert not call_args.detailed
            assert not call_args.show_capabilities
            assert not call_args.show_transport
            assert call_args.output_format == "table"
            assert not call_args.ping_servers

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
            mock_action.assert_called_once()
            call_args = mock_action.call_args[0][0]
            assert call_args.detailed
            # Note: show_capabilities is controlled by 'raw' not 'detailed'

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_with_format(self, command):
        """Test execution with different output formats."""
        with patch(
            "mcp_cli.commands.actions.servers.servers_action_async"
        ) as mock_action:
            mock_action.return_value = {"servers": []}

            # Test with json format
            result = await command.execute(format="json")

            mock_action.assert_called_once()
            call_args = mock_action.call_args[0][0]
            assert call_args.output_format == "json"

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
