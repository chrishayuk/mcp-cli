"""Tests for the resources command."""

import pytest
from unittest.mock import patch
from mcp_cli.commands.definitions.resources import ResourcesCommand


class TestResourcesCommand:
    """Test the ResourcesCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a ResourcesCommand instance."""
        return ResourcesCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "resources"
        assert command.aliases == []  # No aliases in implementation
        assert "resources" in command.description.lower()

        # Check parameters
        params = {p.name for p in command.parameters}
        assert "server" in params
        assert "raw" in params
        assert "uri" in params

    @pytest.mark.asyncio
    async def test_execute_list_all(self, command):
        """Test listing all resources."""
        with patch(
            "mcp_cli.commands.actions.resources.resources_action_async"
        ) as mock_action:
            mock_action.return_value = {
                "resources": [
                    {
                        "name": "database.db",
                        "server": "sqlite",
                        "type": "database",
                        "description": "SQLite database",
                    },
                    {
                        "name": "config.json",
                        "server": "filesystem",
                        "type": "file",
                        "description": "Configuration file",
                    },
                ]
            }

            result = await command.execute()

            mock_action.assert_called_once_with()

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_by_server(self, command):
        """Test listing resources for a specific server."""
        with patch(
            "mcp_cli.commands.actions.resources.resources_action_async"
        ) as mock_action:
            mock_action.return_value = {
                "resources": [
                    {"name": "database.db", "server": "sqlite", "type": "database"}
                ]
            }

            result = await command.execute(server=0)  # server parameter is an index

            mock_action.assert_called_once_with()

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_detailed(self, command):
        """Test listing resources with detailed information."""
        with patch(
            "mcp_cli.commands.actions.resources.resources_action_async"
        ) as mock_action:
            mock_action.return_value = {
                "resources": [
                    {
                        "name": "database.db",
                        "server": "sqlite",
                        "type": "database",
                        "description": "SQLite database",
                        "metadata": {"size": "1024KB", "tables": 10},
                    }
                ]
            }

            result = await command.execute(raw=True)

            mock_action.assert_called_once_with()

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_no_resources(self, command):
        """Test when no resources are available."""
        with patch(
            "mcp_cli.commands.actions.resources.resources_action_async"
        ) as mock_action:
            mock_action.return_value = {"resources": []}

            result = await command.execute()

            assert result.success is True
            # Should indicate no resources available

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, command):
        """Test error handling during execution."""
        with patch(
            "mcp_cli.commands.actions.resources.resources_action_async"
        ) as mock_action:
            mock_action.side_effect = Exception("Server not connected")

            result = await command.execute()

            assert result.success is False
            assert "Server not connected" in result.error or result.output
