"""Tests for the prompts command."""

import pytest
from unittest.mock import patch
from mcp_cli.commands.definitions.prompts import PromptsCommand


class TestPromptsCommand:
    """Test the PromptsCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a PromptsCommand instance."""
        return PromptsCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "prompts"
        assert command.aliases == []  # No aliases in implementation
        assert "prompts" in command.description.lower()

        # Check parameters
        params = {p.name for p in command.parameters}
        assert "server" in params
        assert "raw" in params
        assert "get" in params

    @pytest.mark.asyncio
    async def test_execute_list_all(self, command):
        """Test listing all prompts."""
        with patch(
            "mcp_cli.commands.actions.prompts.prompts_action_async"
        ) as mock_action:
            mock_action.return_value = {
                "prompts": [
                    {
                        "name": "summarize",
                        "server": "text-processor",
                        "description": "Summarize text",
                        "arguments": ["text", "max_length"],
                    },
                    {
                        "name": "translate",
                        "server": "translator",
                        "description": "Translate text",
                        "arguments": ["text", "target_language"],
                    },
                ]
            }

            result = await command.execute()

            mock_action.assert_called_once_with()

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_by_server(self, command):
        """Test listing prompts for a specific server."""
        with patch(
            "mcp_cli.commands.actions.prompts.prompts_action_async"
        ) as mock_action:
            mock_action.return_value = {
                "prompts": [
                    {
                        "name": "summarize",
                        "server": "text-processor",
                        "description": "Summarize text",
                    }
                ]
            }

            result = await command.execute(server=0)  # server parameter is an index

            mock_action.assert_called_once_with()

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_detailed(self, command):
        """Test listing prompts with detailed information."""
        with patch(
            "mcp_cli.commands.actions.prompts.prompts_action_async"
        ) as mock_action:
            mock_action.return_value = {
                "prompts": [
                    {
                        "name": "summarize",
                        "server": "text-processor",
                        "description": "Summarize text",
                        "arguments": [
                            {
                                "name": "text",
                                "type": "string",
                                "required": True,
                                "description": "Text to summarize",
                            },
                            {
                                "name": "max_length",
                                "type": "integer",
                                "required": False,
                                "default": 100,
                                "description": "Maximum summary length",
                            },
                        ],
                    }
                ]
            }

            result = await command.execute(raw=True)

            mock_action.assert_called_once_with()

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_no_prompts(self, command):
        """Test when no prompts are available."""
        with patch(
            "mcp_cli.commands.actions.prompts.prompts_action_async"
        ) as mock_action:
            mock_action.return_value = {"prompts": []}

            result = await command.execute()

            assert result.success is True
            # Should indicate no prompts available

    @pytest.mark.asyncio
    async def test_execute_error_handling(self, command):
        """Test error handling during execution."""
        with patch(
            "mcp_cli.commands.actions.prompts.prompts_action_async"
        ) as mock_action:
            mock_action.side_effect = Exception("Server error")

            result = await command.execute()

            assert result.success is False
            assert "Server error" in result.error or result.output
