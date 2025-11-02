"""Tests for the token command."""

import pytest
from unittest.mock import patch, AsyncMock
from mcp_cli.commands.definitions.token import TokenCommand
from mcp_cli.commands.base import CommandMode


class TestTokenCommand:
    """Test the TokenCommand implementation."""

    @pytest.fixture
    def command(self):
        """Create a TokenCommand instance."""
        return TokenCommand()

    def test_command_properties(self, command):
        """Test command properties."""
        assert command.name == "token"
        assert command.aliases == ["tokens"]
        assert command.description == "Manage OAuth and authentication tokens"
        assert "Manage OAuth and authentication tokens" in command.help_text
        assert "set" in command.help_text  # Should include set command
        assert "get" in command.help_text  # Should include get command
        assert command.modes == (CommandMode.CHAT | CommandMode.INTERACTIVE)
        assert command.requires_context  # Needs context to get server list

    @pytest.mark.asyncio
    async def test_execute_no_args(self, command):
        """Test execution with no arguments - should list tokens."""
        with patch(
            "mcp_cli.commands.actions.token.token_list_action_async",
            new_callable=AsyncMock,
        ) as mock_list:
            result = await command.execute()

            # Should call list action
            mock_list.assert_called_once()
            call_args = mock_list.call_args[0][0]
            assert hasattr(call_args.__class__, "model_fields")  # It's a Pydantic model
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_list_subcommand(self, command):
        """Test execution with 'list' subcommand."""
        with patch(
            "mcp_cli.commands.actions.token.token_list_action_async",
            new_callable=AsyncMock,
        ) as mock_list:
            result = await command.execute(args=["list"])

            # Should call list action
            mock_list.assert_called_once()
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_clear_without_force(self, command):
        """Test execution with 'clear' subcommand without force flag."""
        with patch(
            "mcp_cli.commands.actions.token.token_clear_action_async",
            new_callable=AsyncMock,
        ) as mock_clear:
            result = await command.execute(args=["clear"])

            # Should call clear action with force=False
            mock_clear.assert_called_once()
            call_args = mock_clear.call_args[0][0]
            assert not call_args.force
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_clear_with_force_long(self, command):
        """Test execution with 'clear' subcommand with --force flag."""
        with patch(
            "mcp_cli.commands.actions.token.token_clear_action_async",
            new_callable=AsyncMock,
        ) as mock_clear:
            result = await command.execute(args=["clear", "--force"])

            # Should call clear action with force=True
            mock_clear.assert_called_once()
            call_args = mock_clear.call_args[0][0]
            assert call_args.force
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_clear_with_force_short(self, command):
        """Test execution with 'clear' subcommand with -f flag."""
        with patch(
            "mcp_cli.commands.actions.token.token_clear_action_async",
            new_callable=AsyncMock,
        ) as mock_clear:
            result = await command.execute(args=["clear", "-f"])

            # Should call clear action with force=True
            mock_clear.assert_called_once()
            call_args = mock_clear.call_args[0][0]
            assert call_args.force
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_delete_with_name(self, command):
        """Test execution with 'delete' subcommand and token name."""
        with patch(
            "mcp_cli.commands.actions.token.token_delete_action_async",
            new_callable=AsyncMock,
        ) as mock_delete:
            result = await command.execute(args=["delete", "github"])

            # Should call delete action with the token name
            mock_delete.assert_called_once()
            call_args = mock_delete.call_args[0][0]
            assert call_args.name == "github"
            assert call_args.oauth is True
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_delete_without_name(self, command):
        """Test execution with 'delete' subcommand but no token name."""
        with patch("chuk_term.ui.output") as mock_output:
            result = await command.execute(args=["delete"])

            # Should show error
            mock_output.error.assert_called_with(
                "Token name required for delete command"
            )
            mock_output.hint.assert_called()
            assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_unknown_subcommand(self, command):
        """Test execution with unknown subcommand."""
        with patch("chuk_term.ui.output") as mock_output:
            result = await command.execute(args=["unknown"])

            # Should show error
            mock_output.error.assert_called_with("Unknown token subcommand: unknown")
            assert mock_output.hint.call_count == 2  # Two hints
            assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_string_arg(self, command):
        """Test execution with string argument instead of list."""
        with patch(
            "mcp_cli.commands.actions.token.token_list_action_async",
            new_callable=AsyncMock,
        ) as mock_list:
            # Pass a string arg (should be converted to list)
            result = await command.execute(args="list")

            mock_list.assert_called_once()
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_empty_list(self, command):
        """Test execution with empty list of arguments."""
        with patch(
            "mcp_cli.commands.actions.token.token_list_action_async",
            new_callable=AsyncMock,
        ) as mock_list:
            result = await command.execute(args=[])

            # Should call list action (default)
            mock_list.assert_called_once()
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_case_insensitive_subcommand(self, command):
        """Test that subcommands are case-insensitive."""
        with patch(
            "mcp_cli.commands.actions.token.token_list_action_async",
            new_callable=AsyncMock,
        ) as mock_list:
            result = await command.execute(args=["LIST"])

            # Should call list action (subcommand converted to lowercase)
            mock_list.assert_called_once()
            assert result.success is True

    @pytest.mark.asyncio
    async def test_help_text_content(self, command):
        """Test that help text contains expected information."""
        help_text = command.help_text

        # Check for key usage patterns
        assert "/token" in help_text
        assert "/token list" in help_text
        assert "/token clear" in help_text
        assert "/token delete" in help_text
        assert "--force" in help_text

    def test_modes_flags(self, command):
        """Test that command modes are correctly set."""
        # Should be available in CHAT and INTERACTIVE but not CLI
        assert CommandMode.CHAT in command.modes
        assert CommandMode.INTERACTIVE in command.modes
        assert CommandMode.CLI not in command.modes

    @pytest.mark.asyncio
    async def test_execute_set_with_name_and_value(self, command):
        """Test execution with 'set' subcommand and token name/value."""
        with patch(
            "mcp_cli.commands.actions.token.token_set_action_async",
            new_callable=AsyncMock,
        ) as mock_set:
            result = await command.execute(args=["set", "my-api", "secret-token"])

            # Should call set action with the token name and value
            mock_set.assert_called_once()
            call_args = mock_set.call_args[0][0]
            assert call_args.name == "my-api"
            assert call_args.value == "secret-token"
            assert call_args.token_type == "bearer"
            assert call_args.namespace == "generic"
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_set_without_value(self, command):
        """Test execution with 'set' subcommand but no token value."""
        with patch("chuk_term.ui.output") as mock_output:
            result = await command.execute(args=["set", "my-api"])

            # Should show error
            mock_output.error.assert_called_with(
                "Token name and value required for set command"
            )
            mock_output.hint.assert_called()
            assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_set_without_name_or_value(self, command):
        """Test execution with 'set' subcommand but no arguments."""
        with patch("chuk_term.ui.output") as mock_output:
            result = await command.execute(args=["set"])

            # Should show error
            mock_output.error.assert_called_with(
                "Token name and value required for set command"
            )
            mock_output.hint.assert_called()
            assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_get_with_name(self, command):
        """Test execution with 'get' subcommand and token name."""
        from mcp_cli.constants import OAUTH_NAMESPACE, GENERIC_NAMESPACE

        with patch(
            "mcp_cli.commands.actions.token.token_get_action_async",
            new_callable=AsyncMock,
        ) as mock_get:
            result = await command.execute(args=["get", "notion"])

            # Should call get action twice (OAuth namespace then generic)
            assert mock_get.call_count == 2
            # First call should be for OAuth namespace (mcp-cli)
            first_call_name = mock_get.call_args_list[0][0][0]
            first_call_kwargs = mock_get.call_args_list[0][1]
            assert first_call_name == "notion"
            assert first_call_kwargs["namespace"] == OAUTH_NAMESPACE
            # Second call should be for generic namespace
            second_call_name = mock_get.call_args_list[1][0][0]
            second_call_kwargs = mock_get.call_args_list[1][1]
            assert second_call_name == "notion"
            assert second_call_kwargs["namespace"] == GENERIC_NAMESPACE
            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_get_without_name(self, command):
        """Test execution with 'get' subcommand but no token name."""
        with patch("chuk_term.ui.output") as mock_output:
            result = await command.execute(args=["get"])

            # Should show error
            mock_output.error.assert_called_with("Token name required for get command")
            mock_output.hint.assert_called()
            assert result.success is False
