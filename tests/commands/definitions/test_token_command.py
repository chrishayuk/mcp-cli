"""Tests for the token command."""

import pytest
from unittest.mock import patch, AsyncMock, Mock
from mcp_cli.commands.tokens.token import TokenCommand
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
        assert command.modes == (CommandMode.CLI | CommandMode.CHAT | CommandMode.INTERACTIVE)
        assert command.requires_context  # Needs context to get server list

    @pytest.mark.asyncio
    async def test_execute_no_args(self, command):
        """Test execution with no arguments - should list tokens."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.registry.list_tokens.return_value = []
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                with patch("mcp_cli.commands.tokens.token.format_table"):
                    result = await command.execute(tool_manager=Mock(servers=[]))

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_list_subcommand(self, command):
        """Test execution with 'list' subcommand."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.registry.list_tokens.return_value = []
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                with patch("mcp_cli.commands.tokens.token.format_table"):
                    result = await command.execute(args=["list"], tool_manager=Mock(servers=[]))

                    # Should call list action
                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_clear_without_force(self, command):
        """Test execution with 'clear' subcommand without force flag."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.registry.list_tokens.return_value = []
            mock_mgr.return_value = mock_manager
            with patch("chuk_term.ui.prompts.confirm") as mock_confirm:
                mock_confirm.return_value = False  # User cancels
                with patch("chuk_term.ui.output"):
                    result = await command.execute(args=["clear"], tool_manager=Mock(servers=[]))

                    # Should be cancelled
                    assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_clear_with_force_long(self, command):
        """Test execution with 'clear' subcommand with --force flag."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.registry.list_tokens.return_value = []
            mock_manager.token_store = Mock()
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(args=["clear", "--force"], tool_manager=Mock(servers=[]))

                # Should call clear with force=True
                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_clear_with_force_short(self, command):
        """Test execution with 'clear' subcommand with -f flag."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.registry.list_tokens.return_value = []
            mock_manager.token_store = Mock()
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(args=["clear", "-f"], tool_manager=Mock(servers=[]))

                # Should call clear with force=True
                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_delete_with_name(self, command):
        """Test execution with 'delete' subcommand and token name."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.token_store = Mock()
            mock_manager.token_store.delete_generic.return_value = True
            mock_manager.registry = Mock()
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(args=["delete", "github"], tool_manager=Mock(servers=[]))

                # Should call delete action with the token name
                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_delete_without_name(self, command):
        """Test execution with 'delete' subcommand but no token name."""
        with patch("mcp_cli.commands.tokens.token.output") as mock_output:
            result = await command.execute(args=["delete"], tool_manager=Mock(servers=[]))

            # Should show error
            assert result.success is False
            assert "Token name is required" in result.error

    @pytest.mark.asyncio
    async def test_execute_unknown_subcommand(self, command):
        """Test execution with unknown subcommand."""
        with patch("mcp_cli.commands.tokens.token.output"):
            result = await command.execute(args=["unknown"], tool_manager=Mock(servers=[]))

            # Should show error
            assert result.success is False
            assert "Unknown token action" in result.error

    @pytest.mark.asyncio
    async def test_execute_string_arg(self, command):
        """Test execution with string argument instead of list."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.registry.list_tokens.return_value = []
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                with patch("mcp_cli.commands.tokens.token.format_table"):
                    # Pass a string arg (should be converted to list)
                    result = await command.execute(args="list", tool_manager=Mock(servers=[]))

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_empty_list(self, command):
        """Test execution with empty list of arguments."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.registry.list_tokens.return_value = []
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                with patch("mcp_cli.commands.tokens.token.format_table"):
                    result = await command.execute(args=[], tool_manager=Mock(servers=[]))

                    # Should call list action (default)
                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_case_insensitive_subcommand(self, command):
        """Test that subcommands are case-insensitive."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.registry.list_tokens.return_value = []
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                with patch("mcp_cli.commands.tokens.token.format_table"):
                    result = await command.execute(args=["LIST"], tool_manager=Mock(servers=[]))

                    # Should call list action (subcommand converted to lowercase)
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
        # Should be available in all modes
        assert CommandMode.CHAT in command.modes
        assert CommandMode.INTERACTIVE in command.modes
        assert CommandMode.CLI in command.modes

    @pytest.mark.asyncio
    async def test_execute_set_with_name_and_value(self, command):
        """Test execution with 'set' subcommand and token name/value."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.token_store = Mock()
            mock_manager.registry = Mock()
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(args=["set", "my-api", "secret-token"], tool_manager=Mock(servers=[]))

                # Should call set action with the token name and value
                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_set_without_value(self, command):
        """Test execution with 'set' subcommand but no token value."""
        with patch("chuk_term.ui.output"):
            with patch("getpass.getpass") as mock_getpass:
                mock_getpass.return_value = ""  # Empty value
                result = await command.execute(args=["set", "my-api"], tool_manager=Mock(servers=[]))

                # Should show error
                assert result.success is False
                assert "Token value is required" in result.error

    @pytest.mark.asyncio
    async def test_execute_set_without_name_or_value(self, command):
        """Test execution with 'set' subcommand but no arguments."""
        with patch("mcp_cli.commands.tokens.token.output"):
            result = await command.execute(args=["set"], tool_manager=Mock(servers=[]))

            # Should show error
            assert result.success is False
            assert "Token name is required" in result.error

    @pytest.mark.asyncio
    async def test_execute_get_with_name(self, command):
        """Test execution with 'get' subcommand and token name."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.token_store = Mock()
            mock_manager.token_store._retrieve_raw.return_value = None
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(args=["get", "notion"], tool_manager=Mock(servers=[]))

                # Should call get action
                assert result.success is False  # Token not found

    @pytest.mark.asyncio
    async def test_execute_get_without_name(self, command):
        """Test execution with 'get' subcommand but no token name."""
        with patch("mcp_cli.commands.tokens.token.output"):
            result = await command.execute(args=["get"], tool_manager=Mock(servers=[]))

            # Should show error
            assert result.success is False
            assert "Token name is required" in result.error
