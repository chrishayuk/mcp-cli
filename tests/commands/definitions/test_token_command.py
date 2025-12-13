"""Tests for the token command."""

import pytest
from unittest.mock import patch, Mock
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
        assert command.modes == (
            CommandMode.CLI | CommandMode.CHAT | CommandMode.INTERACTIVE
        )
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
                    result = await command.execute(
                        args=["list"], tool_manager=Mock(servers=[])
                    )

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
                    result = await command.execute(
                        args=["clear"], tool_manager=Mock(servers=[])
                    )

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
                result = await command.execute(
                    args=["clear", "--force"], tool_manager=Mock(servers=[])
                )

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
                result = await command.execute(
                    args=["clear", "-f"], tool_manager=Mock(servers=[])
                )

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
                result = await command.execute(
                    args=["delete", "github"], tool_manager=Mock(servers=[])
                )

                # Should call delete action with the token name
                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_delete_without_name(self, command):
        """Test execution with 'delete' subcommand but no token name."""
        with patch("mcp_cli.commands.tokens.token.output"):
            result = await command.execute(
                args=["delete"], tool_manager=Mock(servers=[])
            )

            # Should show error
            assert result.success is False
            assert "Token name is required" in result.error

    @pytest.mark.asyncio
    async def test_execute_unknown_subcommand(self, command):
        """Test execution with unknown subcommand."""
        with patch("mcp_cli.commands.tokens.token.output"):
            result = await command.execute(
                args=["unknown"], tool_manager=Mock(servers=[])
            )

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
                    result = await command.execute(
                        args="list", tool_manager=Mock(servers=[])
                    )

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
                    result = await command.execute(
                        args=[], tool_manager=Mock(servers=[])
                    )

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
                    result = await command.execute(
                        args=["LIST"], tool_manager=Mock(servers=[])
                    )

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
                result = await command.execute(
                    args=["set", "my-api", "secret-token"],
                    tool_manager=Mock(servers=[]),
                )

                # Should call set action with the token name and value
                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_set_without_value(self, command):
        """Test execution with 'set' subcommand but no token value."""
        with patch("chuk_term.ui.output"):
            with patch("getpass.getpass") as mock_getpass:
                mock_getpass.return_value = ""  # Empty value
                # Pass name via kwargs since we're testing the case where value is prompted
                result = await command.execute(
                    args=["set"], name="my-api", tool_manager=Mock(servers=[])
                )

                # Should show error when getpass returns empty
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
                result = await command.execute(
                    args=["get", "notion"], tool_manager=Mock(servers=[])
                )

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

    @pytest.mark.asyncio
    async def test_execute_backends(self, command):
        """Test execution with 'backends' subcommand."""
        with patch("mcp_cli.commands.tokens.token.TokenStoreFactory") as mock_factory:
            from mcp_cli.auth import TokenStoreBackend

            mock_factory.get_available_backends.return_value = [
                TokenStoreBackend.ENCRYPTED_FILE
            ]
            mock_factory._detect_backend.return_value = TokenStoreBackend.ENCRYPTED_FILE
            with patch("mcp_cli.commands.tokens.token.output"):
                with patch("mcp_cli.commands.tokens.token.format_table"):
                    result = await command.execute(
                        args=["backends"], tool_manager=Mock(servers=[])
                    )

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_set_provider(self, command):
        """Test execution with 'set-provider' subcommand."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_mgr.return_value = mock_manager
            with patch(
                "mcp_cli.auth.provider_tokens.set_provider_token"
            ) as mock_set_provider:
                mock_set_provider.return_value = True
                with patch(
                    "mcp_cli.auth.provider_tokens.get_provider_env_var_name"
                ) as mock_env:
                    mock_env.return_value = "ANTHROPIC_API_KEY"
                    with patch("mcp_cli.commands.tokens.token.output"):
                        result = await command.execute(
                            args=["set-provider"],
                            provider="anthropic",
                            api_key="test-key",
                            tool_manager=Mock(servers=[]),
                        )

                        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_set_provider_no_provider(self, command):
        """Test execution with 'set-provider' but no provider name."""
        with patch("mcp_cli.commands.tokens.token.output"):
            result = await command.execute(
                args=["set-provider"], tool_manager=Mock(servers=[])
            )

            assert result.success is False
            assert "Provider name is required" in result.error

    @pytest.mark.asyncio
    async def test_execute_get_provider(self, command):
        """Test execution with 'get-provider' subcommand."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_mgr.return_value = mock_manager
            with patch(
                "mcp_cli.auth.provider_tokens.check_provider_token_status"
            ) as mock_status:
                mock_status.return_value = {
                    "has_token": True,
                    "source": "storage",
                    "env_var": "ANTHROPIC_API_KEY",
                    "in_env": False,
                    "in_storage": True,
                }
                with patch("mcp_cli.commands.tokens.token.output"):
                    result = await command.execute(
                        args=["get-provider"],
                        provider="anthropic",
                        tool_manager=Mock(servers=[]),
                    )

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_get_provider_no_token(self, command):
        """Test execution with 'get-provider' but no token found."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_mgr.return_value = mock_manager
            with patch(
                "mcp_cli.auth.provider_tokens.check_provider_token_status"
            ) as mock_status:
                mock_status.return_value = {
                    "has_token": False,
                    "source": None,
                    "env_var": "ANTHROPIC_API_KEY",
                    "in_env": False,
                    "in_storage": False,
                }
                with patch("mcp_cli.commands.tokens.token.output"):
                    result = await command.execute(
                        args=["get-provider"],
                        provider="anthropic",
                        tool_manager=Mock(servers=[]),
                    )

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_get_provider_no_provider(self, command):
        """Test execution with 'get-provider' but no provider name."""
        with patch("mcp_cli.commands.tokens.token.output"):
            result = await command.execute(
                args=["get-provider"], tool_manager=Mock(servers=[])
            )

            assert result.success is False
            assert "Provider name is required" in result.error

    @pytest.mark.asyncio
    async def test_execute_delete_provider(self, command):
        """Test execution with 'delete-provider' subcommand."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_mgr.return_value = mock_manager
            with patch(
                "mcp_cli.auth.provider_tokens.delete_provider_token"
            ) as mock_delete:
                mock_delete.return_value = True
                with patch("mcp_cli.commands.tokens.token.output"):
                    result = await command.execute(
                        args=["delete-provider"],
                        provider="anthropic",
                        tool_manager=Mock(servers=[]),
                    )

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_delete_provider_not_found(self, command):
        """Test execution with 'delete-provider' but token not found."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_mgr.return_value = mock_manager
            with patch(
                "mcp_cli.auth.provider_tokens.delete_provider_token"
            ) as mock_delete:
                mock_delete.return_value = False
                with patch("mcp_cli.commands.tokens.token.output"):
                    result = await command.execute(
                        args=["delete-provider"],
                        provider="anthropic",
                        tool_manager=Mock(servers=[]),
                    )

                    assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_delete_provider_no_provider(self, command):
        """Test execution with 'delete-provider' but no provider name."""
        with patch("mcp_cli.commands.tokens.token.output"):
            result = await command.execute(
                args=["delete-provider"], tool_manager=Mock(servers=[])
            )

            assert result.success is False
            assert "Provider name is required" in result.error

    @pytest.mark.asyncio
    async def test_execute_set_api_key(self, command):
        """Test execution with 'set' subcommand with api-key type."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.token_store = Mock()
            mock_manager.registry = Mock()
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["set", "my-api", "secret-token"],
                    token_type="api-key",
                    provider="openai",
                    tool_manager=Mock(servers=[]),
                )

                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_set_api_key_no_provider(self, command):
        """Test execution with 'set' api-key type but no provider."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.token_store = Mock()
            mock_manager.registry = Mock()
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["set", "my-api", "secret-token"],
                    token_type="api-key",
                    tool_manager=Mock(servers=[]),
                )

                assert result.success is False
                assert "Provider name is required" in result.error

    @pytest.mark.asyncio
    async def test_execute_set_generic(self, command):
        """Test execution with 'set' subcommand with generic type."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.token_store = Mock()
            mock_manager.registry = Mock()
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["set", "my-token", "secret-value"],
                    token_type="generic",
                    tool_manager=Mock(servers=[]),
                )

                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_set_unknown_type(self, command):
        """Test execution with 'set' subcommand with unknown type."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.token_store = Mock()
            mock_manager.registry = Mock()
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["set", "my-token", "secret-value"],
                    token_type="invalid-type",
                    tool_manager=Mock(servers=[]),
                )

                assert result.success is False
                assert "Unknown token type" in result.error

    @pytest.mark.asyncio
    async def test_execute_get_found_in_oauth(self, command):
        """Test execution with 'get' finding token in OAuth namespace."""
        import json
        from mcp_cli.auth import TokenType

        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_store = Mock()
            # First call returns None (generic namespace), second returns token (OAuth)
            mock_store._retrieve_raw.side_effect = [
                None,
                json.dumps(
                    {
                        "token_type": TokenType.BEARER.value,
                        "token": "test-token",
                        "created_at": 1234567890,
                    }
                ),
            ]
            mock_manager.token_store = mock_store
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["get", "notion"], tool_manager=Mock(servers=[])
                )

                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_delete_oauth(self, command):
        """Test execution with 'delete' subcommand for OAuth token."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.delete_tokens.return_value = True
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["delete", "notion"],
                    is_oauth=True,
                    tool_manager=Mock(servers=[]),
                )

                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_delete_oauth_not_found(self, command):
        """Test execution with 'delete' OAuth token not found."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.delete_tokens.return_value = False
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["delete", "notion"],
                    is_oauth=True,
                    tool_manager=Mock(servers=[]),
                )

                # Should return True even when not found (just warning)
                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_delete_with_namespace(self, command):
        """Test execution with 'delete' and specific namespace."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.token_store = Mock()
            mock_manager.token_store.delete_generic.return_value = True
            mock_manager.registry = Mock()
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["delete", "my-token"],
                    namespace="bearer",
                    tool_manager=Mock(servers=[]),
                )

                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_delete_not_found(self, command):
        """Test execution with 'delete' token not found."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.token_store = Mock()
            mock_manager.token_store.delete_generic.return_value = False
            mock_manager.registry = Mock()
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["delete", "my-token"],
                    tool_manager=Mock(servers=[]),
                )

                assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_clear_with_tokens(self, command):
        """Test execution with 'clear' subcommand when tokens exist."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.registry.list_tokens.return_value = [
                {"name": "token1", "namespace": "generic"}
            ]
            mock_manager.token_store.delete_generic.return_value = True
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["clear", "--force"],
                    tool_manager=Mock(servers=[]),
                )

                assert result.success is True
                mock_manager.registry.clear_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_clear_with_namespace(self, command):
        """Test execution with 'clear' subcommand with specific namespace."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.registry.list_tokens.return_value = [
                {"name": "token1", "namespace": "bearer"}
            ]
            mock_manager.token_store.delete_generic.return_value = True
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["clear", "--force"],
                    namespace="bearer",
                    tool_manager=Mock(servers=[]),
                )

                assert result.success is True
                mock_manager.registry.clear_namespace.assert_called_once_with("bearer")

    @pytest.mark.asyncio
    async def test_execute_list_with_oauth_tokens(self, command):
        """Test execution with 'list' showing OAuth tokens."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.registry.list_tokens.return_value = []
            # Mock load_tokens to return OAuth tokens
            mock_tokens = Mock()
            mock_tokens.expires_in = 3600
            mock_tokens.issued_at = 1234567890
            mock_manager.load_tokens.return_value = mock_tokens
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                with patch("mcp_cli.commands.tokens.token.format_table"):
                    result = await command.execute(
                        tool_manager=Mock(servers=["notion-server"])
                    )

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_list_with_provider_tokens(self, command):
        """Test execution with 'list' showing provider tokens."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.registry.list_tokens.return_value = []
            mock_manager.load_tokens.return_value = None
            mock_mgr.return_value = mock_manager
            with patch(
                "mcp_cli.auth.provider_tokens.list_all_provider_tokens"
            ) as mock_list:
                mock_list.return_value = {
                    "anthropic": {
                        "env_var": "ANTHROPIC_API_KEY",
                        "in_env": False,
                    }
                }
                with patch("mcp_cli.commands.tokens.token.output"):
                    with patch("mcp_cli.commands.tokens.token.format_table"):
                        result = await command.execute(tool_manager=Mock(servers=[]))

                        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_list_with_provider_tokens_env_override(self, command):
        """Test execution with 'list' showing provider tokens with env override."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.registry.list_tokens.return_value = []
            mock_manager.load_tokens.return_value = None
            mock_mgr.return_value = mock_manager
            with patch(
                "mcp_cli.auth.provider_tokens.list_all_provider_tokens"
            ) as mock_list:
                mock_list.return_value = {
                    "anthropic": {
                        "env_var": "ANTHROPIC_API_KEY",
                        "in_env": True,
                    }
                }
                with patch("mcp_cli.commands.tokens.token.output"):
                    with patch("mcp_cli.commands.tokens.token.format_table"):
                        result = await command.execute(tool_manager=Mock(servers=[]))

                        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_list_error(self, command):
        """Test execution with 'list' when error occurs."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_mgr.side_effect = Exception("Token manager error")
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(tool_manager=Mock(servers=[]))

                assert result.success is False
                assert "Error listing tokens" in result.error

    @pytest.mark.asyncio
    async def test_execute_set_error(self, command):
        """Test execution with 'set' when error occurs."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_mgr.side_effect = Exception("Store error")
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["set", "token", "value"],
                    tool_manager=Mock(servers=[]),
                )

                assert result.success is False
                assert "Error storing token" in result.error

    @pytest.mark.asyncio
    async def test_execute_get_error(self, command):
        """Test execution with 'get' when error occurs."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_mgr.side_effect = Exception("Retrieve error")
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["get", "token"],
                    tool_manager=Mock(servers=[]),
                )

                assert result.success is False
                assert "Error retrieving token" in result.error

    @pytest.mark.asyncio
    async def test_execute_delete_error(self, command):
        """Test execution with 'delete' when error occurs."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_mgr.side_effect = Exception("Delete error")
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["delete", "token"],
                    tool_manager=Mock(servers=[]),
                )

                assert result.success is False
                assert "Error deleting token" in result.error

    @pytest.mark.asyncio
    async def test_execute_clear_error(self, command):
        """Test execution with 'clear' when error occurs."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_mgr.side_effect = Exception("Clear error")
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["clear", "--force"],
                    tool_manager=Mock(servers=[]),
                )

                assert result.success is False
                assert "Error clearing tokens" in result.error

    @pytest.mark.asyncio
    async def test_execute_backends_error(self, command):
        """Test execution with 'backends' when error occurs."""
        with patch("mcp_cli.commands.tokens.token.TokenStoreFactory") as mock_factory:
            mock_factory.get_available_backends.side_effect = Exception("Backend error")
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["backends"], tool_manager=Mock(servers=[])
                )

                assert result.success is False
                assert "Error listing backends" in result.error

    @pytest.mark.asyncio
    async def test_execute_set_provider_error(self, command):
        """Test execution with 'set-provider' when error occurs."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_mgr.side_effect = Exception("Provider error")
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["set-provider"],
                    provider="anthropic",
                    api_key="key",
                    tool_manager=Mock(servers=[]),
                )

                assert result.success is False
                assert "Error storing provider token" in result.error

    @pytest.mark.asyncio
    async def test_execute_get_provider_error(self, command):
        """Test execution with 'get-provider' when error occurs."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_mgr.side_effect = Exception("Provider error")
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["get-provider"],
                    provider="anthropic",
                    tool_manager=Mock(servers=[]),
                )

                assert result.success is False
                assert "Error retrieving provider token info" in result.error

    @pytest.mark.asyncio
    async def test_execute_delete_provider_error(self, command):
        """Test execution with 'delete-provider' when error occurs."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_mgr.side_effect = Exception("Provider error")
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["delete-provider"],
                    provider="anthropic",
                    tool_manager=Mock(servers=[]),
                )

                assert result.success is False
                assert "Error deleting provider token" in result.error

    @pytest.mark.asyncio
    async def test_execute_exception_wrapper(self, command):
        """Test that general exceptions are caught by execute wrapper."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_mgr.return_value = mock_manager
            with patch.object(
                command, "_action_list", side_effect=Exception("Unexpected error")
            ):
                result = await command.execute(tool_manager=Mock(servers=[]))

                assert result.success is False
                assert "Token command error" in result.error

    @pytest.mark.asyncio
    async def test_execute_list_with_registered_tokens(self, command):
        """Test execution with 'list' showing registered tokens."""
        import time
        from mcp_cli.auth import TokenType

        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            # Return tokens in registry
            mock_manager.registry.list_tokens.return_value = [
                {
                    "name": "my-token",
                    "type": TokenType.BEARER.value,
                    "namespace": "generic",
                    "registered_at": time.time(),
                    "metadata": {"expires_at": time.time() + 3600},
                }
            ]
            mock_manager.load_tokens.return_value = None
            mock_mgr.return_value = mock_manager
            with patch(
                "mcp_cli.auth.provider_tokens.list_all_provider_tokens"
            ) as mock_list:
                mock_list.return_value = {}
                with patch("mcp_cli.commands.tokens.token.output"):
                    with patch("mcp_cli.commands.tokens.token.format_table"):
                        result = await command.execute(tool_manager=Mock(servers=[]))

                        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_list_with_expired_token(self, command):
        """Test execution with 'list' showing expired token."""
        import time
        from mcp_cli.auth import TokenType

        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            # Return expired token
            mock_manager.registry.list_tokens.return_value = [
                {
                    "name": "expired-token",
                    "type": TokenType.BEARER.value,
                    "namespace": "generic",
                    "registered_at": time.time() - 7200,
                    "metadata": {"expires_at": time.time() - 3600},  # Expired
                }
            ]
            mock_manager.load_tokens.return_value = None
            mock_mgr.return_value = mock_manager
            with patch(
                "mcp_cli.auth.provider_tokens.list_all_provider_tokens"
            ) as mock_list:
                mock_list.return_value = {}
                with patch("mcp_cli.commands.tokens.token.output"):
                    with patch("mcp_cli.commands.tokens.token.format_table"):
                        result = await command.execute(tool_manager=Mock(servers=[]))

                        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_list_with_api_key_token(self, command):
        """Test execution with 'list' showing API key token."""
        import time
        from mcp_cli.auth import TokenType

        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.registry.list_tokens.return_value = [
                {
                    "name": "api-key",
                    "type": TokenType.API_KEY.value,
                    "namespace": "api-key",
                    "registered_at": time.time(),
                    "metadata": {"provider": "openai"},
                }
            ]
            mock_manager.load_tokens.return_value = None
            mock_mgr.return_value = mock_manager
            with patch(
                "mcp_cli.auth.provider_tokens.list_all_provider_tokens"
            ) as mock_list:
                mock_list.return_value = {}
                with patch("mcp_cli.commands.tokens.token.output"):
                    with patch("mcp_cli.commands.tokens.token.format_table"):
                        result = await command.execute(tool_manager=Mock(servers=[]))

                        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_list_oauth_with_no_issued_at(self, command):
        """Test execution with 'list' showing OAuth token without issued_at."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.registry.list_tokens.return_value = []
            mock_tokens = Mock()
            mock_tokens.expires_in = 3600
            mock_tokens.issued_at = None  # No issued_at
            mock_manager.load_tokens.return_value = mock_tokens
            mock_mgr.return_value = mock_manager
            with patch(
                "mcp_cli.auth.provider_tokens.list_all_provider_tokens"
            ) as mock_list:
                mock_list.return_value = {}
                with patch("mcp_cli.commands.tokens.token.output"):
                    with patch("mcp_cli.commands.tokens.token.format_table"):
                        result = await command.execute(
                            tool_manager=Mock(servers=["notion"])
                        )

                        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_list_oauth_expired(self, command):
        """Test execution with 'list' showing expired OAuth token."""
        import time

        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.registry.list_tokens.return_value = []
            mock_tokens = Mock()
            mock_tokens.expires_in = 3600
            mock_tokens.issued_at = (
                time.time() - 7200
            )  # Issued 2 hours ago, expires in 1 hour
            mock_manager.load_tokens.return_value = mock_tokens
            mock_mgr.return_value = mock_manager
            with patch(
                "mcp_cli.auth.provider_tokens.list_all_provider_tokens"
            ) as mock_list:
                mock_list.return_value = {}
                with patch("mcp_cli.commands.tokens.token.output"):
                    with patch("mcp_cli.commands.tokens.token.format_table"):
                        result = await command.execute(
                            tool_manager=Mock(servers=["notion"])
                        )

                        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_get_with_display_info(self, command):
        """Test execution with 'get' showing full token info."""
        import json

        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_store = Mock()
            mock_store._retrieve_raw.return_value = json.dumps(
                {
                    "token_type": "bearer",
                    "token": "test-token",
                    "created_at": 1234567890,
                }
            )
            mock_manager.token_store = mock_store
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["get", "my-token"], tool_manager=Mock(servers=[])
                )

                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_get_parse_error(self, command):
        """Test execution with 'get' when token data parsing fails."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_store = Mock()
            mock_store._retrieve_raw.return_value = "invalid-json"
            mock_manager.token_store = mock_store
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.output"):
                result = await command.execute(
                    args=["get", "my-token"], tool_manager=Mock(servers=[])
                )

                # Should still return True but with warning
                assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_clear_confirm_with_namespace(self, command):
        """Test execution with 'clear' confirming with namespace."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.registry.list_tokens.return_value = [
                {"name": "token1", "namespace": "bearer"}
            ]
            mock_manager.token_store.delete_generic.return_value = True
            mock_mgr.return_value = mock_manager
            with patch("chuk_term.ui.prompts.confirm") as mock_confirm:
                mock_confirm.return_value = True  # User confirms
                with patch("mcp_cli.commands.tokens.token.output"):
                    result = await command.execute(
                        args=["clear"],
                        namespace="bearer",
                        tool_manager=Mock(servers=[]),
                    )

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_set_bearer_with_expires(self, command):
        """Test execution with 'set' bearer token that has expiration."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.token_store = Mock()
            mock_manager.registry = Mock()
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.commands.tokens.token.BearerToken") as mock_bearer:
                mock_bearer_instance = Mock()
                mock_bearer_instance.expires_at = 1234567890
                mock_bearer_instance.to_stored_token.return_value = Mock(
                    metadata={}, model_dump=Mock(return_value={})
                )
                mock_bearer.return_value = mock_bearer_instance
                with patch("mcp_cli.commands.tokens.token.output"):
                    result = await command.execute(
                        args=["set", "my-token", "secret"],
                        token_type="bearer",
                        tool_manager=Mock(servers=[]),
                    )

                    assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_set_provider_with_prompt(self, command):
        """Test execution with 'set-provider' prompting for api key."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_mgr.return_value = mock_manager
            with patch("getpass.getpass") as mock_getpass:
                mock_getpass.return_value = "prompted-key"
                with patch(
                    "mcp_cli.auth.provider_tokens.set_provider_token"
                ) as mock_set:
                    mock_set.return_value = True
                    with patch(
                        "mcp_cli.auth.provider_tokens.get_provider_env_var_name"
                    ) as mock_env:
                        mock_env.return_value = "ANTHROPIC_API_KEY"
                        with patch("mcp_cli.commands.tokens.token.output"):
                            result = await command.execute(
                                args=["set-provider"],
                                provider="anthropic",
                                tool_manager=Mock(servers=[]),
                            )

                            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_set_provider_empty_key(self, command):
        """Test execution with 'set-provider' with empty key."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_mgr.return_value = mock_manager
            with patch("getpass.getpass") as mock_getpass:
                mock_getpass.return_value = ""  # Empty
                with patch("mcp_cli.commands.tokens.token.output"):
                    result = await command.execute(
                        args=["set-provider"],
                        provider="anthropic",
                        tool_manager=Mock(servers=[]),
                    )

                    assert result.success is False
                    assert "API key is required" in result.error

    @pytest.mark.asyncio
    async def test_execute_set_provider_with_env_var_override(self, command):
        """Test execution with 'set-provider' when env var is also set."""
        import os

        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.auth.provider_tokens.set_provider_token") as mock_set:
                mock_set.return_value = True
                with patch(
                    "mcp_cli.auth.provider_tokens.get_provider_env_var_name"
                ) as mock_env:
                    mock_env.return_value = "ANTHROPIC_API_KEY"
                    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "existing-key"}):
                        with patch("mcp_cli.commands.tokens.token.output"):
                            result = await command.execute(
                                args=["set-provider"],
                                provider="anthropic",
                                api_key="new-key",
                                tool_manager=Mock(servers=[]),
                            )

                            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_set_provider_failure(self, command):
        """Test execution with 'set-provider' when storage fails."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_mgr.return_value = mock_manager
            with patch("mcp_cli.auth.provider_tokens.set_provider_token") as mock_set:
                mock_set.return_value = False  # Failure
                with patch("mcp_cli.commands.tokens.token.output"):
                    result = await command.execute(
                        args=["set-provider"],
                        provider="anthropic",
                        api_key="test-key",
                        tool_manager=Mock(servers=[]),
                    )

                    assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_backends_with_env_override(self, command):
        """Test execution with 'backends' when env override is set."""
        import os
        from mcp_cli.auth import TokenStoreBackend

        with patch.dict(os.environ, {"MCP_CLI_TOKEN_BACKEND": "encrypted"}):
            with patch(
                "mcp_cli.commands.tokens.token.TokenStoreFactory"
            ) as mock_factory:
                mock_factory.get_available_backends.return_value = [
                    TokenStoreBackend.ENCRYPTED_FILE
                ]
                with patch(
                    "mcp_cli.commands.tokens.token.TokenStoreBackend"
                ) as mock_backend:
                    mock_backend.return_value = TokenStoreBackend.ENCRYPTED_FILE
                    mock_backend.side_effect = (
                        lambda x: TokenStoreBackend.ENCRYPTED_FILE
                        if x == "encrypted"
                        else TokenStoreBackend.KEYCHAIN
                    )
                    with patch("mcp_cli.commands.tokens.token.output"):
                        with patch("mcp_cli.commands.tokens.token.format_table"):
                            result = await command.execute(
                                args=["backends"], tool_manager=Mock(servers=[])
                            )

                            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_backends_with_invalid_env_override(self, command):
        """Test execution with 'backends' when env override is invalid."""
        import os
        from mcp_cli.auth import TokenStoreBackend

        with patch.dict(os.environ, {"MCP_CLI_TOKEN_BACKEND": "invalid_backend"}):
            with patch(
                "mcp_cli.commands.tokens.token.TokenStoreFactory"
            ) as mock_factory:
                mock_factory.get_available_backends.return_value = [
                    TokenStoreBackend.ENCRYPTED_FILE
                ]
                mock_factory._detect_backend.return_value = (
                    TokenStoreBackend.ENCRYPTED_FILE
                )
                with patch("mcp_cli.commands.tokens.token.output"):
                    with patch("mcp_cli.commands.tokens.token.format_table"):
                        result = await command.execute(
                            args=["backends"], tool_manager=Mock(servers=[])
                        )

                        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_list_with_no_servers_configured(self, command):
        """Test execution with 'list' showing no servers message."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.registry.list_tokens.return_value = []
            mock_manager.load_tokens.return_value = None
            mock_mgr.return_value = mock_manager
            with patch(
                "mcp_cli.auth.provider_tokens.list_all_provider_tokens"
            ) as mock_list:
                mock_list.return_value = {}
                with patch("mcp_cli.commands.tokens.token.output"):
                    with patch("mcp_cli.commands.tokens.token.format_table"):
                        result = await command.execute(
                            show_providers=False,
                            show_oauth=True,
                            tool_manager=Mock(servers=[]),
                        )

                        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_set_with_value_prompt(self, command):
        """Test execution with 'set' prompting for value."""
        with patch("mcp_cli.commands.tokens.token._get_token_manager") as mock_mgr:
            mock_manager = Mock()
            mock_manager.token_store = Mock()
            mock_manager.registry = Mock()
            mock_mgr.return_value = mock_manager
            with patch("getpass.getpass") as mock_getpass:
                mock_getpass.return_value = "prompted-value"
                with patch("mcp_cli.commands.tokens.token.output"):
                    result = await command.execute(
                        name="my-token",
                        tool_manager=Mock(servers=[]),
                        action="set",
                    )

                    assert result.success is True


class TestGetTokenManager:
    """Test _get_token_manager function."""

    def test_get_token_manager_with_env_override(self):
        """Test _get_token_manager with environment variable override."""
        import os
        from mcp_cli.commands.tokens.token import _get_token_manager

        with patch.dict(os.environ, {"MCP_CLI_TOKEN_BACKEND": "encrypted"}):
            with patch("mcp_cli.commands.tokens.token.TokenManager") as mock_tm:
                _get_token_manager()
                # Should use encrypted backend
                mock_tm.assert_called_once()

    def test_get_token_manager_with_invalid_env_override(self):
        """Test _get_token_manager with invalid environment variable."""
        import os
        from mcp_cli.commands.tokens.token import _get_token_manager

        with patch.dict(os.environ, {"MCP_CLI_TOKEN_BACKEND": "invalid_backend"}):
            with patch("mcp_cli.commands.tokens.token.get_config") as mock_config:
                mock_config.return_value.token_store_backend = "encrypted"
                with patch("mcp_cli.commands.tokens.token.TokenManager") as mock_tm:
                    _get_token_manager()
                    mock_tm.assert_called_once()

    def test_get_token_manager_config_error(self):
        """Test _get_token_manager when config raises error."""
        import os
        from mcp_cli.commands.tokens.token import _get_token_manager, TokenStoreBackend

        # Clear the env var so it falls through to config
        with patch.dict(os.environ, {}, clear=True):
            if "MCP_CLI_TOKEN_BACKEND" in os.environ:
                del os.environ["MCP_CLI_TOKEN_BACKEND"]
            with patch("mcp_cli.commands.tokens.token.get_config") as mock_config:
                mock_config.side_effect = Exception("Config error")
                with patch("mcp_cli.commands.tokens.token.TokenManager") as mock_tm:
                    _get_token_manager()
                    # Should default to AUTO
                    call_args = mock_tm.call_args
                    assert call_args.kwargs["backend"] == TokenStoreBackend.AUTO
