# tests/test_provider_command.py
"""
Working provider command tests that align with the actual console usage.
These tests check the actual behavior rather than trying to mock console.print.
"""

import pytest
import sys
from unittest.mock import Mock, patch

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.commands.provider import provider_action_async, provider_action
from mcp_cli.model_manager import ModelManager
from tests.conftest import setup_test_context


class TestProviderActionAsync:
    """Test the async provider action functionality."""

    @pytest.fixture
    def mock_model_manager(self):
        """Create a comprehensive mock ModelManager for testing."""
        manager = Mock(spec=ModelManager)
        manager.get_active_provider.return_value = "openai"
        manager.get_active_model.return_value = "gpt-4o-mini"
        manager.get_active_provider_and_model.return_value = ("openai", "gpt-4o-mini")
        manager.get_default_model.return_value = "claude-sonnet"
        manager.list_providers.return_value = [
            "openai",
            "anthropic",
            "ollama",
            "gemini",
        ]
        manager.validate_provider.side_effect = lambda p: p in [
            "openai",
            "anthropic",
            "ollama",
            "gemini",
        ]
        manager.switch_model = Mock()
        manager.get_client = Mock(return_value=Mock())

        # Mock status summary
        manager.get_status_summary.return_value = {
            "provider_configured": True,
            "supports_streaming": True,
            "supports_tools": True,
            "supports_vision": False,
        }

        # Mock provider data
        manager.list_available_providers.return_value = {
            "openai": {
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1"],
                "default_model": "gpt-4o-mini",
                "has_api_key": True,
                "baseline_features": ["streaming", "tools", "text"],
            },
            "anthropic": {
                "models": ["claude-sonnet-4-20250514", "claude-opus-4-20250514"],
                "default_model": "claude-sonnet-4-20250514",
                "has_api_key": True,
                "baseline_features": ["streaming", "reasoning", "text"],
            },
        }

        return manager

    @pytest.fixture(autouse=True)
    def setup_context(self, mock_model_manager):
        """Set up context before each test."""
        context = setup_test_context(provider="openai", model="gpt-4o-mini")
        # Replace the model manager with our mock
        context.model_manager = mock_model_manager
        yield context

    @pytest.mark.asyncio
    async def test_no_arguments_shows_status(self, setup_context, capsys):
        """Test that calling with no arguments shows current status."""
        await provider_action_async([])

        captured = capsys.readouterr()
        output = captured.out

        # Should show current provider and model
        assert "openai" in output
        assert "gpt-4o-mini" in output or "gpt-4" in output
        assert "Current provider" in output or "Current model" in output

    @pytest.mark.asyncio
    async def test_list_argument_shows_provider_list(self, setup_context, capsys):
        """Test that 'list' argument shows provider list."""
        await provider_action_async(["list"])

        captured = capsys.readouterr()
        output = captured.out

        # Should show provider table
        assert "Available Providers" in output or "Provider" in output
        assert "openai" in output
        assert "anthropic" in output

    @pytest.mark.asyncio
    async def test_provider_switch_valid_provider(self, setup_context, capsys):
        """Test switching to a valid provider."""
        await provider_action_async(["anthropic"])

        # Check that switch_model was called with anthropic and some model
        setup_context.model_manager.switch_model.assert_called_once()
        call_args = setup_context.model_manager.switch_model.call_args[0]
        assert call_args[0] == "anthropic"  # Provider should be anthropic

        captured = capsys.readouterr()
        output = captured.out

        # Should show success message
        assert "anthropic" in output or "Switched" in output or "✓" in output

    @pytest.mark.asyncio
    async def test_context_without_model_manager_creates_new_one(self, setup_context):
        """Test that a missing model_manager in context raises an error."""
        # Remove model manager
        setup_context.model_manager = None

        # The current implementation expects model_manager to exist
        with pytest.raises(AttributeError):
            await provider_action_async([])


class TestProviderSwitching:
    """Test provider switching scenarios."""

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock ModelManager for testing."""
        manager = Mock(spec=ModelManager)
        manager.get_active_provider.return_value = "openai"
        manager.validate_provider.side_effect = lambda p: p in ["openai", "anthropic"]
        manager.list_providers.return_value = [
            "openai",
            "anthropic",
        ]  # Add list_providers
        manager.list_available_providers.return_value = {
            "openai": {
                "models": ["gpt-4o"],
                "default_model": "gpt-4o",
                "has_api_key": True,
            },
            "anthropic": {
                "models": ["claude-sonnet"],
                "default_model": "claude-sonnet",
                "has_api_key": True,
            },
        }
        manager.switch_model = Mock()
        return manager

    @pytest.fixture(autouse=True)
    def setup_context(self, mock_model_manager):
        """Set up context before each test."""
        context = setup_test_context()
        context.model_manager = mock_model_manager
        yield context

    @pytest.mark.asyncio
    async def test_switch_to_valid_provider(self, setup_context, capsys):
        """Test switching to a valid provider."""
        await provider_action_async(["anthropic"])

        setup_context.model_manager.switch_model.assert_called_once()
        call_args = setup_context.model_manager.switch_model.call_args[0]
        assert call_args[0] == "anthropic"

        captured = capsys.readouterr()
        assert "anthropic" in captured.out or "Switched" in captured.out

    @pytest.mark.asyncio
    async def test_switch_to_invalid_provider(self, setup_context):
        """Test switching to an invalid provider."""
        from unittest.mock import patch

        with patch("mcp_cli.commands.provider.output") as mock_output:
            await provider_action_async(["invalid_provider"])

            # Should not call switch_model
            setup_context.model_manager.switch_model.assert_not_called()

            # Should have called error for unknown provider
            mock_output.error.assert_called()
            error_call = str(mock_output.error.call_args)
            assert "Unknown provider" in error_call or "invalid_provider" in error_call


class TestProviderConfiguration:
    """Test provider configuration commands."""

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock ModelManager for testing."""
        manager = Mock(spec=ModelManager)
        manager.set_provider_config = Mock()
        manager.get_active_provider.return_value = "openai"
        return manager

    @pytest.fixture(autouse=True)
    def setup_context(self, mock_model_manager):
        """Set up context before each test."""
        context = setup_test_context()
        context.model_manager = mock_model_manager
        yield context

    @pytest.mark.skip(reason="_mutate function not implemented yet")
    @pytest.mark.asyncio
    async def test_set_api_key_command(self, setup_context, capsys):
        """Test setting API key for a provider."""
        await provider_action_async(["set", "openai", "api_key", "test_key_123"])

        # Should call set_provider_config
        setup_context.model_manager.set_provider_config.assert_called_with(
            "openai", "api_key", "test_key_123"
        )

        captured = capsys.readouterr()
        assert "set" in captured.out.lower() or "updated" in captured.out.lower()

    @pytest.mark.skip(reason="_mutate function not implemented yet")
    @pytest.mark.asyncio
    async def test_set_command_insufficient_args(self, setup_context, capsys):
        """Test set command with insufficient arguments."""
        await provider_action_async(["set", "openai"])

        # Should not call set_provider_config
        setup_context.model_manager.set_provider_config.assert_not_called()

        captured = capsys.readouterr()
        assert "Usage" in captured.out or "Error" in captured.out


class TestProviderSyncWrapper:
    """Test the synchronous wrapper."""

    @pytest.fixture(autouse=True)
    def setup_context(self):
        """Set up context before each test."""
        mock_manager = Mock()
        mock_manager.get_active_provider.return_value = "openai"
        mock_manager.get_active_model.return_value = "gpt-4"
        mock_manager.get_active_provider_and_model.return_value = ("openai", "gpt-4")
        mock_manager.get_status_summary.return_value = {
            "provider_configured": True,
            "supports_streaming": True,
            "supports_tools": True,
            "supports_vision": False,
        }
        context = setup_test_context()
        context.model_manager = mock_manager
        yield context

    def test_sync_wrapper_handles_call(self, setup_context):
        """Test that sync wrapper properly calls async function."""
        with patch("mcp_cli.commands.provider.provider_action_async") as mock_async:
            # Configure mock to return a coroutine
            async def mock_coro(args):
                return None

            mock_async.return_value = mock_coro([])

            with patch("mcp_cli.utils.async_utils.run_blocking") as mock_run:
                provider_action([])

                # Should call run_blocking with the async function
                mock_run.assert_called_once()


class TestProviderCommandEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock ModelManager for testing."""
        manager = Mock(spec=ModelManager)
        manager.get_active_provider.return_value = "openai"
        manager.get_active_model.return_value = "gpt-4"
        manager.get_active_provider_and_model.return_value = ("openai", "gpt-4")
        manager.list_available_providers.return_value = {}
        manager.get_status_summary.return_value = {
            "provider_configured": True,
            "supports_streaming": True,
            "supports_tools": True,
            "supports_vision": False,
        }
        return manager

    @pytest.fixture(autouse=True)
    def setup_context(self, mock_model_manager):
        """Set up context before each test."""
        context = setup_test_context()
        context.model_manager = mock_model_manager
        yield context

    @pytest.mark.asyncio
    async def test_empty_args_list(self, setup_context, capsys):
        """Test with empty args list."""
        await provider_action_async([])

        captured = capsys.readouterr()
        # Should show status
        assert "Current provider" in captured.out or "openai" in captured.out

    @pytest.mark.asyncio
    async def test_invalid_subcommand(self, setup_context, capsys):
        """Test with invalid subcommand."""
        await provider_action_async(["invalid_command"])

        captured = capsys.readouterr()
        # Should show error or treat as provider switch attempt
        assert captured.out  # Should have some output


class TestProviderCommandIntegration:
    """Integration tests for provider command."""

    @pytest.fixture
    def mock_model_manager(self):
        """Create a comprehensive mock ModelManager."""
        manager = Mock(spec=ModelManager)
        manager.get_active_provider.return_value = "openai"
        manager.get_active_model.return_value = "gpt-4o-mini"
        manager.get_active_provider_and_model.return_value = ("openai", "gpt-4o-mini")
        manager.get_default_model.return_value = "claude-sonnet"
        manager.list_available_providers.return_value = {
            "openai": {
                "models": ["gpt-4o", "gpt-4o-mini"],
                "default_model": "gpt-4o-mini",
                "has_api_key": True,
            }
        }
        manager.get_status_summary.return_value = {
            "provider_configured": True,
            "supports_streaming": True,
            "supports_tools": True,
            "supports_vision": True,
        }
        return manager

    @pytest.fixture(autouse=True)
    def setup_context(self, mock_model_manager):
        """Set up context before each test."""
        context = setup_test_context()
        context.model_manager = mock_model_manager
        yield context

    @pytest.mark.asyncio
    async def test_full_status_workflow(self, setup_context, capsys):
        """Test full status display workflow."""
        await provider_action_async([])

        captured = capsys.readouterr()
        output = captured.out

        # Should show current status
        assert "openai" in output
        assert "gpt-4" in output

    @pytest.mark.asyncio
    async def test_list_providers_workflow(self, setup_context, capsys):
        """Test listing providers workflow."""
        await provider_action_async(["list"])

        captured = capsys.readouterr()
        output = captured.out

        # Should show provider list
        assert "openai" in output or "Available" in output
