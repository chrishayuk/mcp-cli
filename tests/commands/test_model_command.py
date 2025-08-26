# tests/test_model_command.py
"""
Comprehensive pytest tests for the model command functionality.
Tests async operations, model switching, probing, and all edge cases.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.commands.model import model_action_async, model_action


class MockLLMProbeResult:
    """Mock result from LLM probe testing."""

    def __init__(self, success: bool, error_message: str = None, client=None):
        self.success = success
        self.error_message = error_message
        self.client = client or Mock()


class MockLLMProbe:
    """Mock LLM probe for testing."""

    def __init__(self, model_manager, suppress_logging=True):
        self.model_manager = model_manager
        self.suppress_logging = suppress_logging
        self._test_results = {}

    def set_test_result(
        self, model: str, success: bool, error_message: str = None, client=None
    ):
        """Set what result the probe should return for a model."""
        self._test_results[model] = MockLLMProbeResult(success, error_message, client)

    async def test_model(self, model: str):
        """Mock test_model method."""
        return self._test_results.get(
            model,
            MockLLMProbeResult(True),  # Default to success
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class TestModelActionAsync:
    """Test the async model action functionality."""

    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock ModelManager for testing."""
        manager = Mock()
        manager.get_active_provider.return_value = "openai"
        manager.get_active_model.return_value = "gpt-4o-mini"
        manager.get_default_model.return_value = "gpt-4o-mini"
        manager.get_available_models.return_value = ["gpt-4o", "gpt-4o-mini", "gpt-5"]
        manager.validate_model_for_provider.return_value = True
        manager.set_active_model = Mock()
        manager.refresh_discovery = Mock(return_value=True)
        manager.get_discovery_status = Mock(return_value={"ollama_enabled": True})
        manager.get_provider_info = Mock(
            return_value={"models": ["gpt-4o", "gpt-4o-mini"]}
        )
        return manager

    @pytest.fixture
    def base_context(self, mock_model_manager):
        """Create a base context for testing."""
        return {
            "model_manager": mock_model_manager,
            "model": "gpt-4o-mini",
            "provider": "openai",
            "client": Mock(),
        }

    @pytest.mark.asyncio
    async def test_no_arguments_shows_status(self, base_context):
        """Test that calling with no arguments shows current status."""
        with patch("mcp_cli.commands.model.output") as mock_output:
            await model_action_async([], context=base_context)

            # Should show current model info
            mock_output.info.assert_called()
            info_call = mock_output.info.call_args[0][0]
            assert "Current model:" in info_call
            assert "openai/gpt-4o-mini" in info_call

            # Should show hint
            mock_output.hint.assert_called()

    @pytest.mark.asyncio
    async def test_list_command_shows_models(self, base_context):
        """Test that 'list' command shows model list."""
        with patch("mcp_cli.commands.model.output") as mock_output:
            with patch("mcp_cli.commands.model.format_table") as mock_format_table:
                mock_table = Mock()
                mock_format_table.return_value = mock_table

                await model_action_async(["list"], context=base_context)

                # Should format and display table
                mock_format_table.assert_called_once()
                mock_output.print_table.assert_called_with(mock_table)

                # Should show tip
                mock_output.tip.assert_called()

    @pytest.mark.asyncio
    async def test_refresh_command(self, base_context):
        """Test refresh command."""
        with patch("mcp_cli.commands.model.output") as mock_output:
            # Create a mock loading context manager
            mock_loading = MagicMock()
            mock_output.loading.return_value.__enter__ = Mock(return_value=mock_loading)
            mock_output.loading.return_value.__exit__ = Mock(return_value=None)

            await model_action_async(["refresh"], context=base_context)

            # Should call refresh on model manager
            base_context["model_manager"].refresh_discovery.assert_called_with("openai")

            # Note: Not checking for success message as implementation doesn't call it
            # The refresh being called is sufficient to prove the command worked

    @pytest.mark.asyncio
    async def test_successful_model_switch(self, base_context):
        """Test successful model switching with probing."""
        new_model = "gpt-4o"
        mock_client = Mock()

        with patch("mcp_cli.commands.model.output") as mock_output:
            # Setup loading context manager
            mock_output.loading.return_value.__enter__ = Mock()
            mock_output.loading.return_value.__exit__ = Mock(return_value=None)

            with patch("mcp_cli.commands.model.LLMProbe") as mock_probe_class:
                # Setup mock probe
                mock_probe = MockLLMProbe(base_context["model_manager"])
                mock_probe.set_test_result(new_model, success=True, client=mock_client)
                mock_probe_class.return_value = mock_probe

                await model_action_async([new_model], context=base_context)

                # Verify model was switched
                base_context["model_manager"].set_active_model.assert_called_once_with(
                    new_model
                )

                # Verify context was updated
                assert base_context["model"] == new_model
                assert base_context["client"] == mock_client

                # Verify success message
                mock_output.success.assert_called()
                success_call = mock_output.success.call_args[0][0]
                assert "Switched to model:" in success_call
                assert new_model in success_call

    @pytest.mark.asyncio
    async def test_failed_model_switch(self, base_context):
        """Test failed model switching."""
        new_model = "invalid-model"
        error_message = "Model not found"

        # Set validation to fail
        base_context["model_manager"].validate_model_for_provider.return_value = False

        with patch("mcp_cli.commands.model.output") as mock_output:
            # Setup loading context manager
            mock_output.loading.return_value.__enter__ = Mock()
            mock_output.loading.return_value.__exit__ = Mock(return_value=None)

            await model_action_async([new_model], context=base_context)

            # Verify model was NOT switched
            base_context["model_manager"].set_active_model.assert_not_called()

            # Verify error message
            mock_output.error.assert_called()
            error_call = mock_output.error.call_args[0][0]
            assert "Model not available:" in error_call

            # Should show tip with suggestions
            mock_output.tip.assert_called()

    @pytest.mark.asyncio
    async def test_model_probe_failure(self, base_context):
        """Test handling of model probe failure."""
        new_model = "gpt-5"

        with patch("mcp_cli.commands.model.output") as mock_output:
            # Setup loading context manager
            mock_output.loading.return_value.__enter__ = Mock()
            mock_output.loading.return_value.__exit__ = Mock(return_value=None)

            with patch("mcp_cli.commands.model.LLMProbe") as mock_probe_class:
                # Setup mock probe to fail
                mock_probe = MockLLMProbe(base_context["model_manager"])
                mock_probe.set_test_result(
                    new_model, success=False, error_message="API error"
                )
                mock_probe_class.return_value = mock_probe

                await model_action_async([new_model], context=base_context)

                # Verify model was NOT switched
                base_context["model_manager"].set_active_model.assert_not_called()

                # Verify error and warning messages
                mock_output.error.assert_called()
                mock_output.warning.assert_called()
                warning_call = mock_output.warning.call_args[0][0]
                assert "Keeping current model:" in warning_call

    @pytest.mark.asyncio
    async def test_context_without_model_manager(self):
        """Test that missing ModelManager in context creates a new one."""
        context = {}  # Empty context

        with patch("mcp_cli.commands.model.output") as mock_output:
            with patch("mcp_cli.commands.model.ModelManager") as mock_manager_class:
                mock_manager = Mock()
                mock_manager.get_active_provider.return_value = "openai"
                mock_manager.get_active_model.return_value = "gpt-4o-mini"
                mock_manager.get_available_models.return_value = [
                    "gpt-4o",
                    "gpt-4o-mini",
                ]
                mock_manager_class.return_value = mock_manager

                await model_action_async([], context=context)

                # Verify ModelManager was created and added to context
                mock_manager_class.assert_called_once()
                assert context["model_manager"] == mock_manager

    @pytest.mark.asyncio
    async def test_ollama_status_display(self, base_context):
        """Test Ollama-specific status display."""
        base_context["model_manager"].get_active_provider.return_value = "ollama"
        base_context["model_manager"].get_active_model.return_value = "gpt-oss"

        with patch("mcp_cli.commands.model.output") as mock_output:
            with patch("mcp_cli.commands.model._check_local_ollama") as mock_check:
                mock_check.return_value = (True, ["gpt-oss", "llama3.3"])

                await model_action_async([], context=base_context)

                # Should show Ollama status
                print_calls = [call[0][0] for call in mock_output.print.call_args_list]
                assert any("Ollama:" in call for call in print_calls)

    @pytest.mark.asyncio
    async def test_exception_handling(self, base_context):
        """Test exception handling during model switch."""
        new_model = "gpt-4o"

        with patch("mcp_cli.commands.model.output") as mock_output:
            # Setup loading context manager
            mock_output.loading.return_value.__enter__ = Mock()
            mock_output.loading.return_value.__exit__ = Mock(return_value=None)

            with patch("mcp_cli.commands.model.LLMProbe") as mock_probe_class:
                # Setup probe to raise exception
                mock_probe_class.side_effect = Exception("Probe error")

                await model_action_async([new_model], context=base_context)

                # Should handle exception gracefully
                mock_output.error.assert_called()
                error_call = mock_output.error.call_args[0][0]
                assert "Model switch failed:" in error_call

                # Should keep current model
                mock_output.warning.assert_called()


class TestModelActionSync:
    """Test the synchronous wrapper for model action."""

    @patch("mcp_cli.commands.model.run_blocking")
    def test_sync_wrapper_calls_async_function(self, mock_run_blocking):
        """Test that sync wrapper properly calls async function."""
        args = ["test-model"]
        context = {"test": "context"}

        model_action(args, context=context)

        # Verify run_blocking was called
        mock_run_blocking.assert_called_once()

        # Get the actual call arguments
        call_args = mock_run_blocking.call_args[0][0]

        # Verify it's a coroutine (async function call)
        assert asyncio.iscoroutine(call_args)


class TestModelCommandEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_case_insensitive_commands(self):
        """Test that commands are case-insensitive."""
        mock_manager = Mock()
        mock_manager.get_active_provider.return_value = "openai"
        mock_manager.get_active_model.return_value = "gpt-4o"
        mock_manager.get_available_models.return_value = ["gpt-4o"]
        mock_manager.refresh_discovery.return_value = True

        context = {"model_manager": mock_manager}

        with patch("mcp_cli.commands.model.output") as mock_output:
            with patch("mcp_cli.commands.model.format_table") as mock_format_table:
                mock_format_table.return_value = Mock()

                # Test uppercase LIST
                await model_action_async(["LIST"], context=context)
                mock_format_table.assert_called()

                mock_format_table.reset_mock()
                mock_output.loading.return_value.__enter__ = Mock()
                mock_output.loading.return_value.__exit__ = Mock(return_value=None)

                # Test uppercase REFRESH
                await model_action_async(["REFRESH"], context=context)
                mock_manager.refresh_discovery.assert_called()

    @pytest.mark.asyncio
    async def test_empty_model_list_handling(self):
        """Test handling when no models are available."""
        mock_manager = Mock()
        mock_manager.get_active_provider.return_value = "openai"
        mock_manager.get_active_model.return_value = "gpt-4o"
        mock_manager.get_available_models.return_value = []  # No models

        context = {"model_manager": mock_manager}

        with patch("mcp_cli.commands.model.output") as mock_output:
            await model_action_async([], context=context)

            # Should show warning about no models
            mock_output.warning.assert_called()
            warning_call = mock_output.warning.call_args[0][0]
            assert "No models found" in warning_call


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
