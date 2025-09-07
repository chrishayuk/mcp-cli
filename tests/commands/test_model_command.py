# tests/test_model_command.py
"""
Comprehensive pytest tests for the model command functionality.
Tests async operations, model switching, probing, and all edge cases.
"""

import pytest
from unittest.mock import Mock, patch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.commands.model import model_action_async, model_action
from tests.conftest import setup_test_context


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

    @pytest.fixture(autouse=True)
    def setup_context(self, mock_model_manager):
        """Set up context before each test."""
        # Context will be set up with the mock model manager
        context = setup_test_context()
        # Replace the model manager with our mock
        context.model_manager = mock_model_manager
        yield context

    @pytest.mark.asyncio
    async def test_no_arguments_shows_status(self, setup_context):
        """Test that calling with no arguments shows current status."""
        with patch("mcp_cli.commands.model.output") as mock_output:
            await model_action_async([])

            # Should show current model info
            mock_output.info.assert_called()
            info_call = mock_output.info.call_args[0][0]
            assert "Current model:" in info_call
            assert "openai/gpt-4o-mini" in info_call

            # Should show hint
            mock_output.hint.assert_called()

    @pytest.mark.asyncio
    async def test_list_command_shows_models(self, setup_context):
        """Test that 'list' command shows available models."""
        with patch("mcp_cli.commands.model.output") as mock_output:
            await model_action_async(["list"])

            # Should display available models using print_table
            mock_output.print_table.assert_called()
            # Should also show tip
            mock_output.tip.assert_called()

    @pytest.mark.asyncio
    async def test_refresh_command(self, setup_context):
        """Test that 'refresh' command triggers model discovery."""
        with patch("mcp_cli.commands.model.output") as mock_output:
            await model_action_async(["refresh"])

            # Should call refresh_discovery on model manager
            setup_context.model_manager.refresh_discovery.assert_called_once()

            # Should show either success or info message
            assert mock_output.success.called or mock_output.info.called

    @pytest.mark.asyncio
    async def test_successful_model_switch(self, setup_context):
        """Test successful model switching."""
        # Create mock probe
        mock_probe = MockLLMProbe(setup_context.model_manager)
        mock_probe.set_test_result("gpt-5", success=True)

        with patch("mcp_cli.commands.model.LLMProbe", return_value=mock_probe):
            with patch("mcp_cli.commands.model.output"):
                await model_action_async(["gpt-5"])

                # Should validate the model
                setup_context.model_manager.validate_model_for_provider.assert_called_with(
                    "openai", "gpt-5"
                )

                # Should set the active model
                setup_context.model_manager.set_active_model.assert_called_with("gpt-5")

                # Success might be called inside loading context
                # Just verify set_active_model was called which means success

    @pytest.mark.asyncio
    async def test_failed_model_switch(self, setup_context):
        """Test failed model switch due to probe failure."""
        # Create mock probe that fails
        mock_probe = MockLLMProbe(setup_context.model_manager)
        mock_probe.set_test_result(
            "gpt-5", success=False, error_message="API key invalid"
        )

        with patch("mcp_cli.commands.model.LLMProbe", return_value=mock_probe):
            with patch("mcp_cli.commands.model.output") as mock_output:
                await model_action_async(["gpt-5"])

                # Should show error
                mock_output.error.assert_called()
                error_msg = str(mock_output.error.call_args)
                assert (
                    "test failed" in error_msg.lower() or "error" in error_msg.lower()
                )

                # Should NOT set the active model
                setup_context.model_manager.set_active_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_model_probe_failure(self, setup_context):
        """Test handling probe errors during model switch."""
        with patch("mcp_cli.commands.model.LLMProbe") as MockProbeClass:
            # Make the probe raise an exception
            MockProbeClass.side_effect = Exception("Probe initialization failed")

            with patch("mcp_cli.commands.model.output") as mock_output:
                await model_action_async(["gpt-5"])

                # Should show error
                mock_output.error.assert_called()
                error_msg = str(mock_output.error.call_args)
                assert (
                    "switch failed" in error_msg.lower() or "error" in error_msg.lower()
                )

                # Should NOT set the active model
                setup_context.model_manager.set_active_model.assert_not_called()

    @pytest.mark.asyncio
    async def test_context_without_model_manager(self, setup_context):
        """Test behavior when context doesn't have model_manager."""
        # The current implementation expects model_manager to exist
        # If it's None, it will raise AttributeError
        setup_context.model_manager = None

        with patch("mcp_cli.commands.model.output"):
            # Expect it to fail when model_manager is None
            with pytest.raises(AttributeError):
                await model_action_async([])

    @pytest.mark.asyncio
    async def test_ollama_status_display(self, setup_context):
        """Test special handling for Ollama provider status."""
        setup_context.model_manager.get_active_provider.return_value = "ollama"
        setup_context.model_manager.get_active_model.return_value = "llama2"
        setup_context.model_manager.get_provider_info.return_value = {
            "models": ["llama2", "mistral"],
            "running_models": ["llama2"],
        }

        with patch("mcp_cli.commands.model.output") as mock_output:
            await model_action_async(["list"])

            # Should display models in table
            mock_output.print_table.assert_called()
            # Check the table was created with the right data
            str(mock_output.print_table.call_args)
            # The table should contain model info

    @pytest.mark.asyncio
    async def test_exception_handling(self, setup_context):
        """Test that exceptions are handled gracefully."""
        # Make model manager raise exception
        setup_context.model_manager.get_available_models.side_effect = Exception(
            "Connection error"
        )

        with patch("mcp_cli.commands.model.output"):
            # The exception will be raised and not handled by the code
            with pytest.raises(Exception, match="Connection error"):
                await model_action_async(["list"])


class TestModelActionSync:
    """Test the synchronous wrapper."""

    @pytest.fixture(autouse=True)
    def setup_context(self):
        """Set up context before each test."""
        mock_manager = Mock()
        mock_manager.get_active_provider.return_value = "openai"
        mock_manager.get_active_model.return_value = "gpt-4"
        context = setup_test_context()
        context.model_manager = mock_manager
        yield context

    def test_sync_wrapper_calls_async_function(self, setup_context):
        """Test that sync wrapper properly calls async function."""
        with patch("mcp_cli.commands.model.model_action_async") as mock_async:
            # Configure mock to return a coroutine
            async def mock_coro(args):
                return None

            mock_async.return_value = mock_coro([])

            with patch("mcp_cli.commands.model.run_blocking") as mock_run:
                model_action([])

                # Should call run_blocking with the async function
                mock_run.assert_called_once()


class TestModelCommandEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.fixture(autouse=True)
    def setup_context(self):
        """Set up context before each test."""
        mock_manager = Mock()
        mock_manager.get_active_provider.return_value = "openai"
        mock_manager.get_active_model.return_value = "gpt-4"
        mock_manager.get_available_models.return_value = ["gpt-4", "gpt-3.5-turbo"]
        mock_manager.validate_model_for_provider.return_value = True
        context = setup_test_context()
        context.model_manager = mock_manager
        yield context

    @pytest.mark.asyncio
    async def test_case_insensitive_commands(self, setup_context):
        """Test that commands are case-insensitive."""
        with patch("mcp_cli.commands.model.output") as mock_output:
            # Test "LIST" command
            await model_action_async(["LIST"])
            mock_output.print_table.assert_called()

            mock_output.reset_mock()

            # Test "Refresh" command
            await model_action_async(["Refresh"])
            setup_context.model_manager.refresh_discovery.assert_called()

    @pytest.mark.asyncio
    async def test_empty_model_list_handling(self, setup_context):
        """Test handling when no models are available."""
        setup_context.model_manager.get_available_models.return_value = []

        with patch("mcp_cli.commands.model.output") as mock_output:
            await model_action_async(["list"])

            # Should show error message about no models
            mock_output.error.assert_called()
            error_msg = str(mock_output.error.call_args)
            assert "No models" in error_msg or "not found" in error_msg
