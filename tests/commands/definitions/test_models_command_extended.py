"""Extended tests for the models command definition to improve coverage."""

import pytest
from unittest.mock import patch, MagicMock

from mcp_cli.commands.providers.models import (
    ModelCommand,
    ModelListCommand,
    ModelSetCommand,
    ModelShowCommand,
)
from mcp_cli.commands.base import CommandResult


class TestModelCommand:
    """Test the ModelCommand group."""

    @pytest.fixture
    def command(self):
        """Create a ModelCommand instance."""
        return ModelCommand()

    def test_model_command_properties(self, command):
        """Test model command properties."""
        assert command.name == "models"
        assert command.aliases == ["model"]
        assert command.description == "Manage LLM models"
        assert "Manage LLM models" in command.help_text

        # Check subcommands
        assert "list" in command.subcommands
        assert "set" in command.subcommands
        assert "show" in command.subcommands

    @pytest.mark.asyncio
    async def test_model_command_default_execution(self, command):
        """Test executing model command without subcommand."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.get_active_model.return_value = "gpt-4"
            mock_ctx.model_manager.get_active_provider.return_value = "openai"

            with patch("chuk_term.ui.output"):
                result = await command.execute()

            assert result.success is True


class TestModelListCommand:
    """Test the ModelListCommand subcommand."""

    @pytest.fixture
    def command(self):
        """Create a ModelListCommand instance."""
        return ModelListCommand()

    def test_list_command_properties(self, command):
        """Test list command properties."""
        assert command.name == "list"
        assert command.aliases == ["ls"]
        assert "List available models" in command.description

    @pytest.mark.asyncio
    async def test_list_command_execute(self, command):
        """Test executing list command."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.get_active_provider.return_value = "openai"
            mock_ctx.model_manager.get_active_model.return_value = "gpt-4"

            # Mock model discovery through chuk_llm
            with patch(
                "mcp_cli.commands.providers.models.ModelListCommand._get_provider_models"
            ) as mock_discover:
                mock_discover.return_value = ["gpt-4", "gpt-3.5-turbo"]

                with patch("chuk_term.ui.output"):
                    result = await command.execute()

            assert result.success is True
            mock_ctx.model_manager.get_active_provider.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_command_with_refresh(self, command):
        """Test list command with refresh parameter (refresh not used currently)."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.get_active_provider.return_value = "openai"
            mock_ctx.model_manager.get_active_model.return_value = "gpt-4"

            # Mock model discovery through chuk_llm
            with patch(
                "mcp_cli.commands.providers.models.ModelListCommand._get_provider_models"
            ) as mock_discover:
                mock_discover.return_value = ["gpt-4", "gpt-3.5-turbo"]

                with patch("chuk_term.ui.output"):
                    result = await command.execute(refresh=True)

            assert result.success is True

    @pytest.mark.asyncio
    async def test_list_command_error(self, command):
        """Test list command error handling."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_get_ctx.side_effect = Exception("Failed to list models")

            result = await command.execute()

            assert result.success is False
            assert "Failed to list models" in result.error


class TestModelSetCommand:
    """Test the ModelSetCommand subcommand."""

    @pytest.fixture
    def command(self):
        """Create a ModelSetCommand instance."""
        return ModelSetCommand()

    def test_set_command_properties(self, command):
        """Test set command properties."""
        assert command.name == "set"
        assert command.aliases == ["use", "switch"]
        assert "Set the active model" in command.description
        assert len(command.parameters) > 0

    @pytest.mark.asyncio
    async def test_set_command_execute(self, command):
        """Test executing set command."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.switch_model.return_value = None

            with patch("chuk_term.ui.output"):
                result = await command.execute(model_name="gpt-4")

            assert result.success is True
            mock_ctx.model_manager.switch_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_command_no_model_name(self, command):
        """Test set command without model name."""
        result = await command.execute()

        assert result.success is False
        assert "Model name is required" in result.error

    @pytest.mark.asyncio
    async def test_set_command_from_args(self, command):
        """Test set command with model name from args."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.switch_model.return_value = None

            with patch("chuk_term.ui.output"):
                result = await command.execute(args=["gpt-3.5-turbo"])

            assert result.success is True
            mock_ctx.model_manager.switch_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_command_error(self, command):
        """Test set command error handling."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_get_ctx.side_effect = Exception("Model not found")

            result = await command.execute(model_name="invalid-model")

            assert result.success is False
            assert "Failed to set model" in result.error


class TestModelShowCommand:
    """Test the ModelShowCommand subcommand."""

    @pytest.fixture
    def command(self):
        """Create a ModelShowCommand instance."""
        return ModelShowCommand()

    def test_show_command_properties(self, command):
        """Test show command properties."""
        assert command.name == "show"
        assert command.aliases == ["current", "status"]
        assert (
            "Show" in command.description and "current" in command.description.lower()
        )

    @pytest.mark.asyncio
    async def test_show_command_execute(self, command):
        """Test executing show command."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.get_active_model.return_value = "gpt-4"
            mock_ctx.model_manager.get_active_provider.return_value = "openai"

            with patch("chuk_term.ui.output"):
                result = await command.execute()

            assert result.success is True
            mock_ctx.model_manager.get_active_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_show_command_error(self, command):
        """Test show command error handling."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_get_ctx.side_effect = Exception("Failed to show model")

            result = await command.execute()

            assert result.success is False
            assert "Failed to get model info" in result.error


class TestModelCommandIntegration:
    """Test model command integration scenarios."""

    @pytest.fixture
    def command(self):
        """Create a ModelCommand instance."""
        return ModelCommand()

    @pytest.mark.asyncio
    async def test_execute_with_model_name_directly(self, command):
        """Test executing model command with model name directly."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.switch_model.return_value = None

            with patch("chuk_term.ui.output"):
                # When model name is provided directly
                result = await command.execute(args=["gpt-4o"])

            assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_list_subcommand(self, command):
        """Test executing model list subcommand."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.get_active_provider.return_value = "openai"
            mock_ctx.model_manager.get_active_model.return_value = "gpt-4"

            # Mock model discovery through chuk_llm
            with patch(
                "mcp_cli.commands.providers.models.ModelListCommand._get_provider_models"
            ) as mock_discover:
                mock_discover.return_value = ["gpt-4"]

                with patch("chuk_term.ui.output"):
                    result = await command.execute(subcommand="list")

        # Should delegate to list subcommand
        assert isinstance(result, CommandResult)

    @pytest.mark.asyncio
    async def test_execute_invalid_subcommand(self, command):
        """Test executing with invalid subcommand that gets treated as model name."""
        # The 'invalid' will be treated as model name to switch to
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.switch_model.side_effect = Exception(
                "Model not found: invalid"
            )

            result = await command.execute(args=["invalid"])

            assert result.success is False
            assert "Failed to switch model" in result.error

    @pytest.mark.asyncio
    async def test_execute_known_subcommand_routing(self, command):
        """Test that known subcommands are routed to parent class."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.get_active_provider.return_value = "openai"
            mock_ctx.model_manager.get_active_model.return_value = "gpt-4"

            with patch(
                "mcp_cli.commands.providers.models.ModelListCommand._get_provider_models"
            ) as mock_discover:
                mock_discover.return_value = ["gpt-4"]

                with patch("chuk_term.ui.output"):
                    # Test various known subcommand names
                    for subcmd in ["list", "ls", "set", "show", "current", "status"]:
                        result = await command.execute(args=[subcmd])
                        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_no_context(self, command):
        """Test execution when context is None."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_get_ctx.return_value = None

            result = await command.execute(args=["gpt-4"])

            assert result.success is False
            assert "No LLM manager available" in result.error

    @pytest.mark.asyncio
    async def test_execute_no_model_manager(self, command):
        """Test execution when model_manager is None."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager = None

            result = await command.execute(args=["gpt-4"])

            assert result.success is False
            assert "No LLM manager available" in result.error

    @pytest.mark.asyncio
    async def test_execute_args_as_string(self, command):
        """Test execution with args as a string instead of list."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.switch_model.return_value = None

            with patch("chuk_term.ui.output"):
                result = await command.execute(args="gpt-4o")

            assert result.success is True


class TestModelListCommandExtended:
    """Extended tests for ModelListCommand."""

    @pytest.fixture
    def command(self):
        """Create a ModelListCommand instance."""
        return ModelListCommand()

    @pytest.mark.asyncio
    async def test_list_no_context(self, command):
        """Test list command when context is None."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_get_ctx.return_value = None

            result = await command.execute()

            assert result.success is False
            assert "No LLM manager available" in result.error

    @pytest.mark.asyncio
    async def test_list_no_model_manager(self, command):
        """Test list command when model_manager is None."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager = None

            result = await command.execute()

            assert result.success is False
            assert "No LLM manager available" in result.error

    @pytest.mark.asyncio
    async def test_list_ollama_provider(self, command):
        """Test list command with Ollama provider."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.get_active_provider.return_value = "ollama"
            mock_ctx.model_manager.get_active_model.return_value = "llama2"

            with patch.object(
                command, "_get_ollama_models", return_value=["llama2", "codellama"]
            ):
                with patch("chuk_term.ui.output"):
                    with patch("chuk_term.ui.format_table"):
                        result = await command.execute()

            assert result.success is True

    @pytest.mark.asyncio
    async def test_list_no_models_found(self, command):
        """Test list command when no models are discovered."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.get_active_provider.return_value = "openai"
            mock_ctx.model_manager.get_active_model.return_value = "gpt-4"

            with patch.object(command, "_get_provider_models", return_value=[]):
                with patch("chuk_term.ui.output"):
                    result = await command.execute()

            assert result.success is True

    def test_get_ollama_models_success(self, command):
        """Test _get_ollama_models with successful ollama list."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "NAME\nllama2\ncodellama\nmistral"

            models = command._get_ollama_models()

            assert models == ["llama2", "codellama", "mistral"]

    def test_get_ollama_models_failure(self, command):
        """Test _get_ollama_models when ollama list fails."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1

            models = command._get_ollama_models()

            assert models == []

    def test_get_ollama_models_exception(self, command):
        """Test _get_ollama_models when exception occurs."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Command not found")

            models = command._get_ollama_models()

            assert models == []

    def test_get_provider_models_success(self, command):
        """Test _get_provider_models with successful discovery."""
        with patch("chuk_llm.llm.client.list_available_providers") as mock_list:
            mock_list.return_value = {"openai": {"models": ["gpt-4", "gpt-3.5-turbo"]}}

            models = command._get_provider_models("openai")

            assert models == ["gpt-4", "gpt-3.5-turbo"]

    def test_get_provider_models_available_models_key(self, command):
        """Test _get_provider_models with available_models key."""
        with patch("chuk_llm.llm.client.list_available_providers") as mock_list:
            mock_list.return_value = {
                "openai": {"available_models": ["gpt-4", "gpt-3.5-turbo"]}
            }

            models = command._get_provider_models("openai")

            assert models == ["gpt-4", "gpt-3.5-turbo"]

    def test_get_provider_models_exception(self, command):
        """Test _get_provider_models when exception occurs."""
        with patch("chuk_llm.llm.client.list_available_providers") as mock_list:
            mock_list.side_effect = Exception("Import error")

            models = command._get_provider_models("openai")

            assert models == []

    def test_get_provider_models_unknown_provider(self, command):
        """Test _get_provider_models with unknown provider."""
        with patch("chuk_llm.llm.client.list_available_providers") as mock_list:
            mock_list.return_value = {"openai": {"models": ["gpt-4"]}}

            models = command._get_provider_models("unknown")

            assert models == []

    def test_get_provider_models_placeholder_with_default(self, command):
        """Test _get_provider_models when models is ['*'] but default_model exists."""
        with patch("chuk_llm.llm.client.list_available_providers") as mock_list:
            # No API key, so API won't be called - falls back to default_model
            mock_list.return_value = {
                "deepseek": {
                    "models": ["*"],
                    "default_model": "deepseek-chat",
                    "has_api_key": False,
                }
            }

            models = command._get_provider_models("deepseek")

            assert models == ["deepseek-chat"]

    def test_get_provider_models_placeholder_no_default(self, command):
        """Test _get_provider_models when models is ['*'] and no default_model."""
        with patch("chuk_llm.llm.client.list_available_providers") as mock_list:
            mock_list.return_value = {"unknown": {"models": ["*"]}}

            models = command._get_provider_models("unknown")

            assert models == []

    def test_get_provider_models_calls_api_on_placeholder(self, command):
        """Test _get_provider_models calls API when models is ['*'] and has API key."""
        with patch("chuk_llm.llm.client.list_available_providers") as mock_list:
            mock_list.return_value = {
                "deepseek": {
                    "models": ["*"],
                    "default_model": "deepseek-chat",
                    "has_api_key": True,
                    "api_base": "https://api.deepseek.com/v1",
                }
            }
            # Mock the API call
            with patch.object(
                command,
                "_fetch_models_from_api",
                return_value=["deepseek-chat", "deepseek-reasoner"],
            ):
                models = command._get_provider_models("deepseek")

            assert models == ["deepseek-chat", "deepseek-reasoner"]

    def test_fetch_models_from_api_success(self, command):
        """Test _fetch_models_from_api with successful API response."""
        import os

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            with patch("httpx.get") as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {
                    "data": [
                        {"id": "deepseek-chat"},
                        {"id": "deepseek-reasoner"},
                    ]
                }
                mock_get.return_value = mock_response

                models = command._fetch_models_from_api(
                    "deepseek", "https://api.deepseek.com/v1"
                )

            assert models == ["deepseek-chat", "deepseek-reasoner"]

    def test_fetch_models_from_api_no_api_key(self, command):
        """Test _fetch_models_from_api returns empty when no API key."""
        import os

        with patch.dict(os.environ, {}, clear=True):
            # Ensure no DEEPSEEK_API_KEY
            if "DEEPSEEK_API_KEY" in os.environ:
                del os.environ["DEEPSEEK_API_KEY"]

            models = command._fetch_models_from_api(
                "deepseek", "https://api.deepseek.com/v1"
            )

        assert models == []

    def test_fetch_models_from_api_error(self, command):
        """Test _fetch_models_from_api handles errors gracefully."""
        import os

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            with patch("httpx.get") as mock_get:
                mock_get.side_effect = Exception("Network error")

                models = command._fetch_models_from_api(
                    "deepseek", "https://api.deepseek.com/v1"
                )

            assert models == []

    def test_fetch_models_from_api_non_200(self, command):
        """Test _fetch_models_from_api handles non-200 status."""
        import os

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            with patch("httpx.get") as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 401
                mock_get.return_value = mock_response

                models = command._fetch_models_from_api(
                    "deepseek", "https://api.deepseek.com/v1"
                )

            assert models == []


class TestModelSetCommandExtended:
    """Extended tests for ModelSetCommand."""

    @pytest.fixture
    def command(self):
        """Create a ModelSetCommand instance."""
        return ModelSetCommand()

    @pytest.mark.asyncio
    async def test_set_args_as_string(self, command):
        """Test set command with args as string."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager.switch_model.return_value = None

            with patch("chuk_term.ui.output"):
                result = await command.execute(args="gpt-4o")

            assert result.success is True

    @pytest.mark.asyncio
    async def test_set_no_context(self, command):
        """Test set command when context is None."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_get_ctx.return_value = None

            result = await command.execute(model_name="gpt-4")

            assert result.success is False
            assert "No LLM manager available" in result.error

    @pytest.mark.asyncio
    async def test_set_no_model_manager(self, command):
        """Test set command when model_manager is None."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager = None

            result = await command.execute(model_name="gpt-4")

            assert result.success is False
            assert "No LLM manager available" in result.error


class TestModelShowCommandExtended:
    """Extended tests for ModelShowCommand."""

    @pytest.fixture
    def command(self):
        """Create a ModelShowCommand instance."""
        return ModelShowCommand()

    @pytest.mark.asyncio
    async def test_show_no_context(self, command):
        """Test show command when context is None."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_get_ctx.return_value = None

            result = await command.execute()

            assert result.success is False
            assert "No LLM manager available" in result.error

    @pytest.mark.asyncio
    async def test_show_no_model_manager(self, command):
        """Test show command when model_manager is None."""
        with patch("mcp_cli.context.get_context") as mock_get_ctx:
            mock_ctx = mock_get_ctx.return_value
            mock_ctx.model_manager = None

            result = await command.execute()

            assert result.success is False
            assert "No LLM manager available" in result.error
