"""Edge case tests to push providers.py coverage to 90%+."""

import pytest
from unittest.mock import patch, MagicMock
from mcp_cli.commands.actions.providers import (
    _render_list_optimized,
    _render_diagnostic_optimized,
    _switch_provider_enhanced,
    provider_action_async,
)
from mcp_cli.commands.models import ProviderActionParams


# ========== Render List Error Paths ==========


def test_render_list_token_manager_exception():
    """Test render list when TokenManager throws exception."""
    mock_manager = MagicMock()
    mock_manager.get_active_provider.return_value = "anthropic"
    mock_manager.get_available_providers.return_value = ["anthropic"]
    mock_manager.get_available_models.return_value = ["claude-3"]
    mock_manager.get_default_model.return_value = "claude-3"

    with patch("mcp_cli.auth.TokenManager", side_effect=Exception("Token error")):
        with patch("mcp_cli.commands.actions.providers.output"):
            with patch("mcp_cli.commands.actions.providers.format_table"):
                # Should not raise exception, token_manager should be None
                _render_list_optimized(mock_manager)


def test_render_list_no_providers():
    """Test render list when no providers found."""
    mock_manager = MagicMock()
    mock_manager.get_available_providers.return_value = []

    with patch("mcp_cli.commands.actions.providers.output") as mock_output:
        _render_list_optimized(mock_manager)

        mock_output.error.assert_called_with(
            "No providers found. Check chuk-llm installation."
        )


def test_render_list_provider_get_models_exception():
    """Test render list when get_available_models throws exception for a provider."""
    mock_manager = MagicMock()
    mock_manager.get_active_provider.return_value = "anthropic"
    mock_manager.get_available_providers.return_value = ["anthropic", "error-provider"]

    def get_models_side_effect(provider):
        if provider == "error-provider":
            raise Exception("Model fetch error")
        return ["claude-3"]

    mock_manager.get_available_models.side_effect = get_models_side_effect
    mock_manager.get_default_model.return_value = "claude-3"

    with patch("mcp_cli.auth.TokenManager"):
        with patch("mcp_cli.commands.actions.providers.output"):
            with patch("mcp_cli.commands.actions.providers.format_table") as mock_table:
                _render_list_optimized(mock_manager)

                # Should create table with error entry for error-provider
                mock_table.assert_called_once()
                table_data = mock_table.call_args[0][0]
                # Check that error provider is in the table
                error_entries = [
                    row
                    for row in table_data
                    if row.get("Provider")
                    and "error" in row["Provider"].lower()
                    or "Error" in row.get("Status", "")
                ]
                assert len(error_entries) > 0


def test_render_list_get_providers_exception():
    """Test render list when get_available_providers throws exception."""
    mock_manager = MagicMock()
    mock_manager.get_available_providers.side_effect = Exception("Provider list error")

    with patch("mcp_cli.commands.actions.providers.output") as mock_output:
        _render_list_optimized(mock_manager)

        mock_output.error.assert_called()
        assert "Error getting provider list" in str(mock_output.error.call_args)


def test_render_list_with_inactive_providers():
    """Test render list with inactive providers showing hints."""
    mock_manager = MagicMock()
    mock_manager.get_active_provider.return_value = "ollama"
    mock_manager.get_available_providers.return_value = ["ollama", "anthropic"]

    # Ollama has models, anthropic doesn't
    def get_models_side_effect(provider):
        if provider == "ollama":
            return ["llama2"]
        return []

    mock_manager.get_available_models.side_effect = get_models_side_effect

    def get_default_side_effect(provider):
        if provider == "ollama":
            return "llama2"
        return None

    mock_manager.get_default_model.side_effect = get_default_side_effect

    with patch("mcp_cli.auth.TokenManager"):
        with patch(
            "mcp_cli.commands.actions.providers._get_provider_status_enhanced"
        ) as mock_status:
            # Ollama ready, anthropic not ready
            def status_side_effect(name, data):
                from mcp_cli.commands.models.provider import ProviderStatus

                if name == "ollama":
                    return ProviderStatus(icon="✅", text="Ready", reason="Running")
                return ProviderStatus(
                    icon="❌", text="Not Configured", reason="No API key"
                )

            mock_status.side_effect = status_side_effect

            with patch("mcp_cli.commands.actions.providers.output") as mock_output:
                with patch("mcp_cli.commands.actions.providers.format_table"):
                    _render_list_optimized(mock_manager)

                    # Should show hint for inactive provider
                    assert any(
                        "API key" in str(call) or "ANTHROPIC" in str(call)
                        for call in mock_output.hint.call_args_list
                    )


def test_render_list_with_custom_providers_hint():
    """Test render list showing custom provider hint when none exist."""
    mock_manager = MagicMock()
    mock_manager.get_active_provider.return_value = "anthropic"
    mock_manager.get_available_providers.return_value = ["anthropic"]
    mock_manager.get_available_models.return_value = ["claude-3"]
    mock_manager.get_default_model.return_value = "claude-3"

    with patch("mcp_cli.auth.TokenManager"):
        with patch("mcp_cli.commands.actions.providers.output") as mock_output:
            with patch("mcp_cli.commands.actions.providers.format_table"):
                _render_list_optimized(mock_manager)

                # Should show hint about adding custom providers (lines 349-352)
                assert any(
                    "custom" in str(call).lower() or "add" in str(call).lower()
                    for call in mock_output.hint.call_args_list
                )


# ========== Render Diagnostic Error Paths ==========


def test_render_diagnostic_provider_exception():
    """Test render diagnostic when processing a provider throws exception."""
    mock_manager = MagicMock()
    mock_manager.validate_provider.return_value = True
    mock_manager.get_available_providers.return_value = {"test": {}}

    with patch(
        "mcp_cli.commands.actions.providers._dict_to_provider_data",
        side_effect=Exception("Parse error"),
    ):
        with patch("mcp_cli.commands.actions.providers.output"):
            with patch("mcp_cli.commands.actions.providers.format_table") as mock_table:
                _render_diagnostic_optimized(mock_manager, "test")

                # Should create table with error entry
                mock_table.assert_called_once()


def test_render_diagnostic_unknown_provider():
    """Test render diagnostic with unknown provider."""
    mock_manager = MagicMock()
    mock_manager.validate_provider.return_value = False
    mock_manager.get_available_providers.return_value = ["anthropic", "openai"]

    with patch("mcp_cli.commands.actions.providers.output") as mock_output:
        _render_diagnostic_optimized(mock_manager, "unknown")

        mock_output.error.assert_called_with("Unknown provider: unknown")
        mock_output.warning.assert_called()


# ========== Switch Provider Edge Cases ==========


def test_switch_provider_get_available_models_exception():
    """Test switch provider when get_available_models throws exception."""
    mock_manager = MagicMock()
    mock_manager.validate_provider.return_value = True
    mock_manager.get_available_providers.return_value = {
        "anthropic": {"has_api_key": True, "models": []}
    }
    mock_manager.get_default_model.return_value = None
    mock_manager.get_available_models.side_effect = Exception("Models error")
    mock_context = MagicMock()

    with patch("mcp_cli.commands.actions.providers.output"):
        _switch_provider_enhanced(mock_manager, "anthropic", None, mock_context)

        # Should fall back to "default"
        mock_manager.switch_model.assert_called_with("anthropic", "default")


def test_switch_provider_partial_setup_warning():
    """Test switch provider with partial setup warning."""
    mock_manager = MagicMock()
    mock_manager.validate_provider.return_value = True
    mock_manager.get_available_providers.return_value = {
        "anthropic": {"has_api_key": True, "models": []}
    }
    mock_manager.get_default_model.return_value = "claude-3"
    mock_context = MagicMock()

    with patch(
        "mcp_cli.commands.actions.providers._get_provider_status_enhanced"
    ) as mock_status:
        from mcp_cli.commands.models.provider import ProviderStatus

        mock_status.return_value = ProviderStatus(
            icon="⚠️",
            text="Partial Setup",
            reason="API key set but no models",
        )

        with patch("mcp_cli.commands.actions.providers.output") as mock_output:
            _switch_provider_enhanced(mock_manager, "anthropic", None, mock_context)

            # Should show warning but continue
            mock_output.warning.assert_called()
            # Check that "Continuing" or "Switching" message was shown
            assert any(
                "Continuing" in str(call) or "Switching" in str(call)
                for call in mock_output.info.call_args_list
            )
            mock_manager.switch_model.assert_called()


# ========== Provider Action Async Edge Cases ==========


@pytest.mark.asyncio
async def test_provider_action_no_model_manager():
    """Test provider action when model manager is not available."""
    mock_context = MagicMock()
    mock_context.model_manager = None

    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch("mcp_cli.commands.actions.providers.output") as mock_output:
            await provider_action_async(ProviderActionParams(args=["list"]))

            mock_output.error.assert_called_with("Model manager not available")


@pytest.mark.asyncio
async def test_provider_action_switch_with_model():
    """Test provider action switching with specific model."""
    mock_context = MagicMock()
    mock_context.model_manager = MagicMock()

    with patch(
        "mcp_cli.commands.actions.providers.get_context", return_value=mock_context
    ):
        with patch(
            "mcp_cli.commands.actions.providers._switch_provider_enhanced"
        ) as mock_switch:
            await provider_action_async(
                ProviderActionParams(args=["anthropic", "claude-3"])
            )

            mock_switch.assert_called_once_with(
                mock_context.model_manager, "anthropic", "claude-3", mock_context
            )


# ========== Dict to ProviderData Conversion ==========


def test_dict_to_provider_data_with_all_fields():
    """Test converting dict to ProviderData with all fields."""
    from mcp_cli.commands.actions.providers import _dict_to_provider_data

    data = {
        "has_api_key": True,
        "token_source": "env",
        "models": ["model1", "model2"],
        "available_models": ["model3"],
        "default_model": "model1",
        "baseline_features": ["streaming", "tools"],
        "is_custom": True,
        "api_base": "http://localhost",
        "discovery_enabled": True,
        "error": None,
    }

    result = _dict_to_provider_data("test", data)

    assert result.name == "test"
    assert result.has_api_key is True
    assert result.models == ["model1", "model2"]
    assert result.default_model == "model1"


def test_dict_to_provider_data_minimal():
    """Test converting minimal dict to ProviderData."""
    from mcp_cli.commands.actions.providers import _dict_to_provider_data

    data = {}
    result = _dict_to_provider_data("test", data)

    assert result.name == "test"
    assert result.has_api_key is False
    assert result.models == []


def test_dict_to_provider_data_with_provider_data():
    """Test that ProviderData is returned as-is."""
    from mcp_cli.commands.actions.providers import _dict_to_provider_data
    from mcp_cli.commands.models.provider import ProviderData

    provider_data = ProviderData(name="test")
    result = _dict_to_provider_data("test", provider_data)

    assert result is provider_data


# ========== Additional Edge Cases ==========


def test_render_list_inactive_custom_provider():
    """Test render list with inactive custom provider."""
    mock_manager = MagicMock()
    mock_manager.get_active_provider.return_value = "anthropic"
    mock_manager.get_available_providers.return_value = ["anthropic", "custom"]

    def get_models_side_effect(provider):
        if provider == "anthropic":
            return ["claude-3"]
        return []  # Custom provider not configured

    mock_manager.get_available_models.side_effect = get_models_side_effect
    mock_manager.get_default_model.side_effect = (
        lambda p: "claude-3" if p == "anthropic" else None
    )

    with patch("mcp_cli.auth.TokenManager"):
        with patch(
            "mcp_cli.commands.actions.providers._get_provider_status_enhanced"
        ) as mock_status:
            from mcp_cli.commands.models.provider import ProviderStatus

            def status_side_effect(name, data):
                # Mark custom as not ready and custom
                if name == "custom":
                    # Simulate is_custom being True
                    data.is_custom = True
                    return ProviderStatus(
                        icon="❌", text="Not Ready", reason="No config"
                    )
                return ProviderStatus(icon="✅", text="Ready", reason="OK")

            mock_status.side_effect = status_side_effect

            with patch("mcp_cli.commands.actions.providers.output") as mock_output:
                with patch("mcp_cli.commands.actions.providers.format_table"):
                    _render_list_optimized(mock_manager)

                    # Should show hints for inactive custom provider
                    assert mock_output.hint.called


def test_switch_provider_with_provider_error_key():
    """Test switch provider when provider has error key."""
    mock_manager = MagicMock()
    mock_manager.validate_provider.return_value = True
    mock_manager.get_available_providers.return_value = {
        "test": {"error": "Provider is broken"}
    }
    mock_context = MagicMock()

    with patch("mcp_cli.commands.actions.providers.output") as mock_output:
        _switch_provider_enhanced(mock_manager, "test", None, mock_context)

        mock_output.error.assert_called_with("Provider error: Provider is broken")


# ========== Stub Functions ==========


def test_render_config_stub():
    """Test _render_config stub function."""
    from mcp_cli.commands.actions.providers import _render_config

    mock_manager = MagicMock()
    # Should not raise exception, just pass
    _render_config(mock_manager)


def test_mutate_stub():
    """Test _mutate stub function."""
    from mcp_cli.commands.actions.providers import _mutate

    mock_manager = MagicMock()
    # Should not raise exception, just pass
    _mutate(mock_manager, "anthropic", "api_key", "test")
