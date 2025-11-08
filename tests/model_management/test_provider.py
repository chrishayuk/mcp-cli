# tests/model_management/test_provider.py
"""
Comprehensive tests for provider.py Pydantic models.
Target: >90% code coverage
"""

import pytest
from pydantic import ValidationError

from mcp_cli.model_management.provider import (
    RuntimeProviderConfig,
    ProviderCapabilities,
)


class TestRuntimeProviderConfig:
    """Test RuntimeProviderConfig Pydantic model."""

    def test_basic_creation(self):
        """Test creating a basic RuntimeProviderConfig."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["model-1", "model-2"],
        )

        assert config.name == "test-provider"
        assert config.api_base == "https://api.test.com/v1"
        assert config.models == ["model-1", "model-2"]
        assert config.is_runtime is True
        assert config.api_key is None

    def test_with_api_key(self):
        """Test creating config with API key."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            api_key="sk-test-key-123",
            models=["model-1"],
        )

        assert config.api_key == "sk-test-key-123"

    def test_default_model_auto_set(self):
        """Test that default_model is automatically set to first model."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["model-1", "model-2", "model-3"],
        )

        # Should auto-set to first model
        assert config.default_model == "model-1"

    def test_default_model_explicit(self):
        """Test explicitly setting default_model."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["model-1", "model-2"],
            default_model="model-2",
        )

        assert config.default_model == "model-2"

    def test_empty_models_list(self):
        """Test creating config with empty models list."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=[],
        )

        assert config.models == []
        assert config.default_model is None
        assert config.has_models is False

    def test_no_models_provided(self):
        """Test creating config without models field."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
        )

        assert config.models == []
        assert config.default_model is None

    def test_has_models_property_true(self):
        """Test has_models property when models exist."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["model-1"],
        )

        assert config.has_models is True

    def test_has_models_property_false(self):
        """Test has_models property when models list is empty."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=[],
        )

        assert config.has_models is False

    def test_add_models_method(self):
        """Test add_models method."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["model-1"],
        )

        config.add_models(["model-2", "model-3"])

        assert len(config.models) == 3
        assert "model-2" in config.models
        assert "model-3" in config.models

    def test_add_models_sets_default_if_none(self):
        """Test that add_models sets default_model if not set."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=[],
        )

        assert config.default_model is None

        config.add_models(["model-1", "model-2"])

        assert config.default_model == "model-1"

    def test_add_models_preserves_default(self):
        """Test that add_models preserves existing default_model."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["model-1"],
            default_model="model-1",
        )

        config.add_models(["model-2"])

        assert config.default_model == "model-1"  # Should not change

    def test_set_models_method(self):
        """Test set_models method replaces all models."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["old-model-1", "old-model-2"],
        )

        config.set_models(["new-model-1", "new-model-2", "new-model-3"])

        assert config.models == ["new-model-1", "new-model-2", "new-model-3"]
        assert len(config.models) == 3
        assert "old-model-1" not in config.models

    def test_set_models_preserves_existing_default(self):
        """Test that set_models preserves existing default_model."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["old-model"],
            default_model="old-model",
        )

        config.set_models(["new-model-1", "new-model-2"])

        # Should preserve existing default_model
        assert config.default_model == "old-model"
        assert config.models == ["new-model-1", "new-model-2"]

    def test_set_models_empty_list(self):
        """Test set_models with empty list."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["model-1"],
        )

        config.set_models([])

        assert config.models == []
        # default_model might be None or unchanged depending on implementation
        assert config.has_models is False

    def test_is_runtime_default_true(self):
        """Test that is_runtime defaults to True."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
        )

        assert config.is_runtime is True

    def test_is_runtime_can_be_false(self):
        """Test setting is_runtime to False (for persisted providers)."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            is_runtime=False,
        )

        assert config.is_runtime is False

    def test_model_is_mutable(self):
        """Test that RuntimeProviderConfig is mutable (not frozen)."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["model-1"],
        )

        # Should be able to modify
        config.models.append("model-2")
        assert len(config.models) == 2

    def test_required_fields_validation(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError) as exc_info:
            RuntimeProviderConfig()

        errors = exc_info.value.errors()
        field_names = {error["loc"][0] for error in errors}
        assert "name" in field_names
        assert "api_base" in field_names

    def test_models_field_type_validation(self):
        """Test that models field only accepts list of strings."""
        # Valid: list of strings
        config = RuntimeProviderConfig(
            name="test",
            api_base="https://api.test.com/v1",
            models=["m1", "m2"],
        )
        assert config.models == ["m1", "m2"]

        # Invalid: not a list
        with pytest.raises(ValidationError):
            RuntimeProviderConfig(
                name="test",
                api_base="https://api.test.com/v1",
                models="not-a-list",
            )

    def test_dict_export(self):
        """Test exporting config to dict."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["model-1"],
            api_key="sk-test",
        )

        data = config.model_dump()

        assert data["name"] == "test-provider"
        assert data["api_base"] == "https://api.test.com/v1"
        assert data["models"] == ["model-1"]
        assert data["api_key"] == "sk-test"
        assert data["is_runtime"] is True

    def test_json_serialization(self):
        """Test JSON serialization."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["model-1"],
        )

        json_str = config.model_dump_json()

        assert isinstance(json_str, str)
        assert "test-provider" in json_str
        assert "model-1" in json_str


class TestProviderCapabilities:
    """Test ProviderCapabilities Pydantic model."""

    def test_basic_creation(self):
        """Test creating basic ProviderCapabilities."""
        caps = ProviderCapabilities()

        # All should default to False/None
        assert caps.supports_streaming is False
        assert caps.supports_tools is False
        assert caps.supports_vision is False
        assert caps.supports_json_mode is False
        assert caps.max_context_length is None
        assert caps.max_output_tokens is None

    def test_with_all_capabilities(self):
        """Test creating capabilities with all features enabled."""
        caps = ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
            supports_json_mode=True,
            max_context_length=128000,
            max_output_tokens=4096,
        )

        assert caps.supports_streaming is True
        assert caps.supports_tools is True
        assert caps.supports_vision is True
        assert caps.supports_json_mode is True
        assert caps.max_context_length == 128000
        assert caps.max_output_tokens == 4096

    def test_partial_capabilities(self):
        """Test creating capabilities with some features."""
        caps = ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            max_context_length=8192,
        )

        assert caps.supports_streaming is True
        assert caps.supports_tools is True
        assert caps.supports_vision is False  # Still defaults to False
        assert caps.max_context_length == 8192

    def test_model_is_immutable(self):
        """Test that ProviderCapabilities is immutable (frozen)."""
        caps = ProviderCapabilities(supports_streaming=True)

        # Should not be able to modify
        with pytest.raises((ValidationError, AttributeError)):
            caps.supports_streaming = False

    def test_context_length_validation(self):
        """Test context length field accepts positive integers."""
        caps = ProviderCapabilities(max_context_length=100000)
        assert caps.max_context_length == 100000

        # Should accept None
        caps_none = ProviderCapabilities(max_context_length=None)
        assert caps_none.max_context_length is None

    def test_output_tokens_validation(self):
        """Test output tokens field accepts positive integers."""
        caps = ProviderCapabilities(max_output_tokens=4096)
        assert caps.max_output_tokens == 4096

    def test_boolean_fields_validation(self):
        """Test that boolean fields only accept bool values."""
        # Valid
        caps = ProviderCapabilities(
            supports_streaming=True,
            supports_tools=False,
        )
        assert caps.supports_streaming is True
        assert caps.supports_tools is False

        # Pydantic should coerce truthy values
        caps2 = ProviderCapabilities(supports_streaming=1)
        assert caps2.supports_streaming is True

    def test_dict_export(self):
        """Test exporting capabilities to dict."""
        caps = ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            max_context_length=128000,
        )

        data = caps.model_dump()

        assert data["supports_streaming"] is True
        assert data["supports_tools"] is True
        assert data["max_context_length"] == 128000

    def test_json_serialization(self):
        """Test JSON serialization."""
        caps = ProviderCapabilities(
            supports_streaming=True,
            max_context_length=100000,
        )

        json_str = caps.model_dump_json()

        assert isinstance(json_str, str)
        assert "true" in json_str.lower()  # supports_streaming
        assert "100000" in json_str


class TestProviderModelsIntegration:
    """Integration tests for provider models working together."""

    def test_config_with_capabilities(self):
        """Test that config and capabilities can work together."""
        config = RuntimeProviderConfig(
            name="advanced-provider",
            api_base="https://api.advanced.com/v1",
            models=["advanced-model-1"],
        )

        capabilities = ProviderCapabilities(
            supports_streaming=True,
            supports_tools=True,
            supports_vision=True,
            max_context_length=128000,
        )

        # They should be independent but complementary
        assert config.name == "advanced-provider"
        assert capabilities.supports_streaming is True

    def test_multiple_configs_independent(self):
        """Test that multiple configs don't interfere with each other."""
        config1 = RuntimeProviderConfig(
            name="provider-1",
            api_base="https://api1.com/v1",
            models=["model-1a", "model-1b"],
        )

        config2 = RuntimeProviderConfig(
            name="provider-2",
            api_base="https://api2.com/v1",
            models=["model-2a"],
        )

        # Modify config1
        config1.add_models(["model-1c"])

        # config2 should be unaffected
        assert len(config2.models) == 1
        assert "model-1c" not in config2.models
