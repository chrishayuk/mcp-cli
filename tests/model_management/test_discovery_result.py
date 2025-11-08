# tests/model_management/test_discovery.py
"""
Comprehensive tests for discovery.py Pydantic models.
Target: >90% code coverage
"""

import pytest
from pydantic import ValidationError

from mcp_cli.model_management.discovery import DiscoveryResult


class TestDiscoveryResult:
    """Test DiscoveryResult Pydantic model."""

    def test_successful_discovery(self):
        """Test creating a successful discovery result."""
        result = DiscoveryResult(
            provider="test-provider",
            models=["model-1", "model-2", "model-3"],
            success=True,
        )

        assert result.provider == "test-provider"
        assert result.models == ["model-1", "model-2", "model-3"]
        assert result.success is True
        assert result.error is None
        assert result.discovered_count == 3

    def test_failed_discovery(self):
        """Test creating a failed discovery result."""
        result = DiscoveryResult(
            provider="test-provider",
            models=[],
            success=False,
            error="API key invalid",
        )

        assert result.provider == "test-provider"
        assert result.models == []
        assert result.success is False
        assert result.error == "API key invalid"
        assert result.discovered_count == 0

    def test_auto_count_from_models(self):
        """Test that discovered_count is auto-calculated from models."""
        result = DiscoveryResult(
            provider="test-provider",
            models=["m1", "m2", "m3", "m4", "m5"],
            success=True,
        )

        # Should auto-calculate
        assert result.discovered_count == 5

    def test_explicit_discovered_count(self):
        """Test explicitly setting discovered_count."""
        result = DiscoveryResult(
            provider="test-provider",
            models=["m1", "m2"],
            success=True,
            discovered_count=2,
        )

        assert result.discovered_count == 2

    def test_discovered_count_override(self):
        """Test that explicit count overrides auto-calculation."""
        result = DiscoveryResult(
            provider="test-provider",
            models=["m1", "m2"],
            success=True,
            discovered_count=10,  # Explicit value
        )

        # Should use explicit value
        assert result.discovered_count == 10

    def test_has_models_property_true(self):
        """Test has_models property when models exist."""
        result = DiscoveryResult(
            provider="test-provider",
            models=["model-1"],
            success=True,
        )

        assert result.has_models is True

    def test_has_models_property_false(self):
        """Test has_models property when no models."""
        result = DiscoveryResult(
            provider="test-provider",
            models=[],
            success=False,
        )

        assert result.has_models is False

    def test_error_message_property_with_error(self):
        """Test error_message property when error exists."""
        result = DiscoveryResult(
            provider="openai-compatible",
            models=[],
            success=False,
            error="Connection timeout",
        )

        error_msg = result.error_message

        assert "openai-compatible" in error_msg
        assert "Connection timeout" in error_msg
        assert "Discovery failed" in error_msg

    def test_error_message_property_without_error(self):
        """Test error_message property when no error."""
        result = DiscoveryResult(
            provider="test-provider",
            models=["m1"],
            success=True,
        )

        assert result.error_message == ""

    def test_empty_models_list(self):
        """Test with empty models list."""
        result = DiscoveryResult(
            provider="test-provider",
            models=[],
            success=True,
        )

        assert result.models == []
        assert result.discovered_count == 0
        assert result.has_models is False

    def test_no_models_provided(self):
        """Test creating result without models field."""
        result = DiscoveryResult(
            provider="test-provider",
            success=False,
            error="Not found",
        )

        # Should default to empty list
        assert result.models == []
        assert result.discovered_count == 0

    def test_model_is_immutable(self):
        """Test that DiscoveryResult is immutable (frozen)."""
        result = DiscoveryResult(
            provider="test-provider",
            models=["m1"],
            success=True,
        )

        # Should not be able to modify
        with pytest.raises((ValidationError, AttributeError, TypeError)):
            result.provider = "different-provider"

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            result.success = False

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            result.models = ["m2"]

    def test_required_fields_validation(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError) as exc_info:
            DiscoveryResult()

        errors = exc_info.value.errors()
        field_names = {error["loc"][0] for error in errors}
        assert "provider" in field_names
        assert "success" in field_names

    def test_success_field_validation(self):
        """Test success field must be boolean."""
        # Valid
        result = DiscoveryResult(provider="test", success=True)
        assert result.success is True

        result2 = DiscoveryResult(provider="test", success=False)
        assert result2.success is False

        # Pydantic should coerce
        result3 = DiscoveryResult(provider="test", success=1)
        assert result3.success is True

    def test_models_field_type_validation(self):
        """Test models field type validation."""
        # Valid: list of strings
        result = DiscoveryResult(
            provider="test",
            models=["m1", "m2"],
            success=True,
        )
        assert result.models == ["m1", "m2"]

        # Invalid: not a list
        with pytest.raises(ValidationError):
            DiscoveryResult(
                provider="test",
                models="not-a-list",
                success=True,
            )

    def test_error_field_optional(self):
        """Test that error field is optional."""
        # Without error
        result1 = DiscoveryResult(
            provider="test",
            success=True,
            models=["m1"],
        )
        assert result1.error is None

        # With error
        result2 = DiscoveryResult(
            provider="test",
            success=False,
            error="Something went wrong",
        )
        assert result2.error == "Something went wrong"

    def test_dict_export(self):
        """Test exporting result to dict."""
        result = DiscoveryResult(
            provider="test-provider",
            models=["model-1", "model-2"],
            success=True,
            discovered_count=2,
        )

        data = result.model_dump()

        assert data["provider"] == "test-provider"
        assert data["models"] == ["model-1", "model-2"]
        assert data["success"] is True
        assert data["discovered_count"] == 2

    def test_json_serialization(self):
        """Test JSON serialization."""
        result = DiscoveryResult(
            provider="openai-compatible",
            models=["moonshot-v1-8k"],
            success=True,
        )

        json_str = result.model_dump_json()

        assert isinstance(json_str, str)
        assert "openai-compatible" in json_str
        assert "moonshot-v1-8k" in json_str
        assert "true" in json_str.lower()  # success field

    def test_large_model_list(self):
        """Test with a large list of models."""
        models = [f"model-{i}" for i in range(100)]

        result = DiscoveryResult(
            provider="test-provider",
            models=models,
            success=True,
        )

        assert len(result.models) == 100
        assert result.discovered_count == 100
        assert result.has_models is True

    def test_special_characters_in_error(self):
        """Test error field with special characters."""
        error_msg = "API Error: 401 - Invalid authentication 'Bearer token'"

        result = DiscoveryResult(
            provider="test-provider",
            success=False,
            error=error_msg,
        )

        assert result.error == error_msg
        assert error_msg in result.error_message

    def test_unicode_in_provider_name(self):
        """Test provider name with unicode characters."""
        result = DiscoveryResult(
            provider="test-provider-中文",
            models=["model-1"],
            success=True,
        )

        assert result.provider == "test-provider-中文"

    def test_success_true_with_no_models(self):
        """Test edge case: success=True but no models discovered."""
        result = DiscoveryResult(
            provider="test-provider",
            models=[],
            success=True,
        )

        assert result.success is True
        assert result.has_models is False
        assert result.discovered_count == 0

    def test_success_false_with_models(self):
        """Test edge case: success=False but models present."""
        result = DiscoveryResult(
            provider="test-provider",
            models=["partial-result"],
            success=False,
            error="Partial failure",
        )

        assert result.success is False
        assert result.has_models is True
        assert len(result.models) == 1


class TestDiscoveryResultUsagePatterns:
    """Test common usage patterns for DiscoveryResult."""

    def test_typical_success_pattern(self):
        """Test typical successful discovery pattern."""
        # Simulate API discovery success
        result = DiscoveryResult(
            provider="moonshot",
            models=[
                "moonshot-v1-8k",
                "moonshot-v1-32k",
                "moonshot-v1-128k",
            ],
            success=True,
        )

        # Typical checks
        if result.success and result.has_models:
            assert len(result.models) > 0
            first_model = result.models[0]
            assert first_model == "moonshot-v1-8k"

    def test_typical_failure_pattern(self):
        """Test typical failed discovery pattern."""
        # Simulate API discovery failure
        result = DiscoveryResult(
            provider="invalid-provider",
            models=[],
            success=False,
            error="401 Unauthorized",
        )

        # Typical checks
        if not result.success:
            assert result.error is not None
            assert result.has_models is False
            error_msg = result.error_message
            assert len(error_msg) > 0

    def test_conditional_processing(self):
        """Test conditional processing based on result."""
        successful_result = DiscoveryResult(
            provider="test-provider",
            models=["m1", "m2"],
            success=True,
        )

        failed_result = DiscoveryResult(
            provider="test-provider",
            models=[],
            success=False,
            error="Network error",
        )

        # Process based on success
        processed_models = successful_result.models if successful_result.success else []
        assert processed_models == ["m1", "m2"]

        processed_models_failed = failed_result.models if failed_result.success else []
        assert processed_models_failed == []

    def test_result_comparison(self):
        """Test comparing discovery results."""
        result1 = DiscoveryResult(
            provider="provider-a",
            models=["m1", "m2"],
            success=True,
        )

        result2 = DiscoveryResult(
            provider="provider-b",
            models=["m1", "m2", "m3"],
            success=True,
        )

        # Compare counts
        assert result2.discovered_count > result1.discovered_count
        assert result2.has_models and result1.has_models

    def test_multiple_discovery_results(self):
        """Test handling multiple discovery results."""
        results = [
            DiscoveryResult(
                provider="provider-1",
                models=["m1"],
                success=True,
            ),
            DiscoveryResult(
                provider="provider-2",
                models=["m2", "m3"],
                success=True,
            ),
            DiscoveryResult(
                provider="provider-3",
                models=[],
                success=False,
                error="Failed",
            ),
        ]

        successful_results = [r for r in results if r.success]
        assert len(successful_results) == 2

        total_models = sum(r.discovered_count for r in successful_results)
        assert total_models == 3
