# tests/model_management/test_provider_discovery.py
"""
Comprehensive tests for provider_discovery.py.
Target: >90% code coverage

Tests the ProviderDiscovery class which handles discovering models from
OpenAI-compatible APIs and refreshing model lists.
"""

from unittest.mock import Mock, patch, MagicMock
import httpx

from mcp_cli.model_management.provider_discovery import ProviderDiscovery
from mcp_cli.model_management.provider import RuntimeProviderConfig
from mcp_cli.model_management.discovery import DiscoveryResult


class TestDiscoverModelsFromApi:
    """Test the discover_models_from_api() static method."""

    @patch("httpx.Client")
    def test_successful_discovery(self, mock_client_class):
        """Test successful model discovery from API."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "model-1"},
                {"id": "model-2"},
                {"id": "model-3"},
            ]
        }

        # Mock client context manager
        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        result = ProviderDiscovery.discover_models_from_api(
            api_base="https://api.test.com",
            api_key="sk-test-key",
            provider_name="test-provider",
        )

        assert result.success is True
        assert result.provider == "test-provider"
        assert result.models == ["model-1", "model-2", "model-3"]
        assert result.discovered_count == 3
        assert result.error is None

    @patch("httpx.Client")
    def test_discovery_with_v1_suffix(self, mock_client_class):
        """Test discovery when api_base already ends with /v1."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": [{"id": "model-1"}]}

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        ProviderDiscovery.discover_models_from_api(
            api_base="https://api.test.com/v1",
            api_key="sk-test-key",
        )

        # Should not add another /v1
        call_args = mock_client.get.call_args[0]
        assert "/v1/v1" not in call_args[0]

    @patch("httpx.Client")
    def test_discovery_empty_model_list(self, mock_client_class):
        """Test discovery when API returns empty model list."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": []}

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        result = ProviderDiscovery.discover_models_from_api(
            api_base="https://api.test.com",
            api_key="sk-test-key",
        )

        assert result.success is False
        assert result.models == []
        assert "No models found" in result.error

    @patch("httpx.Client")
    def test_discovery_http_error(self, mock_client_class):
        """Test discovery when HTTP request fails."""
        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.HTTPError("Connection failed")
        mock_client_class.return_value.__enter__.return_value = mock_client

        result = ProviderDiscovery.discover_models_from_api(
            api_base="https://api.test.com",
            api_key="sk-test-key",
        )

        assert result.success is False
        assert "Connection failed" in result.error

    @patch("httpx.Client")
    def test_discovery_401_unauthorized(self, mock_client_class):
        """Test discovery with invalid API key."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "401",
            request=Mock(),
            response=Mock(status_code=401),
        )

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        result = ProviderDiscovery.discover_models_from_api(
            api_base="https://api.test.com",
            api_key="sk-invalid-key",
        )

        assert result.success is False

    @patch("httpx.Client")
    def test_discovery_models_without_id(self, mock_client_class):
        """Test discovery when some models don't have 'id' field."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {"id": "model-1"},
                {"name": "no-id"},  # Missing 'id'
                {"id": "model-2"},
            ]
        }

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value.__enter__.return_value = mock_client

        result = ProviderDiscovery.discover_models_from_api(
            api_base="https://api.test.com",
            api_key="sk-test-key",
        )

        # Should only get models with 'id'
        assert result.models == ["model-1", "model-2"]


class TestRefreshProviderModels:
    """Test the refresh_provider_models() static method."""

    @patch.object(ProviderDiscovery, "discover_models_from_api")
    def test_successful_refresh(self, mock_discover):
        """Test successful model refresh."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["old-model-1", "old-model-2"],
            api_key="sk-test-key",
        )

        # Don't pass discovered_count, let it auto-calculate
        mock_discover.return_value = DiscoveryResult(
            provider="test-provider",
            models=["new-model-1", "new-model-2", "new-model-3"],
            success=True,
        )

        result = ProviderDiscovery.refresh_provider_models(config)

        assert result == 3  # Discovered count
        assert config.models == ["new-model-1", "new-model-2", "new-model-3"]

    def test_refresh_no_api_key(self):
        """Test refresh when config has no API key."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["model-1"],
            api_key=None,
        )

        result = ProviderDiscovery.refresh_provider_models(config)

        assert result is None

    def test_refresh_no_api_base(self):
        """Test refresh when config has no API base."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="",
            models=["model-1"],
            api_key="sk-test-key",
        )

        result = ProviderDiscovery.refresh_provider_models(config)

        assert result is None

    @patch.object(ProviderDiscovery, "discover_models_from_api")
    def test_refresh_discovery_fails(self, mock_discover):
        """Test refresh when discovery fails."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["model-1"],
            api_key="sk-test-key",
        )

        mock_discover.return_value = DiscoveryResult(
            provider="test-provider",
            models=[],
            success=False,
            error="Connection failed",
        )

        result = ProviderDiscovery.refresh_provider_models(config)

        assert result is None
        # Original models should remain unchanged
        assert config.models == ["model-1"]

    @patch.object(ProviderDiscovery, "discover_models_from_api")
    def test_refresh_discovery_success_but_no_models(self, mock_discover):
        """Test refresh when discovery succeeds but returns no models."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["old-model"],
            api_key="sk-test-key",
        )

        mock_discover.return_value = DiscoveryResult(
            provider="test-provider",
            models=[],
            success=True,
        )

        result = ProviderDiscovery.refresh_provider_models(config)

        # has_models is False, so should return None
        assert result is None

    @patch.object(ProviderDiscovery, "discover_models_from_api")
    def test_refresh_exception_handling(self, mock_discover):
        """Test that exceptions during refresh are handled gracefully."""
        config = RuntimeProviderConfig(
            name="test-provider",
            api_base="https://api.test.com/v1",
            models=["model-1"],
            api_key="sk-test-key",
        )

        mock_discover.side_effect = Exception("Unexpected error")

        result = ProviderDiscovery.refresh_provider_models(config)

        assert result is None
        # Original models should remain
        assert config.models == ["model-1"]
