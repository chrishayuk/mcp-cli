"""Tests for provider token management with hierarchical resolution."""

import os
from unittest.mock import Mock, patch

from mcp_cli.auth.provider_tokens import (
    get_provider_env_var_name,
    get_provider_token_with_hierarchy,
    check_provider_token_status,
    set_provider_token,
    delete_provider_token,
    get_provider_token_display_status,
)


class TestProviderEnvVarName:
    """Tests for get_provider_env_var_name."""

    def test_known_providers(self):
        """Test known provider mappings."""
        assert get_provider_env_var_name("openai") == "OPENAI_API_KEY"
        assert get_provider_env_var_name("anthropic") == "ANTHROPIC_API_KEY"
        assert get_provider_env_var_name("gemini") == "GEMINI_API_KEY"
        assert get_provider_env_var_name("groq") == "GROQ_API_KEY"

    def test_unknown_provider(self):
        """Test default pattern for unknown providers."""
        assert get_provider_env_var_name("customai") == "CUSTOMAI_API_KEY"
        assert get_provider_env_var_name("my-provider") == "MY_PROVIDER_API_KEY"

    def test_case_insensitive(self):
        """Test case insensitive handling."""
        assert get_provider_env_var_name("OpenAI") == "OPENAI_API_KEY"
        assert get_provider_env_var_name("ANTHROPIC") == "ANTHROPIC_API_KEY"


class TestProviderTokenHierarchy:
    """Tests for hierarchical token resolution."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key-123"})
    def test_env_var_takes_precedence(self):
        """Test that environment variable takes precedence."""
        mock_token_manager = Mock()
        mock_token_manager.token_store.retrieve_generic.return_value = "storage-key-456"

        api_key, source = get_provider_token_with_hierarchy(
            "openai", mock_token_manager
        )

        assert api_key == "env-key-123"
        assert source == "env"
        # Storage should not be checked when env var is set
        mock_token_manager.token_store.retrieve_generic.assert_not_called()

    @patch.dict(os.environ, {}, clear=True)
    def test_storage_fallback(self):
        """Test fallback to storage when env var not set."""
        mock_token_manager = Mock()
        mock_token_manager.token_store.retrieve_generic.return_value = "storage-key-456"

        api_key, source = get_provider_token_with_hierarchy(
            "openai", mock_token_manager
        )

        assert api_key == "storage-key-456"
        assert source == "storage"
        mock_token_manager.token_store.retrieve_generic.assert_called_once_with(
            "openai", namespace="provider"
        )

    @patch.dict(os.environ, {}, clear=True)
    def test_no_token_found(self):
        """Test when no token is found."""
        mock_token_manager = Mock()
        mock_token_manager.token_store.retrieve_generic.return_value = None

        api_key, source = get_provider_token_with_hierarchy(
            "openai", mock_token_manager
        )

        assert api_key is None
        assert source == "none"

    @patch.dict(os.environ, {}, clear=True)
    def test_no_token_manager(self):
        """Test when no token manager is provided."""
        api_key, source = get_provider_token_with_hierarchy("openai", None)

        assert api_key is None
        assert source == "none"


class TestCheckProviderTokenStatus:
    """Tests for check_provider_token_status."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"})
    def test_status_with_env_var(self):
        """Test status when env var is set."""
        mock_token_manager = Mock()
        mock_token_manager.token_store.retrieve_generic.return_value = None

        status = check_provider_token_status("openai", mock_token_manager)

        assert status["has_token"] is True
        assert status["source"] == "env"
        assert status["env_var"] == "OPENAI_API_KEY"
        assert status["in_env"] is True
        assert status["in_storage"] is False

    @patch.dict(os.environ, {}, clear=True)
    def test_status_with_storage(self):
        """Test status when only storage has token."""
        mock_token_manager = Mock()
        mock_token_manager.token_store.retrieve_generic.return_value = "storage-key"

        status = check_provider_token_status("openai", mock_token_manager)

        assert status["has_token"] is True
        assert status["source"] == "storage"
        assert status["env_var"] == "OPENAI_API_KEY"
        assert status["in_env"] is False
        assert status["in_storage"] is True

    @patch.dict(os.environ, {}, clear=True)
    def test_status_no_token(self):
        """Test status when no token exists."""
        mock_token_manager = Mock()
        mock_token_manager.token_store.retrieve_generic.return_value = None

        status = check_provider_token_status("openai", mock_token_manager)

        assert status["has_token"] is False
        assert status["source"] == "none"
        assert status["in_env"] is False
        assert status["in_storage"] is False


class TestSetProviderToken:
    """Tests for set_provider_token."""

    def test_set_provider_token_success(self):
        """Test successfully storing a provider token."""
        mock_token_manager = Mock()
        mock_token_manager.token_store.store_generic.return_value = None
        mock_token_manager.registry.register.return_value = None

        result = set_provider_token("openai", "test-api-key", mock_token_manager)

        assert result is True
        mock_token_manager.token_store.store_generic.assert_called_once_with(
            "openai", "test-api-key", namespace="provider"
        )
        mock_token_manager.registry.register.assert_called_once()

    def test_set_provider_token_failure(self):
        """Test handling of storage failure."""
        mock_token_manager = Mock()
        mock_token_manager.token_store.store_generic.side_effect = Exception(
            "Storage error"
        )

        result = set_provider_token("openai", "test-api-key", mock_token_manager)

        assert result is False


class TestDeleteProviderToken:
    """Tests for delete_provider_token."""

    def test_delete_provider_token_success(self):
        """Test successfully deleting a provider token."""
        mock_token_manager = Mock()
        mock_token_manager.token_store.delete_generic.return_value = True
        mock_token_manager.registry.unregister.return_value = None

        result = delete_provider_token("openai", mock_token_manager)

        assert result is True
        mock_token_manager.token_store.delete_generic.assert_called_once_with(
            "openai", namespace="provider"
        )
        mock_token_manager.registry.unregister.assert_called_once_with(
            "openai", "provider"
        )

    def test_delete_provider_token_not_found(self):
        """Test deleting non-existent token."""
        mock_token_manager = Mock()
        mock_token_manager.token_store.delete_generic.return_value = False

        result = delete_provider_token("openai", mock_token_manager)

        assert result is False
        # Registry unregister should not be called if delete failed
        mock_token_manager.registry.unregister.assert_not_called()

    def test_delete_provider_token_exception(self):
        """Test delete_provider_token with exception."""
        mock_token_manager = Mock()
        mock_token_manager.token_store.delete_generic.side_effect = Exception(
            "Delete error"
        )

        result = delete_provider_token("openai", mock_token_manager)

        assert result is False


class TestGetProviderTokenDisplayStatus:
    """Tests for get_provider_token_display_status."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"})
    def test_display_status_env(self):
        """Test display status for env var."""
        mock_token_manager = Mock()
        mock_token_manager.token_store.retrieve_generic.return_value = None

        status = get_provider_token_display_status("openai", mock_token_manager)

        assert "✅" in status
        assert "OPENAI_API_KEY" in status

    @patch.dict(os.environ, {}, clear=True)
    def test_display_status_storage(self):
        """Test display status for storage."""
        mock_token_manager = Mock()
        mock_token_manager.token_store.retrieve_generic.return_value = "storage-key"

        status = get_provider_token_display_status("openai", mock_token_manager)

        assert "🔐" in status
        assert "storage" in status

    @patch.dict(os.environ, {}, clear=True)
    def test_display_status_not_set(self):
        """Test display status when not set."""
        mock_token_manager = Mock()
        mock_token_manager.token_store.retrieve_generic.return_value = None

        status = get_provider_token_display_status("openai", mock_token_manager)

        assert "❌" in status
        assert "not set" in status


class TestProviderTokenEdgeCases:
    """Tests for edge cases and error handling."""

    @patch.dict(os.environ, {}, clear=True)
    def test_storage_retrieval_exception(self):
        """Test handling of storage retrieval exceptions."""
        mock_token_manager = Mock()
        mock_token_manager.token_store.retrieve_generic.side_effect = Exception(
            "Storage error"
        )

        # Should not raise, should return None
        api_key, source = get_provider_token_with_hierarchy(
            "openai", mock_token_manager
        )

        assert api_key is None
        assert source == "none"

    @patch.dict(os.environ, {}, clear=True)
    def test_check_status_storage_exception(self):
        """Test check_provider_token_status with storage exception."""
        mock_token_manager = Mock()
        mock_token_manager.token_store.retrieve_generic.side_effect = Exception(
            "Storage error"
        )

        status = check_provider_token_status("openai", mock_token_manager)

        # Should gracefully handle exception
        assert status["has_token"] is False
        assert status["in_storage"] is False
        assert status["in_env"] is False

    @patch("chuk_llm.llm.client.list_available_providers")
    @patch.dict(os.environ, {}, clear=True)
    def test_list_all_provider_tokens_empty(self, mock_list_providers):
        """Test list_all_provider_tokens with no tokens in storage."""
        from mcp_cli.auth.provider_tokens import list_all_provider_tokens

        mock_list_providers.return_value = {"openai": {}, "anthropic": {}}

        mock_token_manager = Mock()
        mock_token_manager.token_store.retrieve_generic.return_value = None

        result = list_all_provider_tokens(mock_token_manager)

        # Should return empty since no tokens are in storage
        assert result == {}

    @patch("chuk_llm.llm.client.list_available_providers")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_list_all_provider_tokens_with_entries(self, mock_list_providers):
        """Test list_all_provider_tokens only returns storage tokens."""
        from mcp_cli.auth.provider_tokens import list_all_provider_tokens

        mock_list_providers.return_value = {
            "openai": {},
            "anthropic": {},
            "ollama": {},  # Should be skipped
        }

        mock_token_manager = Mock()
        mock_token_manager.token_store.retrieve_generic.side_effect = (
            lambda name, namespace: "storage-key" if name == "anthropic" else None
        )

        result = list_all_provider_tokens(mock_token_manager)

        # Only anthropic should be in result (it's in storage)
        # openai is in env but NOT storage, so it should be excluded
        assert "openai" not in result  # In env, not in storage
        assert "anthropic" in result  # In storage
        assert "ollama" not in result  # Should skip ollama
        assert result["anthropic"]["in_storage"] is True

    @patch("chuk_llm.llm.client.list_available_providers")
    def test_list_all_provider_tokens_exception(self, mock_list_providers):
        """Test list_all_provider_tokens with exception."""
        from mcp_cli.auth.provider_tokens import list_all_provider_tokens

        mock_list_providers.side_effect = Exception("Provider list error")
        mock_token_manager = Mock()

        result = list_all_provider_tokens(mock_token_manager)

        # Should return empty dict on error
        assert result == {}
