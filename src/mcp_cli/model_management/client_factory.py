# src/mcp_cli/model_management/client_factory.py
"""
from __future__ import annotations

Client factory for creating LLM clients using chuk_llm.

This module handles the creation and caching of LLM clients for all providers
using chuk_llm's unified client factory. For custom OpenAI-compatible providers,
we pass api_key and api_base overrides to chuk_llm's get_client() function.

NO direct OpenAI client creation - everything goes through chuk_llm.
"""

import logging
from typing import TYPE_CHECKING, Any

from mcp_cli.model_management.provider import RuntimeProviderConfig

if TYPE_CHECKING:
    from chuk_llm.llm.core.base import BaseLLMClient

logger = logging.getLogger(__name__)


class ClientFactory:
    """
    Factory for creating and caching LLM clients.

    This class manages the creation and caching of LLM clients for different
    providers and models, ensuring efficient reuse and proper configuration.
    """

    def __init__(self) -> None:
        """Initialize the client factory with an empty cache."""
        self._client_cache: dict[str, Any] = {}

    def get_client(
        self,
        provider: str,
        model: str | None,
        config: RuntimeProviderConfig | None = None,
        chuk_config: Any = None,
    ) -> "BaseLLMClient":
        """
        Get or create a client for the specified provider and model.

        Args:
            provider: Provider name
            model: Model name (optional)
            config: RuntimeProviderConfig for custom providers (optional)
            chuk_config: chuk_llm configuration (optional)

        Returns:
            LLM client instance

        Raises:
            ValueError: If provider not found or configuration invalid
        """
        # Custom provider path
        if config:
            return self._get_custom_provider_client(provider, model, config)

        # Standard provider path (chuk_llm)
        if chuk_config:
            return self._get_chuk_llm_client(provider, model, chuk_config)

        raise ValueError(f"No configuration available for provider: {provider}")

    def _get_custom_provider_client(
        self,
        provider: str,
        model: str | None,
        config: RuntimeProviderConfig,
    ) -> Any:
        """
        Get a client for a custom OpenAI-compatible provider using chuk_llm.

        Args:
            provider: Provider name
            model: Model name (optional)
            config: RuntimeProviderConfig

        Returns:
            chuk_llm client instance

        Raises:
            ValueError: If API key not found
        """
        from mcp_cli.auth.provider_tokens import get_provider_token_with_hierarchy
        from mcp_cli.auth import TokenManager

        # Check Pydantic model's api_key first (for runtime providers)
        api_key = config.api_key

        if not api_key:
            # Use hierarchical resolution (env vars > storage)
            try:
                token_manager = TokenManager(service_name="mcp-cli")
                api_key, source = get_provider_token_with_hierarchy(
                    provider, token_manager
                )
                if api_key:
                    logger.debug(f"Using {provider} API key from {source}")
            except Exception as e:
                logger.debug(f"Token resolution failed for {provider}: {e}")

        if not api_key:
            env_var = f"{provider.upper().replace('-', '_')}_API_KEY"
            raise ValueError(
                f"No API key found for provider {provider}. "
                f"Use 'mcp-cli token set-provider {provider}' or set {env_var}"
            )

        cache_key = f"custom:{provider}:{model or 'default'}"

        if cache_key not in self._client_cache:
            # Use chuk_llm's client factory with api_key and api_base overrides
            # This works for ALL OpenAI-compatible providers
            try:
                from chuk_llm.llm.client import get_client

                client = get_client(
                    provider="openai_compatible",  # Use chuk_llm's openai_compatible provider
                    model=model or config.default_model,
                    api_key=api_key,
                    api_base=config.api_base,
                )
                self._client_cache[cache_key] = client
                logger.debug(f"Created chuk_llm client for custom provider {provider}")
            except Exception as e:
                logger.error(f"Failed to create chuk_llm client for {provider}: {e}")
                raise

        return self._client_cache[cache_key]

    def _get_chuk_llm_client(
        self, provider: str, model: str | None, chuk_config: Any
    ) -> Any:
        """
        Get a client from chuk_llm for standard providers.

        Args:
            provider: Provider name
            model: Model name (optional)
            chuk_config: chuk_llm configuration

        Returns:
            chuk_llm client instance
        """
        cache_key = f"{provider}:{model or 'default'}"

        if cache_key not in self._client_cache:
            try:
                from chuk_llm.llm.client import get_client

                client = get_client(provider=provider, model=model)
                self._client_cache[cache_key] = client
                logger.debug(f"Created chuk_llm client for {provider}/{model}")
            except Exception as e:
                logger.error(f"Failed to create client for {provider}: {e}")
                raise

        return self._client_cache[cache_key]

    def clear_cache(self):
        """Clear the client cache."""
        self._client_cache.clear()
        logger.debug("Cleared client cache")

    def get_cache_size(self) -> int:
        """Get the number of cached clients."""
        return len(self._client_cache)
