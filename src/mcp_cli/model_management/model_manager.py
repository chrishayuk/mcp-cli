# src/mcp_cli/model_management/model_manager.py
"""
from __future__ import annotations

ModelManager - Clean, type-safe LLM provider and model management.

This module provides the main ModelManager class that orchestrates:
- Provider discovery and listing (from chuk_llm configuration)
- Model discovery and listing (from providers and APIs)
- Runtime provider management (OpenAI-compatible APIs)
- Client creation and caching

NO HARDCODED MODELS - All model data comes from:
1. chuk_llm configuration (for standard providers)
2. Runtime provider configs (for custom providers)
3. API discovery (for OpenAI-compatible providers)
"""

import logging
from typing import Any

from mcp_cli.model_management.provider import RuntimeProviderConfig
from mcp_cli.model_management.client_factory import ClientFactory
from mcp_cli.model_management.provider_discovery import ProviderDiscovery

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages LLM providers, models, and client creation.

    This is a clean, refactored implementation that:
    - Uses Pydantic models for type safety
    - Delegates to specialized classes for concerns
    - Contains ZERO hardcoded model names
    - Provides a simple, intuitive API
    """

    def __init__(self) -> None:
        """Initialize ModelManager with chuk_llm configuration."""
        self._chuk_config = None
        self._active_provider: str | None = None
        self._active_model: str | None = None
        self._custom_providers: dict[str, RuntimeProviderConfig] = {}
        self._client_factory = ClientFactory()
        self._discovery_triggered = False

        self._initialize_chuk_llm()
        self._load_custom_providers()

    # ── Initialization ────────────────────────────────────────────────────────

    def _initialize_chuk_llm(self) -> None:
        """Initialize chuk_llm configuration and trigger discovery."""
        try:
            from chuk_llm.configuration import get_config

            self._chuk_config = get_config()
            logger.debug("Loaded chuk_llm configuration")

            # Set defaults from chuk_llm
            if self._chuk_config:
                self._active_provider = "ollama"  # type: ignore[unreachable]  # Safe default
                # Defer model resolution to avoid circular dependencies during __init__
                self._active_model = None

            # Trigger discovery
            self._trigger_discovery()

        except Exception as e:
            logger.error(f"Failed to initialize chuk_llm: {e}")
            # Minimal fallback
            self._chuk_config = None
            self._active_provider = "ollama"
            self._active_model = None  # Will be determined on first use

    def _trigger_discovery(self) -> None:
        """Trigger model discovery for providers."""
        if self._discovery_triggered:
            return

        try:
            from chuk_llm.api.providers import trigger_ollama_discovery_and_refresh

            new_functions = trigger_ollama_discovery_and_refresh()
            logger.info(
                f"ModelManager discovery: {len(new_functions)} new Ollama functions"
            )
            self._discovery_triggered = True
        except Exception as e:
            logger.warning(f"ModelManager discovery failed (continuing anyway): {e}")

    def _load_custom_providers(self) -> None:
        """Load custom providers from preferences."""
        try:
            from mcp_cli.utils.preferences import get_preference_manager

            prefs = get_preference_manager()
            custom_providers = prefs.get_custom_providers()

            for name, provider_data in custom_providers.items():
                # Convert dict to Pydantic model
                config = RuntimeProviderConfig(
                    name=name,
                    api_base=provider_data.get("api_base", ""),
                    models=provider_data.get("models", []),
                    default_model=provider_data.get("default_model"),
                    api_key=None,  # Not stored in preferences for security
                    is_runtime=False,  # These are persisted in preferences
                )
                self._custom_providers[name] = config
                logger.debug(f"Loaded custom provider: {name}")

        except Exception as e:
            logger.warning(f"Failed to load custom providers: {e}")

    # ── Provider Management ───────────────────────────────────────────────────

    def get_available_providers(self) -> list[str]:
        """
        Get list of all available providers.

        Returns:
            List of provider names
        """
        providers = []

        # Get chuk_llm providers
        if self._chuk_config:
            try:  # type: ignore[unreachable]
                all_providers = self._chuk_config.get_all_providers()
                # Ollama first, then others alphabetically
                if "ollama" in all_providers:
                    providers.append("ollama")
                providers.extend([p for p in sorted(all_providers) if p != "ollama"])
            except Exception as e:
                logger.error(f"Failed to get chuk_llm providers: {e}")

        # Add custom providers
        for custom_name in self._custom_providers.keys():
            if custom_name not in providers:
                providers.append(custom_name)

        return providers if providers else ["ollama"]  # Safe fallback

    def add_runtime_provider(
        self,
        name: str,
        api_base: str,
        api_key: str | None = None,
        models: list[str] | None = None,
    ) -> RuntimeProviderConfig:
        """
        Add a provider at runtime (not persisted).

        Args:
            name: Provider name
            api_base: API base URL
            api_key: API key (kept in memory only)
            models: List of available models (if None, will attempt to discover)

        Returns:
            The created RuntimeProviderConfig
        """
        # If no models provided, try to discover them from the API
        if not models and api_key:
            logger.info(f"Attempting to discover models from {name} at {api_base}")
            discovery_result = ProviderDiscovery.discover_models_from_api(
                api_base, api_key, name
            )

            if discovery_result.success and discovery_result.has_models:
                logger.info(
                    f"Discovered {discovery_result.discovered_count} models from {name}"
                )
                models = list(discovery_result.models)
            else:
                logger.warning(
                    f"Discovery failed for {name}: {discovery_result.error or 'No models found'}"
                )

        # Create the RuntimeProviderConfig
        config = RuntimeProviderConfig(
            name=name,
            api_base=api_base,
            models=models if models else [] or [],
            api_key=api_key,
            is_runtime=True,
            default_model=None,  # Will be auto-set by model_validator
        )

        # Store the config
        self._custom_providers[name] = config

        logger.info(f"Added runtime provider: {name} with {len(config.models)} models")
        return config

    def is_custom_provider(self, name: str) -> bool:
        """Check if a provider is custom (either from preferences or runtime)."""
        return name in self._custom_providers

    def is_runtime_provider(self, name: str) -> bool:
        """Check if a provider was added at runtime."""
        return (
            name in self._custom_providers and self._custom_providers[name].is_runtime
        )

    # ── Model Management ──────────────────────────────────────────────────────

    def get_available_models(self, provider: str | None = None) -> list[str]:
        """
        Get available models for a provider.

        NO HARDCODED MODELS - All come from configuration or discovery.

        Args:
            provider: Provider name (uses active provider if None)

        Returns:
            List of available model names
        """
        target_provider = provider or self._active_provider
        if not target_provider:
            logger.warning("No provider specified and no active provider")
            return []

        # Check custom providers first (type-safe Pydantic model)
        if target_provider in self._custom_providers:
            config = self._custom_providers[target_provider]
            logger.debug(
                f"Returning {len(config.models)} models for custom provider {target_provider}"
            )
            return list(config.models)

        # Use chuk_llm configuration
        if not self._chuk_config:
            logger.warning("No chuk_llm config available, cannot get models")
            return []

        try:  # type: ignore[unreachable]
            from chuk_llm.llm.client import list_available_providers

            providers = list_available_providers()
            provider_info = providers.get(target_provider, {})

            if "error" in provider_info:
                logger.warning(
                    f"Provider {target_provider} has error: {provider_info['error']}"
                )
                return []

            # Get models from chuk_llm (which handles discovery)
            models = provider_info.get("models", [])
            return list(models) if models else []

        except Exception as e:
            logger.error(f"Failed to get models for {target_provider}: {e}")
            return []

    def get_default_model(self, provider: str) -> str:
        """
        Get the default model for a provider.

        NO HARDCODED MODELS - All come from configuration.

        Args:
            provider: Provider name

        Returns:
            Default model name, or "default" if none found
        """
        try:
            # Check custom providers first
            if provider in self._custom_providers:
                config = self._custom_providers[provider]
                if config.default_model:
                    return config.default_model
                if config.has_models:
                    return config.models[0]

            # Use chuk_llm configuration
            if self._chuk_config:
                provider_config = self._chuk_config.get_provider(provider)  # type: ignore[unreachable]
                default = provider_config.default_model
                if default:
                    return str(default)

            # Fallback: get first available model
            available_models = self.get_available_models(provider)
            return available_models[0] if available_models else "default"

        except Exception as e:
            logger.warning(f"Could not get default model for {provider}: {e}")
            # Last resort: try first available model
            available_models = self.get_available_models(provider)
            return available_models[0] if available_models else "default"

    def refresh_models(self, provider: str | None = None) -> int:
        """
        Manually refresh models for a provider.

        Args:
            provider: Provider name (refreshes all if None)

        Returns:
            Number of new models discovered
        """
        # Check if it's a runtime/custom provider first
        if provider and provider in self._custom_providers:
            config = self._custom_providers[provider]
            count = ProviderDiscovery.refresh_provider_models(config)
            if count is not None:
                self._client_factory.clear_cache()  # Clear cache after refresh
                return count
            return 0

        # Use chuk_llm refresh for standard providers
        try:
            if provider == "ollama" or provider is None:
                from chuk_llm.api.providers import trigger_ollama_discovery_and_refresh

                new_functions = trigger_ollama_discovery_and_refresh()
                logger.info(f"Refreshed Ollama: {len(new_functions)} functions")
                return len(new_functions)
            else:
                from chuk_llm.api.providers import refresh_provider_functions

                new_functions = refresh_provider_functions(provider)
                logger.info(f"Refreshed {provider}: {len(new_functions)} functions")
                return len(new_functions)
        except Exception as e:
            logger.error(f"Failed to refresh models for {provider}: {e}")
            return 0

    # ── Active Provider/Model Management ──────────────────────────────────────

    def get_active_provider(self) -> str:
        """Get the currently active provider."""
        return self._active_provider or "ollama"

    def get_active_model(self) -> str:
        """Get the currently active model."""
        if not self._active_model:
            self._active_model = self.get_default_model(self.get_active_provider())
        return self._active_model

    def set_active_provider(self, provider: str):
        """Set the active provider."""
        self._active_provider = provider
        logger.debug(f"Set active provider: {provider}")

    def switch_provider(self, provider: str):
        """Switch to a different provider and its default model."""
        self.set_active_provider(provider)
        self._active_model = self.get_default_model(provider)
        logger.debug(f"Switched to {provider}/{self._active_model}")

    def switch_model(self, provider: str, model: str):
        """Switch to a specific provider and model."""
        self._active_provider = provider
        self._active_model = model
        logger.debug(f"Switched to {provider}/{model}")

    # ── Client Management ─────────────────────────────────────────────────────

    def get_client(self, provider: str | None = None, model: str | None = None) -> Any:
        """
        Get a client for the specified or active provider/model.

        Args:
            provider: Provider name (uses active if None)
            model: Model name (uses active if None)

        Returns:
            LLM client instance
        """
        target_provider = provider or self._active_provider
        target_model = model or self._active_model

        if not target_provider:
            raise ValueError("No provider specified and no active provider")

        # Custom provider
        if target_provider in self._custom_providers:
            config = self._custom_providers[target_provider]
            return self._client_factory.get_client(
                target_provider, target_model, config=config
            )

        # Standard provider (chuk_llm)
        return self._client_factory.get_client(
            target_provider, target_model, chuk_config=self._chuk_config
        )

    # ── Validation ────────────────────────────────────────────────────────────

    def validate_provider(self, provider: str) -> bool:
        """Check if a provider is available."""
        return provider in self.get_available_providers()

    def validate_model(self, model: str, provider: str | None = None) -> bool:
        """Check if a model is available for a provider."""
        target_provider = provider or self._active_provider
        if not target_provider:
            return False

        available = self.get_available_models(target_provider)
        return model in available

    def detect_provider_for_model(self, model: str) -> str | None:
        """
        Detect which provider supports a given model.

        Uses a two-phase approach:
        1. Check providers with specific models (exact match)
        2. Use pattern-based detection for providers with wildcard models

        Args:
            model: Model name to search for

        Returns:
            Provider name if found, None otherwise
        """
        if not model:
            return None

        available_providers = self.get_available_providers()

        # Phase 1: Check for exact matches (providers with specific models)
        exact_matches = []
        wildcard_providers = []

        for provider in available_providers:
            available_models = self.get_available_models(provider)

            # Check if this is a wildcard provider (returns ["*"])
            if available_models == ["*"]:
                wildcard_providers.append(provider)
            elif model in available_models:
                exact_matches.append(provider)
            # Special handling for ollama-style tags (e.g., "llama3.2" matches "llama3.2:latest")
            elif provider == "ollama":
                for available_model in available_models:
                    if ":" in available_model and available_model.startswith(
                        model + ":"
                    ):
                        exact_matches.append(provider)
                        break

        # If we have exact matches, prefer those
        if exact_matches:
            if len(exact_matches) == 1:
                logger.info(
                    f"Detected provider '{exact_matches[0]}' for model '{model}'"
                )
                return exact_matches[0]

            # Multiple exact matches - prefer non-ollama
            for provider in exact_matches:
                if provider != "ollama":
                    logger.info(
                        f"Detected provider '{provider}' for model '{model}' (multiple matches, preferring non-ollama)"
                    )
                    return provider
            return exact_matches[0]

        # Phase 2: Pattern-based detection for wildcard providers
        model_lower = model.lower()

        # Common model name patterns
        pattern_map = {
            "gpt-": "openai",
            "o1": "openai",
            "claude": "anthropic",
            "gemini": "gemini",
            "llama": "groq",  # Prefer groq for llama models (faster than ollama for API)
            "mixtral": "mistral",
            "mistral": "mistral",
            "deepseek": "deepseek",
        }

        for pattern, provider in pattern_map.items():
            if model_lower.startswith(pattern) and provider in wildcard_providers:
                logger.info(
                    f"Detected provider '{provider}' for model '{model}' (pattern-based)"
                )
                return provider

        # No pattern match - return None (caller will fall back to current provider)
        logger.debug(f"No provider found for model: {model}")
        return None

    # ── Utility Methods ───────────────────────────────────────────────────────

    def __str__(self):
        return f"ModelManager(provider={self._active_provider}, model={self._active_model})"

    def __repr__(self):
        return (
            f"ModelManager(provider='{self._active_provider}', "
            f"model='{self._active_model}', "
            f"cached_clients={self._client_factory.get_cache_size()})"
        )
