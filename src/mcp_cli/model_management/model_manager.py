# src/mcp_cli/model_management/model_manager.py
"""
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
from typing import List, Dict, Any, Optional

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

    def __init__(self):
        """Initialize ModelManager with chuk_llm configuration."""
        self._chuk_config = None
        self._active_provider: Optional[str] = None
        self._active_model: Optional[str] = None
        self._custom_providers: Dict[str, RuntimeProviderConfig] = {}
        self._client_factory = ClientFactory()
        self._discovery_triggered = False

        self._initialize_chuk_llm()
        self._load_custom_providers()

    # ── Initialization ────────────────────────────────────────────────────────

    def _initialize_chuk_llm(self):
        """Initialize chuk_llm configuration and trigger discovery."""
        try:
            from chuk_llm.configuration import get_config

            self._chuk_config = get_config()
            logger.debug("Loaded chuk_llm configuration")

            # Set defaults from chuk_llm
            if self._chuk_config:
                self._active_provider = "ollama"  # Safe default
                # Get default model from provider config (no hardcoding!)
                self._active_model = self.get_default_model("ollama")

            # Trigger discovery
            self._trigger_discovery()

        except Exception as e:
            logger.error(f"Failed to initialize chuk_llm: {e}")
            # Minimal fallback
            self._chuk_config = None
            self._active_provider = "ollama"
            self._active_model = None  # Will be determined on first use

    def _trigger_discovery(self):
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

    def _load_custom_providers(self):
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
                    is_runtime=False,  # These are persisted in preferences
                )
                self._custom_providers[name] = config
                logger.debug(f"Loaded custom provider: {name}")

        except Exception as e:
            logger.warning(f"Failed to load custom providers: {e}")

    # ── Provider Management ───────────────────────────────────────────────────

    def get_available_providers(self) -> List[str]:
        """
        Get list of all available providers.

        Returns:
            List of provider names
        """
        providers = []

        # Get chuk_llm providers
        if self._chuk_config:
            try:
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
        api_key: Optional[str] = None,
        models: Optional[List[str]] = None,
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
                models = discovery_result.models
            else:
                logger.warning(
                    f"Discovery failed for {name}: {discovery_result.error or 'No models found'}"
                )

        # Create the RuntimeProviderConfig
        config = RuntimeProviderConfig(
            name=name,
            api_base=api_base,
            models=models or [],
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

    def get_available_models(self, provider: Optional[str] = None) -> List[str]:
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

        try:
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
                provider_config = self._chuk_config.get_provider(provider)
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

    def refresh_models(self, provider: Optional[str] = None) -> int:
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

    def get_client(
        self, provider: Optional[str] = None, model: Optional[str] = None
    ) -> Any:
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

    def validate_model(self, model: str, provider: Optional[str] = None) -> bool:
        """Check if a model is available for a provider."""
        target_provider = provider or self._active_provider
        if not target_provider:
            return False

        available = self.get_available_models(target_provider)
        return model in available

    # ── Utility Methods ───────────────────────────────────────────────────────

    def __str__(self):
        return f"ModelManager(provider={self._active_provider}, model={self._active_model})"

    def __repr__(self):
        return (
            f"ModelManager(provider='{self._active_provider}', "
            f"model='{self._active_model}', "
            f"cached_clients={self._client_factory.get_cache_size()})"
        )
