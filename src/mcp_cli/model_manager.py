# mcp_cli/model_manager.py
"""
Enhanced ModelManager that wraps chuk_llm's provider system.
Now properly handles the updated OpenAI client with universal tool compatibility.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Enhanced ModelManager that wraps chuk_llm's provider system.
    FIXED: Updated to work with the new OpenAI client universal tool compatibility.
    """

    def __init__(self):
        self._chuk_config = None
        self._active_provider = None
        self._active_model = None
        self._discovery_triggered = False
        self._client_cache = {}  # Cache clients to avoid recreation
        self._custom_providers = {}  # Custom OpenAI-compatible providers
        self._runtime_api_keys = {}  # Temporary API keys for runtime providers
        self._initialize_chuk_llm()
        self._load_custom_providers()

    def _initialize_chuk_llm(self):
        """Initialize chuk_llm configuration and trigger discovery"""
        try:
            from chuk_llm.configuration import get_config

            self._chuk_config = get_config()

            # TRIGGER DISCOVERY IMMEDIATELY to get all available models
            self._trigger_discovery()

            # CHANGED: Default to ollama with gpt-oss model
            available_providers = self.get_available_providers()
            if "ollama" in available_providers:
                self._active_provider = "ollama"
                # Check if gpt-oss is available, otherwise try other defaults
                available_models = self.get_available_models("ollama")
                if "gpt-oss" in available_models:
                    self._active_model = "gpt-oss"
                elif "llama3.3" in available_models:
                    self._active_model = "llama3.3"
                elif available_models:
                    # Use first available model
                    self._active_model = available_models[0]
                else:
                    # Fallback to gpt-oss even if not discovered yet
                    self._active_model = "gpt-oss"
                    logger.info("Defaulting to gpt-oss model (may need to be pulled)")
            elif available_providers:
                # Fallback: Use first available provider if ollama not available
                self._active_provider = available_providers[0]
                try:
                    provider_config = self._chuk_config.get_provider(
                        self._active_provider
                    )
                    self._active_model = provider_config.default_model
                except Exception:
                    # Fallback if no default model
                    available_models = self.get_available_models(self._active_provider)
                    self._active_model = (
                        available_models[0] if available_models else "default"
                    )
            else:
                # Hard fallback to ollama/gpt-oss if nothing is configured
                self._active_provider = "ollama"
                self._active_model = "gpt-oss"
                logger.warning("No providers found, defaulting to ollama/gpt-oss")

            logger.debug(
                f"Initialized with provider: {self._active_provider}, model: {self._active_model}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize chuk_llm: {e}")
            # CHANGED: Fallback to ollama/gpt-oss instead of llama3.3
            self._chuk_config = None
            self._active_provider = "ollama"
            self._active_model = "gpt-oss"

    def _load_custom_providers(self):
        """Load custom providers from preferences."""
        try:
            from mcp_cli.utils.preferences import get_preference_manager

            prefs = get_preference_manager()
            custom_providers = prefs.get_custom_providers()

            for name, provider_data in custom_providers.items():
                self._custom_providers[name] = provider_data
                logger.debug(f"Loaded custom provider: {name}")

        except Exception as e:
            logger.warning(f"Failed to load custom providers: {e}")

    def _trigger_discovery(self):
        """Trigger discovery to ensure all models are available"""
        if self._discovery_triggered:
            return

        try:
            # Import discovery functions
            from chuk_llm.api.providers import trigger_ollama_discovery_and_refresh

            logger.debug("ModelManager triggering Ollama discovery...")

            # Trigger discovery for Ollama (most important for local usage)
            new_functions = trigger_ollama_discovery_and_refresh()

            if new_functions:
                logger.info(
                    f"ModelManager discovery: {len(new_functions)} new Ollama functions"
                )
            else:
                logger.debug("ModelManager discovery: no new models found")

            self._discovery_triggered = True

        except Exception as e:
            logger.warning(f"ModelManager discovery failed (continuing anyway): {e}")
            # Don't fail initialization if discovery fails

    def refresh_models(self, provider: str | None = None):
        """Manually refresh models for a provider"""
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

    def refresh_discovery(self, provider: str | None = None):
        """Refresh discovery for a provider (alias for refresh_models)"""
        return self.refresh_models(provider) > 0

    def get_available_providers(self) -> List[str]:
        """Get list of available providers from chuk_llm and custom providers"""
        providers = []

        # Get chuk_llm providers
        if self._chuk_config:
            try:
                # Get all configured providers
                all_providers = self._chuk_config.get_all_providers()

                # CHANGED: Put ollama first in the preferred order
                preferred_order = [
                    "ollama",
                    "openai",
                    "anthropic",
                    "gemini",
                    "groq",
                    "mistral",
                ]

                # Add providers in preferred order
                for provider in preferred_order:
                    if provider in all_providers:
                        providers.append(provider)

                # Add any other providers not in preferred list
                for provider in all_providers:
                    if provider not in providers:
                        providers.append(provider)

            except Exception as e:
                logger.error(f"Failed to get chuk_llm providers: {e}")
                providers = ["ollama"]  # Safe fallback
        else:
            providers = ["ollama"]  # Safe fallback

        # Add custom providers
        for custom_name in self._custom_providers.keys():
            if custom_name not in providers:
                providers.append(custom_name)

        return providers

    def get_available_models(self, provider: str | None = None) -> List[str]:
        """Get available models for a provider (including discovered ones)"""
        target_provider = provider or self._active_provider
        if not target_provider:
            return []

        # Check if it's a custom provider first
        if target_provider in self._custom_providers:
            models = self._custom_providers[target_provider].get(
                "models", ["gpt-4", "gpt-3.5-turbo"]
            )
            return models if isinstance(models, list) else []

        if not self._chuk_config:
            # Return default models even without config
            if provider == "ollama":
                return [
                    "gpt-oss",
                    "llama3.3",
                    "llama3.2",
                    "qwen3",
                    "qwen2.5-coder",
                    "granite3.3",
                    "mistral",
                    "gemma3",
                    "deepseek-coder",
                    "phi3",
                    "codellama",
                ]
            elif provider == "openai":
                return [
                    "gpt-5",
                    "gpt-5-mini",
                    "gpt-5-nano",
                    "gpt-5-chat",
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-4-turbo",
                    "gpt-3.5-turbo",
                    "o3",
                    "o3-mini",
                ]
            elif provider == "anthropic":
                return [
                    "claude-4-1-opus",
                    "claude-4-sonnet",
                    "claude-3-5-sonnet",
                    "claude-3-opus",
                    "claude-3-sonnet",
                    "claude-3-haiku",
                ]
            elif provider == "azure_openai":
                return ["gpt-5", "gpt-5-mini", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
            elif provider == "gemini":
                return ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
            elif provider == "groq":
                return ["llama-3.1-70b", "llama-3.1-8b", "mixtral-8x7b"]
            return []

        target_provider = provider or self._active_provider
        if not target_provider:
            return []

        try:
            from chuk_llm.llm.client import list_available_providers

            # Get all providers with latest info (includes discovered models)
            providers = list_available_providers()
            provider_info = providers.get(target_provider, {})

            if "error" in provider_info:
                logger.warning(
                    f"Provider {target_provider} error: {provider_info['error']}"
                )
                # Return default models for ollama
                if target_provider == "ollama":
                    return [
                        "gpt-oss",
                        "llama3.3",
                        "qwen3",
                        "granite3.3",
                        "mistral",
                        "gemma3",
                    ]
                return []

            # Return all available models (should include discovered ones)
            models = provider_info.get("models", [])

            # Sort models for better UX
            if models:
                # CHANGED: Put gpt-oss first for Ollama, extensive model list
                if target_provider == "ollama":
                    priority_models = [
                        "gpt-oss",
                        "llama3.3",
                        "llama3.2",
                        "qwen3",
                        "qwen2.5-coder",
                        "granite3.3",
                        "mistral",
                        "gemma3",
                        "deepseek-coder",
                    ]
                    sorted_models: List[str] = []
                elif target_provider == "openai":
                    # GPT-5 models first, then GPT-4, then reasoning models
                    priority_models = [
                        "gpt-5",
                        "gpt-5-mini",
                        "gpt-4o",
                        "gpt-4o-mini",
                        "gpt-3.5-turbo",
                        "o3",
                        "o3-mini",
                    ]
                    sorted_models = []
                elif target_provider == "anthropic":
                    # Claude 4 models first
                    priority_models = [
                        "claude-4-1-opus",
                        "claude-4-sonnet",
                        "claude-3-5-sonnet",
                        "claude-3-opus",
                    ]
                    sorted_models = []

                    # Add priority models first (if they exist)
                    for priority in priority_models:
                        if priority in models:
                            sorted_models.append(priority)

                    # Add remaining models
                    for model in models:
                        if model not in sorted_models:
                            sorted_models.append(model)

                    return sorted_models
                else:
                    return sorted(models)

            # CHANGED: Return default models for each provider if no models found
            if target_provider == "ollama":
                return [
                    "gpt-oss",
                    "llama3.3",
                    "llama3.2",
                    "qwen3",
                    "qwen2.5-coder",
                    "granite3.3",
                    "mistral",
                    "gemma3",
                    "deepseek-coder",
                ]
            elif target_provider == "openai":
                return [
                    "gpt-5",
                    "gpt-5-mini",
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-3.5-turbo",
                    "o3",
                    "o3-mini",
                ]
            elif target_provider == "anthropic":
                return [
                    "claude-4-1-opus",
                    "claude-4-sonnet",
                    "claude-3-5-sonnet",
                    "claude-3-opus",
                ]

            return list(models) if models else []

        except Exception as e:
            logger.error(f"Failed to get models for {target_provider}: {e}")
            # Return defaults for each provider
            if target_provider == "ollama":
                return [
                    "gpt-oss",
                    "llama3.3",
                    "llama3.2",
                    "qwen3",
                    "qwen2.5-coder",
                    "granite3.3",
                    "mistral",
                    "gemma3",
                    "deepseek-coder",
                ]
            elif target_provider == "openai":
                return [
                    "gpt-5",
                    "gpt-5-mini",
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-3.5-turbo",
                    "o3",
                    "o3-mini",
                ]
            elif target_provider == "anthropic":
                return [
                    "claude-4-1-opus",
                    "claude-4-sonnet",
                    "claude-3-5-sonnet",
                    "claude-3-opus",
                ]
            return []

    def list_available_providers(self) -> Dict[str, Any]:
        """Get detailed provider information (matches ChukLLM API)"""
        from chuk_llm.llm.client import list_available_providers
        import os

        # Get all providers from chuk_llm
        result = list_available_providers()

        # Enhance all providers with hierarchical token status
        for provider_name in list(result.keys()):
            if "error" not in result[provider_name]:
                # Use hierarchical token resolution
                result[provider_name]["has_api_key"] = self._check_provider_has_api_key(
                    provider_name
                )
                result[provider_name]["token_source"] = self._get_provider_token_source(
                    provider_name
                )

        # Add custom providers
        for name, provider_data in self._custom_providers.items():
            env_var = (
                provider_data.get("env_var_name")
                or f"{name.upper().replace('-', '_')}_API_KEY"
            )
            has_api_key = bool(
                os.environ.get(env_var) or self._runtime_api_keys.get(name)
            )

            result[name] = {
                "models": provider_data.get("models", ["gpt-4", "gpt-3.5-turbo"]),
                "model_count": len(
                    provider_data.get("models", ["gpt-4", "gpt-3.5-turbo"])
                ),
                "has_api_key": has_api_key,
                "token_source": "env"
                if os.environ.get(env_var)
                else ("runtime" if self._runtime_api_keys.get(name) else "none"),
                "baseline_features": [
                    "streaming",
                    "tools",
                    "text",
                ],  # OpenAI-compatible
                "default_model": provider_data.get("default_model", "gpt-4"),
                "api_base": provider_data.get("api_base"),
                "is_custom": True,
            }

        return dict(result)

    def _check_provider_has_api_key(self, provider_name: str) -> bool:
        """
        Check if a provider has an API key using hierarchical resolution.

        Checks: environment variables > token storage

        Args:
            provider_name: Provider name

        Returns:
            True if API key is available, False otherwise
        """
        # Special case: Ollama doesn't need API keys
        if provider_name.lower() == "ollama":
            return True

        try:
            from mcp_cli.auth.provider_tokens import get_provider_token_with_hierarchy
            from mcp_cli.auth import TokenManager

            # Get token manager
            token_manager = TokenManager(service_name="mcp-cli")

            # Check hierarchically (env vars > storage)
            api_key, source = get_provider_token_with_hierarchy(
                provider_name, token_manager
            )
            return api_key is not None

        except Exception as e:
            logger.debug(f"Error checking provider API key: {e}")
            return False

    def _get_provider_token_source(self, provider_name: str) -> str:
        """
        Get the source of a provider's token.

        Args:
            provider_name: Provider name

        Returns:
            'env', 'storage', 'config', or 'none'
        """
        # Special case: Ollama doesn't need API keys
        if provider_name.lower() == "ollama":
            return "none"

        try:
            from mcp_cli.auth.provider_tokens import get_provider_token_with_hierarchy
            from mcp_cli.auth import TokenManager

            token_manager = TokenManager(service_name="mcp-cli")
            _, source = get_provider_token_with_hierarchy(provider_name, token_manager)
            return source

        except Exception:
            return "none"

    def get_active_provider(self) -> str:
        """Get current active provider"""
        return self._active_provider or "ollama"

    def get_active_model(self) -> str:
        """Get current active model"""
        return self._active_model or "gpt-oss"

    def get_active_provider_and_model(self) -> Tuple[str, str]:
        """Get current active provider and model as tuple"""
        return (self.get_active_provider(), self.get_active_model())

    def set_active_provider(self, provider: str):
        """Set the active provider"""
        available = self.get_available_providers()
        if provider not in available:
            raise ValueError(
                f"Provider {provider} not available. Available: {available}"
            )

        self._active_provider = provider

        # Clear client cache when changing provider
        self._client_cache.clear()

        # Set default model for this provider
        try:
            if provider == "ollama":
                # CHANGED: Prefer gpt-oss for ollama
                available_models = self.get_available_models(provider)
                if "gpt-oss" in available_models:
                    self._active_model = "gpt-oss"
                elif available_models:
                    self._active_model = available_models[0]
                else:
                    self._active_model = "gpt-oss"
            elif self._chuk_config:
                provider_config = self._chuk_config.get_provider(provider)
                self._active_model = provider_config.default_model
            else:
                # Fallback: get first available model
                available_models = self.get_available_models(provider)
                self._active_model = (
                    available_models[0] if available_models else "default"
                )
        except Exception as e:
            logger.warning(f"Could not get default model for {provider}: {e}")
            # Fallback: get first available model
            available_models = self.get_available_models(provider)
            self._active_model = available_models[0] if available_models else "default"

    def set_active_model(self, model: str, provider: str | None = None):
        """Set the active model"""

        if provider and provider != self._active_provider:
            self.set_active_provider(provider)

        self._active_model = model

        # Clear client cache when changing model
        self._client_cache.clear()

    def switch_model(self, provider: str, model: str):
        """Switch to a specific provider and model"""
        self.set_active_provider(provider)
        self.set_active_model(model, provider)
        logger.info(f"Switched to {provider}:{model}")

    def switch_provider(self, provider: str):
        """Switch to a provider with its default model"""
        self.set_active_provider(provider)
        logger.info(f"Switched to provider {provider}")

    def switch_to_model(self, model: str):
        """Switch to a model with current provider"""
        self.set_active_model(model)
        logger.info(f"Switched to model {model}")

    def validate_provider(self, provider: str) -> bool:
        """Check if a provider is valid/available"""
        return provider in self.get_available_providers()

    def validate_model(self, model: str, provider: str | None = None) -> bool:
        """Check if a model is available for a provider"""
        target_provider = provider or self._active_provider
        available_models = self.get_available_models(target_provider)
        return model in available_models

    def validate_model_for_provider(self, provider: str, model: str) -> bool:
        """Check if a model is available for a specific provider"""
        return self.validate_model(model, provider)

    def get_default_model(self, provider: str) -> str:
        """Get the default model for a provider"""
        try:
            # CHANGED: Special handling for ollama
            if provider == "ollama":
                available_models = self.get_available_models(provider)
                if "gpt-oss" in available_models:
                    return "gpt-oss"
                elif available_models:
                    return available_models[0]
                else:
                    return "gpt-oss"  # Default even if not pulled

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
            # CHANGED: Special fallback for ollama
            if provider == "ollama":
                return "gpt-oss"
            # Fallback: get first available model
            available_models = self.get_available_models(provider)
            return available_models[0] if available_models else "default"

    def list_providers(self) -> List[str]:
        """Get list of all available providers (alias for get_available_providers)"""
        return self.get_available_providers()

    def get_client(self, provider: str | None = None, model: str | None = None):
        """
        Get a chuk_llm client for the specified or active provider/model.
        FIXED: Now uses caching and properly handles the updated OpenAI client.
        Enhanced: Uses hierarchical token resolution (env vars > storage > config)
        """
        target_provider = provider or self._active_provider
        target_model = model or self._active_model

        # Check if it's a custom provider
        if target_provider in self._custom_providers:
            return self._get_custom_provider_client(target_provider, target_model)

        try:
            from chuk_llm.llm.client import get_client

            # Before creating client, ensure API key is available via hierarchical resolution
            self._ensure_provider_api_key(target_provider)

            # Use cache key to avoid recreating clients
            cache_key = f"{target_provider}:{target_model}"

            if cache_key not in self._client_cache:
                # Create new client with explicit provider and model
                client = get_client(provider=target_provider, model=target_model)
                self._client_cache[cache_key] = client
                logger.debug(f"Created new client for {cache_key}")

            return self._client_cache[cache_key]

        except Exception as e:
            logger.error(
                f"Failed to get client for {target_provider}:{target_model}: {e}"
            )
            raise

    def _ensure_provider_api_key(self, provider_name: str) -> None:
        """
        Ensure provider has API key configured using hierarchical resolution.
        Injects token from storage into chuk_llm config if needed.

        Args:
            provider_name: Provider name
        """
        # Skip for providers that don't need API keys
        if provider_name.lower() == "ollama":
            return

        try:
            from mcp_cli.auth.provider_tokens import get_provider_token_with_hierarchy
            from mcp_cli.auth import TokenManager
            import os

            # Check if env var is already set (highest priority)
            from mcp_cli.auth.provider_tokens import get_provider_env_var_name

            env_var = get_provider_env_var_name(provider_name)
            if os.environ.get(env_var):
                # Env var is set, chuk_llm will use it
                logger.debug(f"Using {env_var} from environment for {provider_name}")
                return

            # Get token from storage
            token_manager = TokenManager(service_name="mcp-cli")
            api_key, source = get_provider_token_with_hierarchy(
                provider_name, token_manager
            )

            if api_key and source == "storage":
                # Inject into chuk_llm config temporarily via environment
                # This ensures chuk_llm can use the stored token
                os.environ[env_var] = api_key
                logger.debug(
                    f"Injected stored token for {provider_name} into environment"
                )

        except Exception as e:
            logger.debug(f"Could not ensure API key for {provider_name}: {e}")

    def _get_custom_provider_client(self, provider: str, model: str | None = None):
        """Get a client for a custom OpenAI-compatible provider."""
        from openai import OpenAI

        provider_data = self._custom_providers.get(provider)
        if not provider_data:
            raise ValueError(f"Custom provider {provider} not found")

        # Use hierarchical token resolution for custom providers too
        from mcp_cli.auth.provider_tokens import get_provider_token_with_hierarchy
        from mcp_cli.auth import TokenManager

        # Check runtime keys first
        api_key = self._runtime_api_keys.get(provider)

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
            env_var = (
                provider_data.get("env_var_name")
                or f"{provider.upper().replace('-', '_')}_API_KEY"
            )
            raise ValueError(
                f"No API key found for provider {provider}. Use 'mcp-cli token set-provider {provider}' or set {env_var}"
            )

        cache_key = f"custom:{provider}:{model or 'default'}"

        if cache_key not in self._client_cache:
            # Create OpenAI-compatible client
            client = OpenAI(api_key=api_key, base_url=provider_data["api_base"])
            self._client_cache[cache_key] = client
            logger.debug(f"Created custom OpenAI client for {provider}")

        return self._client_cache[cache_key]

    def get_client_for_provider(self, provider: str, model: str | None = None):
        """Get a client for a specific provider (alias for get_client)"""
        return self.get_client(provider=provider, model=model)

    def configure_provider(
        self, provider: str, api_key: str | None = None, api_base: str | None = None
    ):
        """Configure a provider with API settings"""
        try:
            if self._chuk_config:
                # Update provider configuration
                provider_config = self._chuk_config.get_provider(provider)
                if api_key:
                    provider_config.api_key = api_key
                if api_base:
                    provider_config.api_base = api_base

                # Clear cache to force recreation with new settings
                self._client_cache.clear()
                logger.info(f"Configured provider {provider}")
        except Exception as e:
            logger.error(f"Failed to configure provider {provider}: {e}")
            raise

    def test_model_access(self, provider: str, model: str) -> bool:
        """Test if a specific model is accessible"""
        try:
            client = self.get_client(provider, model)
            # Try to get model info as a test
            model_info = client.get_model_info()
            return not model_info.get("error")
        except Exception as e:
            logger.debug(f"Model {provider}:{model} not accessible: {e}")
            return False

    def get_model_info(
        self, provider: str | None = None, model: str | None = None
    ) -> Dict[str, Any]:
        """Get information about a model"""
        try:
            client = self.get_client(provider, model)
            info: Dict[str, Any] = client.get_model_info()
            return info
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}

    def get_provider_info(self, provider: str) -> Dict[str, Any]:
        """Get information about a provider"""
        try:
            from chuk_llm.llm.client import get_provider_info

            info: Dict[str, Any] = get_provider_info(provider)
            return info
        except Exception as e:
            logger.error(f"Failed to get provider info for {provider}: {e}")
            return {"error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get current ModelManager status"""
        return {
            "active_provider": self._active_provider,
            "active_model": self._active_model,
            "discovery_triggered": self._discovery_triggered,
            "available_providers": self.get_available_providers(),
            "provider_model_counts": {
                provider: len(self.get_available_models(provider))
                for provider in self.get_available_providers()
            },
            "cached_clients": len(self._client_cache),
        }

    def get_status_summary(self) -> Dict[str, Any]:
        """Get status summary with capability info"""
        try:
            provider_info = self.get_provider_info(self._active_provider)
            supports = provider_info.get("supports", {})

            return {
                "provider": self._active_provider,
                "model": self._active_model,
                "supports_streaming": supports.get("streaming", False),
                "supports_tools": supports.get("tools", False),
                "supports_vision": supports.get("vision", False),
                "supports_json_mode": supports.get("json_mode", False),
            }
        except Exception:
            return {
                "provider": self._active_provider,
                "model": self._active_model,
                "supports_streaming": False,
                "supports_tools": False,
                "supports_vision": False,
                "supports_json_mode": False,
            }

    def get_discovery_status(self) -> Dict[str, Any]:
        """Get discovery status information"""
        try:
            from mcp_cli.config import get_discovery_status

            return get_discovery_status()
        except Exception:
            return {
                "discovery_triggered": self._discovery_triggered,
                "ollama_enabled": True,  # Safe default
            }

    def add_runtime_provider(
        self,
        name: str,
        api_base: str,
        api_key: Optional[str] = None,
        models: Optional[List[str]] = None,
    ) -> None:
        """Add a provider at runtime (not persisted).

        Args:
            name: Provider name
            api_base: API base URL
            api_key: API key (kept in memory only)
            models: List of available models
        """
        self._custom_providers[name] = {
            "name": name,
            "api_base": api_base,
            "models": models or ["gpt-4", "gpt-3.5-turbo"],
            "default_model": models[0] if models else "gpt-4",
            "is_runtime": True,
        }

        if api_key:
            self._runtime_api_keys[name] = api_key

        logger.info(f"Added runtime provider: {name}")

    def is_custom_provider(self, name: str) -> bool:
        """Check if a provider is custom (either from preferences or runtime)."""
        return name in self._custom_providers

    def is_runtime_provider(self, name: str) -> bool:
        """Check if a provider was added at runtime."""
        return name in self._custom_providers and self._custom_providers[name].get(
            "is_runtime", False
        )

    def __str__(self):
        return f"ModelManager(provider={self._active_provider}, model={self._active_model})"

    def __repr__(self):
        return f"ModelManager(provider='{self._active_provider}', model='{self._active_model}', discovery={self._discovery_triggered}, cached_clients={len(self._client_cache)})"
