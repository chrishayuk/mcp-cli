# mcp_cli/model_manager.py
"""
Enhanced ModelManager that wraps chuk_llm's provider system.
Now properly handles the updated OpenAI client with universal tool compatibility.
"""

import logging
from typing import List, Dict, Any, Tuple

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
        self._initialize_chuk_llm()

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

    def refresh_models(self, provider: str = None):
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

    def refresh_discovery(self, provider: str = None):
        """Refresh discovery for a provider (alias for refresh_models)"""
        return self.refresh_models(provider) > 0

    def get_available_providers(self) -> List[str]:
        """Get list of available providers from chuk_llm"""
        if not self._chuk_config:
            return ["ollama"]  # Safe fallback

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
            available = []

            # Add providers in preferred order
            for provider in preferred_order:
                if provider in all_providers:
                    available.append(provider)

            # Add any other providers not in preferred list
            for provider in all_providers:
                if provider not in available:
                    available.append(provider)

            return available

        except Exception as e:
            logger.error(f"Failed to get available providers: {e}")
            return ["ollama"]  # Safe fallback

    def get_available_models(self, provider: str = None) -> List[str]:
        """Get available models for a provider (including discovered ones)"""
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
                    sorted_models = []
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

            return models

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
        try:
            from chuk_llm.llm.client import list_available_providers

            return list_available_providers()
        except Exception as e:
            logger.error(f"Failed to get detailed provider info: {e}")
            # Fallback to basic info
            basic_info = {}
            for provider in self.get_available_providers():
                try:
                    models = self.get_available_models(provider)
                    has_api_key = (
                        self._chuk_config.get_api_key(provider) is not None
                        if self._chuk_config
                        else False
                    )

                    # CHANGED: Set gpt-oss as default for ollama
                    default_model = (
                        "gpt-oss"
                        if provider == "ollama"
                        else (models[0] if models else None)
                    )

                    basic_info[provider] = {
                        "models": models,
                        "model_count": len(models),
                        "has_api_key": has_api_key,
                        "baseline_features": ["text"],  # Safe default
                        "default_model": default_model,
                    }
                except Exception:
                    basic_info[provider] = {"error": "Could not get provider info"}

            return basic_info

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

    def set_active_model(self, model: str, provider: str = None):
        """Set the active model"""
        target_provider = provider or self._active_provider

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

    def validate_model(self, model: str, provider: str = None) -> bool:
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
                    return default

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

    def get_client(self, provider: str = None, model: str = None):
        """
        Get a chuk_llm client for the specified or active provider/model.
        FIXED: Now uses caching and properly handles the updated OpenAI client.
        """
        try:
            from chuk_llm.llm.client import get_client

            target_provider = provider or self._active_provider
            target_model = model or self._active_model

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

    def get_client_for_provider(self, provider: str, model: str = None):
        """Get a client for a specific provider (alias for get_client)"""
        return self.get_client(provider=provider, model=model)

    def configure_provider(
        self, provider: str, api_key: str = None, api_base: str = None
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

    def get_model_info(self, provider: str = None, model: str = None) -> Dict[str, Any]:
        """Get information about a model"""
        try:
            client = self.get_client(provider, model)
            return client.get_model_info()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)}

    def get_provider_info(self, provider: str) -> Dict[str, Any]:
        """Get information about a provider"""
        try:
            from chuk_llm.llm.client import get_provider_info

            return get_provider_info(provider)
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
            from mcp_cli.cli_options import get_discovery_status

            return get_discovery_status()
        except Exception:
            return {
                "discovery_triggered": self._discovery_triggered,
                "ollama_enabled": True,  # Safe default
            }

    def __str__(self):
        return f"ModelManager(provider={self._active_provider}, model={self._active_model})"

    def __repr__(self):
        return f"ModelManager(provider='{self._active_provider}', model='{self._active_model}', discovery={self._discovery_triggered}, cached_clients={len(self._client_cache)})"
