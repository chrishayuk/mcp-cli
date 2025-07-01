# mcp_cli/model_manager.py
"""
Clean ModelManager that fully delegates to chuk-llm 0.8's unified configuration.
FIXED: Added comprehensive model validation to prevent internal inconsistencies.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, List
from pathlib import Path

from chuk_llm.llm.client import get_client, list_available_providers, get_provider_info, validate_provider_setup
from chuk_llm.configuration.unified_config import get_config

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Clean ModelManager that delegates everything to chuk-llm's unified configuration.
    
    Responsibilities:
    - Provide MCP CLI interface to chuk-llm configuration
    - Handle user preference persistence (active provider/model only)
    - Bridge between MCP CLI and chuk-llm APIs
    - FIXED: Comprehensive validation for all model operations
    """

    def __init__(self):
        # Get chuk-llm's configuration manager
        self.chuk_config = get_config()
        
        # Simple user preferences file for active selections
        self.user_prefs_file = Path.home() / ".mcp-cli" / "preferences.yaml"
        self._user_prefs = self._load_user_preferences()
        
        logger.debug("ModelManager initialized with chuk-llm unified configuration")

    def _load_user_preferences(self) -> Dict[str, str]:
        """Load simple user preferences (just active provider/model)."""
        import yaml
        
        if self.user_prefs_file.exists():
            try:
                with open(self.user_prefs_file, 'r') as f:
                    prefs = yaml.safe_load(f) or {}
                    return {
                        "active_provider": prefs.get("active_provider", "openai"),
                        "active_model": prefs.get("active_model", "gpt-4o-mini")
                    }
            except Exception as e:
                logger.warning(f"Failed to load user preferences: {e}")
        
        # Default preferences
        return {
            "active_provider": "openai",
            "active_model": "gpt-4o-mini"
        }

    def _save_user_preferences(self):
        """Save user preferences."""
        import yaml
        
        self.user_prefs_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.user_prefs_file, 'w') as f:
            yaml.dump(self._user_prefs, f, indent=2)

    # ── Active model management ─────────────────────────────────────────
    def get_active_provider(self) -> str:
        """Get currently active provider."""
        return self._user_prefs["active_provider"]

    def get_active_model(self) -> str:
        """Get currently active model."""
        return self._user_prefs["active_model"]

    def get_active_provider_and_model(self) -> tuple[str, str]:
        """Get both active provider and model."""
        return self.get_active_provider(), self.get_active_model()

    def set_active_provider(self, provider: str) -> None:
        """Set active provider with validation."""
        if not self.validate_provider(provider):
            available = ", ".join(self.list_providers())
            raise ValueError(f"Unknown provider: {provider}. Available: {available}")
        
        self._user_prefs["active_provider"] = provider
        
        # Set default model for this provider
        info = self.get_provider_info(provider)
        if info.get("default_model"):
            self._user_prefs["active_model"] = info["default_model"]
        
        self._save_user_preferences()
        logger.info(f"Switched to provider: {provider}")

    def set_active_model(self, model: str) -> None:
        """Set active model."""
        self._user_prefs["active_model"] = model
        self._save_user_preferences()
        logger.info(f"Switched to model: {model}")

    def validate_model_for_provider(self, provider: str, model: str) -> bool:
        """
        FIXED: Validate that a specific model exists for the given provider.
        Returns True if model is valid, False otherwise.
        """
        try:
            available_models = self.get_available_models(provider)
            
            if not available_models:
                logger.warning(f"No models available for provider {provider}")
                return False
            
            # Check if model exists in available models
            model_exists = model in available_models
            
            if not model_exists:
                logger.warning(f"Model '{model}' not available for provider '{provider}'. Available: {available_models[:5]}...")
            
            return model_exists
            
        except Exception as e:
            logger.error(f"Error validating model {model} for provider {provider}: {e}")
            return False

    def switch_model(self, provider: str, model: str) -> None:
        """
        FIXED: Switch to specific provider and model with comprehensive validation.
        """
        # Validate provider first
        if not self.validate_provider(provider):
            available = ", ".join(self.list_providers())
            raise ValueError(f"Unknown provider: {provider}. Available: {available}")
        
        # FIXED: Validate model exists for provider
        if not self.validate_model_for_provider(provider, model):
            available_models = self.get_available_models(provider)
            model_list = ", ".join(available_models[:5])
            if len(available_models) > 5:
                model_list += f"... ({len(available_models)} total)"
            raise ValueError(f"Model '{model}' not available for provider '{provider}'. Available: [{model_list}]")
        
        # If validation passes, proceed with switch
        self.set_active_provider(provider)
        self.set_active_model(model)
        
        logger.info(f"Successfully switched to {provider}/{model}")

    def switch_provider(self, provider: str, model: Optional[str] = None) -> None:
        """
        FIXED: Switch provider with optional model validation.
        """
        if not self.validate_provider(provider):
            available = ", ".join(self.list_providers())
            raise ValueError(f"Unknown provider: {provider}. Available: {available}")
        
        # If model is specified, validate it
        if model:
            if not self.validate_model_for_provider(provider, model):
                available_models = self.get_available_models(provider)
                model_list = ", ".join(available_models[:5])
                if len(available_models) > 5:
                    model_list += f"... ({len(available_models)} total)"
                raise ValueError(f"Model '{model}' not available for provider '{provider}'. Available: [{model_list}]")
        
        self.set_active_provider(provider)
        
        if model:
            self.set_active_model(model)
        else:
            # Set default model for provider
            default_model = self.get_default_model(provider)
            if default_model:
                self.set_active_model(default_model)

    def switch_to_model(self, model: str, provider: Optional[str] = None) -> None:
        """
        FIXED: Switch to specific model with provider validation.
        """
        target_provider = provider or self.get_active_provider()
        
        # Use the fixed switch_model method which includes validation
        self.switch_model(target_provider, model)

    # ── Delegate everything to chuk-llm ────────────────────────────────────
    def list_providers(self) -> List[str]:
        """Get list of available providers."""
        try:
            return self.chuk_config.get_all_providers()
        except Exception as e:
            logger.error(f"Failed to get providers: {e}")
            return []

    def get_client(self, force_refresh: bool = False) -> Any:
        """
        FIXED: Get LLM client with validation before creating client.
        """
        provider = self.get_active_provider()
        model = self.get_active_model()
        
        # Validate current configuration before creating client
        if not self.validate_provider(provider):
            available = ", ".join(self.list_providers())
            raise ValueError(f"Current provider '{provider}' is not valid. Available: {available}")
        
        if not self.validate_model_for_provider(provider, model):
            available_models = self.get_available_models(provider)
            model_list = ", ".join(available_models[:5])
            if len(available_models) > 5:
                model_list += f"... ({len(available_models)} total)"
            raise ValueError(f"Current model '{model}' not available for provider '{provider}'. Available: [{model_list}]")
        
        try:
            return get_client(provider=provider, model=model)
        except Exception as e:
            logger.error(f"Failed to create client for {provider}/{model}: {e}")
            raise

    def get_client_for_provider(self, provider: str, model: Optional[str] = None) -> Any:
        """
        FIXED: Get client for specific provider/model with validation.
        """
        if not self.validate_provider(provider):
            available = ", ".join(self.list_providers())
            raise ValueError(f"Provider '{provider}' is not valid. Available: {available}")
        
        target_model = model or self.get_default_model(provider)
        
        if target_model and not self.validate_model_for_provider(provider, target_model):
            available_models = self.get_available_models(provider)
            model_list = ", ".join(available_models[:5])
            if len(available_models) > 5:
                model_list += f"... ({len(available_models)} total)"
            raise ValueError(f"Model '{target_model}' not available for provider '{provider}'. Available: [{model_list}]")
        
        try:
            return get_client(provider=provider, model=target_model)
        except Exception as e:
            logger.error(f"Failed to create client for {provider}/{target_model}: {e}")
            raise

    def refresh_client(self) -> Any:
        """Force refresh of current client."""
        # chuk-llm handles caching internally, so just return a new client
        return self.get_client()

    def get_provider_info(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive provider information."""
        provider = provider or self.get_active_provider()
        return get_provider_info(provider) or {}

    def validate_provider_setup(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Validate provider setup and configuration."""
        provider = provider or self.get_active_provider()
        return validate_provider_setup(provider)

    def list_available_providers(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed info about all available providers with improved error handling."""
        try:
            return list_available_providers()
        except Exception as e:
            logger.error(f"list_available_providers failed: {e}")
            return {}

    # ── Provider configuration ────────────────────────────────────────────
    def configure_provider(
        self, 
        provider: str, 
        api_key: Optional[str] = None, 
        api_base: Optional[str] = None,
        default_model: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Configure provider by creating/updating user's chuk-llm YAML config.
        
        This creates a providers.yaml file in ~/.chuk_llm/ that extends
        chuk-llm's built-in configuration.
        """
        import yaml
        
        # chuk-llm config directory
        config_dir = Path.home() / ".chuk_llm"
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # User's provider overrides file
        user_config_file = config_dir / "providers.yaml"
        
        # Load existing user config
        user_config = {}
        if user_config_file.exists():
            with open(user_config_file, 'r') as f:
                user_config = yaml.safe_load(f) or {}
        
        # Ensure provider section exists
        if provider not in user_config:
            user_config[provider] = {}
        
        # Update configuration
        if api_base:
            user_config[provider]["api_base"] = api_base
        if default_model:
            user_config[provider]["default_model"] = default_model
        
        # Add other configuration options
        user_config[provider].update(kwargs)
        
        # Handle API key separately in .env file for security
        if api_key:
            self._set_api_key(provider, api_key)
        
        # Save user config
        with open(user_config_file, 'w') as f:
            yaml.dump(user_config, f, indent=2)
        
        # Force chuk-llm to reload configuration
        self.chuk_config.reload()
        
        logger.info(f"Updated configuration for provider: {provider}")

    def _set_api_key(self, provider: str, api_key: str):
        """Set API key in environment file."""
        env_file = Path.home() / ".chuk_llm" / ".env"
        env_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine environment variable name
        # chuk-llm uses standard patterns like OPENAI_API_KEY, ANTHROPIC_API_KEY
        env_var_name = f"{provider.upper()}_API_KEY"
        
        # Update .env file
        lines = []
        if env_file.exists():
            with open(env_file, 'r') as f:
                lines = f.readlines()
        
        # Update existing key or add new one
        key_found = False
        for i, line in enumerate(lines):
            if line.startswith(f"{env_var_name}="):
                lines[i] = f"{env_var_name}={api_key}\n"
                key_found = True
                break
        
        if not key_found:
            lines.append(f"{env_var_name}={api_key}\n")
        
        with open(env_file, 'w') as f:
            f.writelines(lines)
        
        logger.info(f"Updated API key for {provider}")

    # ── Feature and capability queries ────────────────────────────────────
    def supports_feature(self, feature: str, provider: Optional[str] = None, model: Optional[str] = None) -> bool:
        """Check if provider/model supports a feature."""
        provider = provider or self.get_active_provider()
        model = model or self.get_active_model()
        
        info = self.get_provider_info(provider)
        supports = info.get("supports", {})
        
        # If model-specific info is available, use it
        if model and "model_capabilities" in info:
            model_caps = info["model_capabilities"]
            if model in model_caps:
                model_features = model_caps[model]
                return feature in model_features
        
        # Fall back to provider-level support
        return supports.get(feature, False)

    def debug_provider_models(self):
        """Debug method to see what list_available_providers actually returns."""
        try:
            providers_info = list_available_providers()
            
            print("🔍 Debug: list_available_providers() output:")
            for name, info in providers_info.items():
                print(f"\n  Provider: {name}")
                print(f"    All keys: {list(info.keys())}")
                
                # Check all possible model keys
                for key in ["models", "available_models", "model_list", "supported_models"]:
                    if key in info:
                        value = info[key]
                        print(f"    {key}: {type(value)} - {len(value) if isinstance(value, list) else 'not a list'}")
                        if isinstance(value, list) and value:
                            print(f"      First few: {value[:3]}")
                
                # Check if it's an error or configuration issue
                if "error" in info:
                    print(f"    ❌ Error: {info['error']}")
                
                # Check configuration status
                if "has_api_key" in info:
                    print(f"    🔑 Has API key: {info['has_api_key']}")
                    
        except Exception as e:
            print(f"❌ Debug failed: {e}")

    def get_available_models(self, provider: Optional[str] = None) -> List[str]:
        """Get available models for provider with improved discovery."""
        provider = provider or self.get_active_provider()
        
        # First try the standard provider info approach
        info = self.get_provider_info(provider)
        
        # Try multiple possible keys for models (chuk-llm 0.8 might use different keys)
        for key in ["models", "available_models", "model_list", "supported_models"]:
            models = info.get(key, [])
            if models and isinstance(models, list):
                logger.debug(f"Found {len(models)} models for {provider} via key '{key}'")
                return models
        
        # If that fails, try to get models directly from chuk-llm config
        try:
            provider_config = self.chuk_config.get_provider(provider)
            if provider_config and hasattr(provider_config, 'models'):
                models = provider_config.models
                logger.debug(f"Found {len(models)} models for {provider} via direct config access")
                return models
                
        except Exception as e:
            logger.debug(f"Direct config access failed for {provider}: {e}")
        
        # Try alternative approach via list_available_providers with debug
        try:
            all_providers = self.list_available_providers()
            if provider in all_providers:
                provider_info = all_providers[provider]
                
                # Try different model keys from the list_available_providers output
                for key in ["models", "available_models", "model_list", "supported_models"]:
                    if key in provider_info:
                        models = provider_info[key]
                        if isinstance(models, list) and models:
                            logger.debug(f"Found {len(models)} models for {provider} via list_available_providers['{key}']")
                            return models
                            
        except Exception as e:
            logger.debug(f"list_available_providers fallback failed for {provider}: {e}")
        
        # Return empty list if all methods fail
        logger.warning(f"Could not retrieve models for provider {provider}")
        return []

    def get_default_model(self, provider: str) -> str:
        """Get default model for provider."""
        info = self.get_provider_info(provider)
        return info.get("default_model", "")

    def get_model_for_provider(self, provider: str) -> str:
        """Get appropriate model for provider."""
        if provider == self.get_active_provider():
            return self.get_active_model()
        return self.get_default_model(provider)

    # ── Validation methods ────────────────────────────────────────────────
    def validate_provider(self, provider: str) -> bool:
        """Check if provider is known."""
        return provider in self.list_providers()

    def has_api_key(self, provider: Optional[str] = None) -> bool:
        """Check if provider has API key."""
        provider = provider or self.get_active_provider()
        info = self.get_provider_info(provider)
        return info.get("has_api_key", False)

    def is_provider_configured(self, provider: Optional[str] = None) -> bool:
        """Check if provider is properly configured."""
        provider = provider or self.get_active_provider()
        validation = self.validate_provider_setup(provider)
        return validation.get("valid", False)

    def is_current_provider_configured(self) -> bool:
        """Check if current active provider is configured."""
        return self.is_provider_configured()

    # ── Global aliases and advanced features ───────────────────────────────
    def get_global_aliases(self) -> Dict[str, str]:
        """Get global model aliases from chuk-llm configuration."""
        try:
            return self.chuk_config.get_global_aliases()
        except Exception as e:
            logger.debug(f"Failed to get global aliases: {e}")
            return {}

    def add_global_alias(self, alias: str, target: str):
        """Add a global model alias."""
        try:
            self.chuk_config.add_global_alias(alias, target)
        except Exception as e:
            logger.error(f"Failed to add global alias: {e}")

    def find_best_provider_for_request(
        self,
        required_features: Optional[List[str]] = None,
        model_pattern: Optional[str] = None,
        exclude_providers: Optional[List[str]] = None
    ) -> Optional[str]:
        """Find best provider for specific requirements."""
        try:
            from chuk_llm.llm.client import find_best_provider_for_request
            
            result = find_best_provider_for_request(
                required_features=required_features,
                model_pattern=model_pattern,
                exclude_providers=exclude_providers
            )
            
            return result.get("provider") if result else None
        except Exception as e:
            logger.debug(f"find_best_provider_for_request failed: {e}")
            return None

    # ── Context manager and configuration management ─────────────────────────
    def save_config(self) -> None:
        """Save user preferences."""
        self._save_user_preferences()

    def reload_config(self) -> None:
        """Reload chuk-llm configuration."""
        try:
            self.chuk_config.reload()
        except Exception as e:
            logger.error(f"Failed to reload config: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_config()

    # ── Debugging and status methods ──────────────────────────────────────
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary."""
        provider = self.get_active_provider()
        model = self.get_active_model()
        
        return {
            "active_provider": provider,
            "active_model": model,
            "provider_configured": self.is_provider_configured(provider),
            "has_api_key": self.has_api_key(provider),
            "available_providers": self.list_providers(),
            "available_models": self.get_available_models(provider),
            "global_aliases": self.get_global_aliases(),
            "supports_streaming": self.supports_feature("streaming"),
            "supports_tools": self.supports_feature("tools"),
            "supports_vision": self.supports_feature("vision"),
        }

    def test_model_discovery(self):
        """Test method to verify model discovery is working."""
        print("🧪 Testing model discovery...")
        
        # Debug the raw provider output
        self.debug_provider_models()
        
        # Test getting models for each provider
        providers = self.list_providers()
        for provider in providers[:3]:  # Test first 3 providers
            print(f"\n🔍 Testing {provider}:")
            try:
                models = self.get_available_models(provider)
                print(f"  ✅ Found {len(models)} models")
                if models:
                    print(f"  📋 First few: {models[:3]}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        # Test the status summary
        print(f"\n📊 Status Summary:")
        try:
            status = self.get_status_summary()
            active_provider = status["active_provider"]
            model_count = len(status["available_models"])
            print(f"  Active: {active_provider}")
            print(f"  Models for {active_provider}: {model_count}")
        except Exception as e:
            print(f"  ❌ Status summary error: {e}")

    def __repr__(self) -> str:
        return f"ModelManager(provider='{self.get_active_provider()}', model='{self.get_active_model()}')"

    def __str__(self) -> str:
        provider = self.get_active_provider()
        model = self.get_active_model()
        return f"Active: {provider}/{model}"