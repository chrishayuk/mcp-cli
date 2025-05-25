# mcp_cli/model_manager.py
"""
Complete model management system for MCP-CLI.

Handles all model-related operations including LLM client creation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, List

# Direct import from chuk-llm (no wrapper needed)
from chuk_llm.llm.llm_client import get_llm_client as _chuk_get_llm_client
from chuk_llm.llm.configuration.provider_config import DEFAULTS as CHUK_DEFAULTS

DEFAULTS: Dict[str, Dict[str, Any]] = CHUK_DEFAULTS
CFG_PATH = Path(os.path.expanduser("~/.mcp-cli/models.json"))


class ModelManager:
    """
    Complete model management system for MCP-CLI.

    Handles:
    - Provider/model configuration and persistence
    - Active model selection and switching  
    - LLM client creation and management
    - API configuration (keys, base URLs)
    - Model validation and testing
    """

    def __init__(self, config_path: str | None = None) -> None:
        self._path = Path(os.path.expanduser(config_path)) if config_path else CFG_PATH
        self.providers: Dict[str, Dict[str, Any]] = self._load_and_sync()
        self._cached_client: Optional[Any] = None
        self._cached_provider: Optional[str] = None
        self._cached_model: Optional[str] = None

    # ── Core configuration management ─────────────────────────────────────
    def _load_and_sync(self) -> Dict[str, Dict[str, Any]]:
        """Load and sync configuration with defaults."""
        on_disk: Dict[str, Dict[str, Any]] = {}
        if self._path.is_file():
            try:
                on_disk = json.loads(self._path.read_text())
            except json.JSONDecodeError:
                pass

        # Deep-copy defaults then overlay user config
        merged: Dict[str, Dict[str, Any]] = json.loads(json.dumps(DEFAULTS))
        for prov, cfg in on_disk.items():
            merged.setdefault(prov, {}).update(cfg)

        # Prune legacy keys
        for prov_cfg in merged.values():
            prov_cfg.pop("client", None)

        # Write back if changed
        if merged != on_disk:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(merged, indent=2))

        return merged

    def _save(self) -> None:
        """Save configuration to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self.providers, indent=2))

    def _ensure_section(self, name: str) -> None:
        """Ensure provider section exists."""
        if name not in self.providers:
            self.providers[name] = {}

    def _merge_env_key(self, cfg: Dict[str, Any]) -> None:
        """Merge environment variable API keys."""
        if not cfg.get("api_key") and (env := cfg.get("api_key_env")):
            cfg["api_key"] = os.getenv(env)

    def _invalidate_client_cache(self) -> None:
        """Invalidate cached client when model changes."""
        self._cached_client = None
        self._cached_provider = None
        self._cached_model = None

    # ── Provider configuration ────────────────────────────────────────────
    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """Get complete configuration for a provider."""
        if provider not in DEFAULTS and provider not in self.providers:
            raise ValueError(f"Unknown provider: {provider}")
        self._ensure_section(provider)
        cfg = {**DEFAULTS.get(provider, {}), **self.providers[provider]}
        self._merge_env_key(cfg)
        return cfg

    def set_provider_config(self, provider: str, updates: Dict[str, Any]) -> None:
        """Update configuration for a provider."""
        self._ensure_section(provider)
        self.providers[provider].update(updates)
        self._invalidate_client_cache()
        self._save()

    def configure_provider(
        self, 
        provider: str, 
        api_key: Optional[str] = None, 
        api_base: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Configure a provider with API settings.
        
        Args:
            provider: Provider name
            api_key: API key
            api_base: API base URL  
            **kwargs: Additional provider-specific settings
        """
        config_updates = {}
        
        if api_key is not None:
            config_updates["api_key"] = api_key
        if api_base is not None:
            config_updates["api_base"] = api_base
            
        config_updates.update(kwargs)
        
        if config_updates:
            self.set_provider_config(provider, config_updates)

    def list_providers(self) -> List[str]:
        """Get list of available providers."""
        providers = set(DEFAULTS.keys())
        providers.update(self.providers.keys())
        providers.discard("__global__")
        return sorted(providers)

    # ── Active model management ───────────────────────────────────────────
    @property
    def _global_config(self) -> Dict[str, Any]:
        """Get global configuration section."""
        self._ensure_section("__global__")
        return self.providers["__global__"]

    def get_active_provider(self) -> str:
        """Get currently active provider."""
        return self._global_config.get("active_provider", DEFAULTS["__global__"]["active_provider"])

    def get_active_model(self) -> str:
        """Get currently active model."""
        return self._global_config.get("active_model", DEFAULTS["__global__"]["active_model"])

    def get_active_provider_and_model(self) -> tuple[str, str]:
        """Get both active provider and model."""
        return self.get_active_provider(), self.get_active_model()

    def set_active_provider(self, provider: str) -> None:
        """Set active provider."""
        if provider not in self.list_providers():
            raise ValueError(f"Unknown provider: {provider}")
        self._global_config["active_provider"] = provider
        self._invalidate_client_cache()
        self._save()

    def set_active_model(self, model: str) -> None:
        """Set active model."""
        self._global_config["active_model"] = model
        self._invalidate_client_cache()
        self._save()

    def switch_model(self, provider: str, model: str) -> None:
        """
        Switch to specific provider and model.
        
        Args:
            provider: Target provider
            model: Target model
        """
        if provider not in self.list_providers():
            raise ValueError(f"Unknown provider: {provider}")
            
        global_cfg = self._global_config
        global_cfg["active_provider"] = provider
        global_cfg["active_model"] = model
        self._invalidate_client_cache()
        self._save()

    def switch_provider(self, provider: str, model: Optional[str] = None) -> None:
        """
        Switch provider, optionally specifying model.
        
        Args:
            provider: Target provider
            model: Target model (uses provider default if None)
        """
        if model is None:
            model = self.get_default_model(provider)
        self.switch_model(provider, model)

    def switch_to_model(self, model: str, provider: Optional[str] = None) -> None:
        """
        Switch to specific model, optionally changing provider.
        
        Args:
            model: Target model
            provider: Target provider (uses current if None)
        """
        if provider is None:
            provider = self.get_active_provider()
        self.switch_model(provider, model)

    # ── Model information ─────────────────────────────────────────────────
    def get_default_model(self, provider: str) -> str:
        """Get default model for a provider."""
        return self.get_provider_config(provider).get("default_model", "")

    def get_model_for_provider(self, provider: str) -> str:
        """
        Get appropriate model for a provider.
        
        If provider is currently active, returns active model.
        Otherwise returns provider's default model.
        """
        if provider == self.get_active_provider():
            return self.get_active_model()
        return self.get_default_model(provider)

    # ── LLM Client Management ─────────────────────────────────────────────
    def get_client(self, force_refresh: bool = False) -> Any:
        """
        Get LLM client for current active provider/model.
        
        Args:
            force_refresh: Force creation of new client
            
        Returns:
            LLM client instance
        """
        current_provider = self.get_active_provider()
        current_model = self.get_active_model()
        
        # Return cached client if available and unchanged
        if (not force_refresh and 
            self._cached_client is not None and 
            self._cached_provider == current_provider and 
            self._cached_model == current_model):
            return self._cached_client
        
        # Create new client directly with chuk-llm
        client = _chuk_get_llm_client(
            provider=current_provider,
            model=current_model,
            config=self  # ModelManager implements same interface as ProviderConfig
        )
        
        # Cache the client
        self._cached_client = client
        self._cached_provider = current_provider
        self._cached_model = current_model
        
        return client

    def get_client_for_provider(self, provider: str, model: Optional[str] = None) -> Any:
        """
        Get LLM client for specific provider/model without changing active selection.
        
        Args:
            provider: Target provider
            model: Target model (uses provider default if None)
            
        Returns:
            LLM client instance
        """
        if model is None:
            model = self.get_default_model(provider)
            
        return _chuk_get_llm_client(
            provider=provider,
            model=model,
            config=self
        )

    def refresh_client(self) -> Any:
        """Force refresh of current client and return it."""
        return self.get_client(force_refresh=True)

    # ── Convenience getters ───────────────────────────────────────────────
    def get_api_key(self, provider: Optional[str] = None) -> Optional[str]:
        """Get API key for provider (current provider if None)."""
        provider = provider or self.get_active_provider()
        return self.get_provider_config(provider).get("api_key")

    def get_api_base(self, provider: Optional[str] = None) -> Optional[str]:
        """Get API base for provider (current provider if None)."""
        provider = provider or self.get_active_provider()
        return self.get_provider_config(provider).get("api_base")

    # ── Validation ────────────────────────────────────────────────────────
    def validate_provider(self, provider: str) -> bool:
        """Check if provider is known."""
        try:
            self.get_provider_config(provider)
            return True
        except ValueError:
            return False

    def has_api_key(self, provider: Optional[str] = None) -> bool:
        """Check if provider has API key."""
        provider = provider or self.get_active_provider()
        return bool(self.get_api_key(provider))

    def is_provider_configured(self, provider: Optional[str] = None) -> bool:
        """Check if provider is properly configured."""
        provider = provider or self.get_active_provider()
        return self.validate_provider(provider) and self.has_api_key(provider)

    def is_current_provider_configured(self) -> bool:
        """Check if current active provider is configured."""
        return self.is_provider_configured()

    # ── Context manager ───────────────────────────────────────────────────
    def save_config(self) -> None:
        """Explicitly save configuration."""
        self._save()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._save()
