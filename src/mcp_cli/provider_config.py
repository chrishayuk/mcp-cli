# mcp_cli/provider_config.py
"""Provider configuration management for MCP CLI."""
import os
import json
from pathlib import Path
from typing import Dict, Optional, Any

class ProviderConfig:
    """Manages provider configuration including API keys and base URLs."""
    
    DEFAULT_CONFIG_PATH = "~/.mcp-cli/providers.json"
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = os.path.expanduser(config_path or self.DEFAULT_CONFIG_PATH)
        self.providers = self._load_providers()
        
    def _load_providers(self) -> Dict[str, Dict[str, Any]]:
        """Load provider configurations from file or create defaults."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return self._create_default_config()
        else:
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Dict[str, Any]]:
        """Create default provider configurations."""
        config = {
            "openai": {
                "api_key_env": "OPENAI_API_KEY",
                "api_key": None,
                "api_base": None,
                "default_model": "gpt-4o-mini"
            },
            "ollama": {
                "api_key": None,
                "api_base": "http://localhost:11434",
                "default_model": "llama3.2"
            },
            "anthropic": {
                "api_key_env": "ANTHROPIC_API_KEY",
                "api_key": None,
                "api_base": None,
                "default_model": "claude-3-opus-20240229"
            }
        }
        return config
    
    def save_config(self) -> None:
        """Save the current configuration to disk."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            json.dump(self.providers, f, indent=2)
    
    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """Get configuration for a specific provider."""
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not configured")
        
        provider_config = self.providers[provider_name].copy()
        
        # Try to get API key from environment if not set directly
        if not provider_config.get("api_key") and provider_config.get("api_key_env"):
            provider_config["api_key"] = os.environ.get(provider_config["api_key_env"])
            
        return provider_config
    
    def set_provider_config(self, provider_name: str, config: Dict[str, Any]) -> None:
        """Update configuration for a provider."""
        if provider_name not in self.providers:
            self.providers[provider_name] = {}
            
        # Update config but preserve existing keys
        self.providers[provider_name].update(config)
        
        # Save changes
        self.save_config()
        
    def get_api_key(self, provider_name: str) -> Optional[str]:
        """Get API key for a provider."""
        config = self.get_provider_config(provider_name)
        return config.get("api_key")
    
    def get_api_base(self, provider_name: str) -> Optional[str]:
        """Get API base URL for a provider."""
        config = self.get_provider_config(provider_name)
        return config.get("api_base")
    
    def get_default_model(self, provider_name: str) -> str:
        """Get default model for a provider."""
        config = self.get_provider_config(provider_name)
        return config.get("default_model", "")