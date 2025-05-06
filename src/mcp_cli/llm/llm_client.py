# mcp_cli/llm/llm_client.py
"""LLM client factory with improved provider configuration."""
from typing import Optional, Dict, Any
import logging

from mcp_cli.llm.providers.base import BaseLLMClient
from mcp_cli.provider_config import ProviderConfig

log = logging.getLogger(__name__)

def get_llm_client(
    provider: str = "openai", 
    model: Optional[str] = None, 
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    config: Optional[ProviderConfig] = None
) -> BaseLLMClient:
    """
    Get an LLM client for the specified provider.
    
    Args:
        provider: Provider name (e.g., "openai", "ollama")
        model: Model name (e.g., "gpt-4o-mini")
        api_key: API key (overrides configuration)
        api_base: API base URL (overrides configuration)
        config: ProviderConfig instance (created if None)
        
    Returns:
        An instance of BaseLLMClient for the specified provider
    """
    # Load provider configuration
    provider_config = config or ProviderConfig()
    
    try:
        # Get provider-specific configuration
        provider_settings = provider_config.get_provider_config(provider)
        
        # Override with function parameters if provided
        if api_key:
            provider_settings["api_key"] = api_key
        if api_base:
            provider_settings["api_base"] = api_base
        if model:
            provider_settings["model"] = model
        else:
            provider_settings["model"] = provider_settings.get("default_model")
            
        # Instantiate appropriate client based on provider
        if provider.lower() == "openai":
            from mcp_cli.llm.providers.openai_client import OpenAILLMClient
            return OpenAILLMClient(
                model=provider_settings["model"],
                api_key=provider_settings["api_key"],
                api_base=provider_settings["api_base"]
            )
        elif provider.lower() == "ollama":
            from mcp_cli.llm.providers.ollama_client import OllamaLLMClient
            
            # Original OllamaLLMClient doesn't accept api_base in constructor
            # So we'll create the client first
            client = OllamaLLMClient(
                model=provider_settings["model"]
            )
            
            # Then handle the api_base if provided (will be ignored in current implementation)
            if api_base:
                log.warning(
                    f"The current OllamaLLMClient implementation doesn't support api_base. "
                    f"Parameter '{api_base}' will be ignored."
                )
                
                # Try to set the host if the ollama library supports it
                import ollama
                if hasattr(ollama, 'set_host'):
                    log.info(f"Setting Ollama host to: {api_base}")
                    ollama.set_host(api_base)
            
            return client
            
        elif provider.lower() == "anthropic":
            from mcp_cli.llm.providers.anthropic_client import AnthropicLLMClient
            return AnthropicLLMClient(
                model=provider_settings["model"],
                api_key=provider_settings["api_key"],
                api_base=provider_settings["api_base"]
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    except Exception as e:
        raise ValueError(f"Error initializing {provider} client: {e}")