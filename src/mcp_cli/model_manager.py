# mcp_cli/model_manager.py - Enhanced with all missing methods
"""
Enhanced ModelManager that wraps chuk_llm's provider system.
Provides comprehensive interface with discovery capabilities and prettier displays.
"""
import logging
from typing import List, Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Enhanced ModelManager that wraps chuk_llm's provider system.
    Provides mcp-cli specific interface while using chuk_llm under the hood.
    """
    
    def __init__(self):
        self._chuk_config = None
        self._active_provider = None
        self._active_model = None
        self._discovery_triggered = False
        self._initialize_chuk_llm()
    
    def _initialize_chuk_llm(self):
        """Initialize chuk_llm configuration and trigger discovery"""
        try:
            from chuk_llm.configuration import get_config
            self._chuk_config = get_config()
            
            # TRIGGER DISCOVERY IMMEDIATELY to get all available models
            self._trigger_discovery()
            
            # Set default provider to ollama if available
            available_providers = self.get_available_providers()
            if 'ollama' in available_providers:
                self._active_provider = 'ollama'
                # Get default model for ollama
                try:
                    ollama_provider = self._chuk_config.get_provider('ollama')
                    self._active_model = ollama_provider.default_model
                except Exception:
                    # Fallback if no default model configured
                    available_models = self.get_available_models('ollama')
                    self._active_model = available_models[0] if available_models else 'llama3.3'
            elif available_providers:
                # Use first available provider
                self._active_provider = available_providers[0]
                try:
                    provider_config = self._chuk_config.get_provider(self._active_provider)
                    self._active_model = provider_config.default_model
                except Exception:
                    # Fallback if no default model
                    available_models = self.get_available_models(self._active_provider)
                    self._active_model = available_models[0] if available_models else 'default'
            
            logger.debug(f"Initialized with provider: {self._active_provider}, model: {self._active_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize chuk_llm: {e}")
            # Fallback to empty state
            self._chuk_config = None
            self._active_provider = 'ollama'  # Safe fallback
            self._active_model = 'llama3.3'   # Safe fallback
    
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
                logger.info(f"ModelManager discovery: {len(new_functions)} new Ollama functions")
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
            return ['ollama']  # Safe fallback
        
        try:
            # Get all configured providers
            all_providers = self._chuk_config.get_all_providers()
            
            # Filter to commonly used providers
            preferred_order = ['ollama', 'openai', 'anthropic', 'gemini', 'groq', 'mistral']
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
            return ['ollama']  # Safe fallback
    
    def get_available_models(self, provider: str = None) -> List[str]:
        """Get available models for a provider (including discovered ones)"""
        if not self._chuk_config:
            return []
        
        target_provider = provider or self._active_provider
        if not target_provider:
            return []
        
        try:
            from chuk_llm.llm.client import list_available_providers
            
            # Get all providers with latest info (includes discovered models)
            providers = list_available_providers()
            provider_info = providers.get(target_provider, {})
            
            if 'error' in provider_info:
                logger.warning(f"Provider {target_provider} error: {provider_info['error']}")
                return []
            
            # Return all available models (should include discovered ones)
            models = provider_info.get('models', [])
            
            # Sort models for better UX
            if models:
                # Put common models first for Ollama
                if target_provider == 'ollama':
                    priority_models = ['llama3.3', 'qwen3', 'granite3.3', 'mistral', 'gemma3']
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
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to get models for {target_provider}: {e}")
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
                    has_api_key = self._chuk_config.get_api_key(provider) is not None if self._chuk_config else False
                    
                    basic_info[provider] = {
                        "models": models,
                        "model_count": len(models),
                        "has_api_key": has_api_key,
                        "baseline_features": ["text"],  # Safe default
                        "default_model": models[0] if models else None
                    }
                except Exception:
                    basic_info[provider] = {"error": "Could not get provider info"}
            
            return basic_info
    
    def get_active_provider(self) -> str:
        """Get current active provider"""
        return self._active_provider or "ollama"
    
    def get_active_model(self) -> str:
        """Get current active model"""
        return self._active_model or "llama3.3"
    
    def get_active_provider_and_model(self) -> Tuple[str, str]:
        """Get current active provider and model as tuple"""
        return (self.get_active_provider(), self.get_active_model())
    
    def set_active_provider(self, provider: str):
        """Set the active provider"""
        available = self.get_available_providers()
        if provider not in available:
            raise ValueError(f"Provider {provider} not available. Available: {available}")
        
        self._active_provider = provider
        
        # Set default model for this provider
        try:
            if self._chuk_config:
                provider_config = self._chuk_config.get_provider(provider)
                self._active_model = provider_config.default_model
            else:
                # Fallback: get first available model
                available_models = self.get_available_models(provider)
                self._active_model = available_models[0] if available_models else 'default'
        except Exception as e:
            logger.warning(f"Could not get default model for {provider}: {e}")
            # Fallback: get first available model
            available_models = self.get_available_models(provider)
            self._active_model = available_models[0] if available_models else 'default'
    
    def set_active_model(self, model: str, provider: str = None):
        """Set the active model"""
        target_provider = provider or self._active_provider
        
        # Check if model is available
        available_models = self.get_available_models(target_provider)
        # if available_models and model not in available_models:
        #     logger.warning(f"Model {model} not in available models for {target_provider}")
        #     # Don't raise error - let chuk_llm handle discovery if needed
        
        if provider and provider != self._active_provider:
            self.set_active_provider(provider)
        
        self._active_model = model
    
    def switch_model(self, provider: str, model: str):
        """Switch to a specific provider and model"""
        self.set_active_provider(provider)
        self.set_active_model(model, provider)
        logger.info(f"Switched to {provider}:{model}")
    
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
            if self._chuk_config:
                provider_config = self._chuk_config.get_provider(provider)
                default = provider_config.default_model
                if default:
                    return default
            
            # Fallback: get first available model
            available_models = self.get_available_models(provider)
            return available_models[0] if available_models else 'default'
            
        except Exception as e:
            logger.warning(f"Could not get default model for {provider}: {e}")
            # Fallback: get first available model
            available_models = self.get_available_models(provider)
            return available_models[0] if available_models else 'default'
    
    def list_providers(self) -> List[str]:
        """Get list of all available providers (alias for get_available_providers)"""
        return self.get_available_providers()
    
    def get_client(self, provider: str = None, model: str = None):
        """Get a chuk_llm client for the specified or active provider/model"""
        try:
            from chuk_llm.llm.client import get_client
            
            target_provider = provider or self._active_provider
            target_model = model or self._active_model
            
            return get_client(target_provider, model=target_model)
            
        except Exception as e:
            logger.error(f"Failed to get client for {target_provider}:{target_model}: {e}")
            raise
    
    def test_model_access(self, provider: str, model: str) -> bool:
        """Test if a specific model is accessible"""
        try:
            client = self.get_client(provider, model)
            # Try to get model info as a test
            model_info = client.get_model_info()
            return not model_info.get('error')
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
            }
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
                "supports_json_mode": supports.get("json_mode", False)
            }
        except Exception:
            return {
                "provider": self._active_provider,
                "model": self._active_model,
                "supports_streaming": False,
                "supports_tools": False, 
                "supports_vision": False,
                "supports_json_mode": False
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
        return f"ModelManager(provider='{self._active_provider}', model='{self._active_model}', discovery={self._discovery_triggered})"