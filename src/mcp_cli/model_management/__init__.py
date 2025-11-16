# src/mcp_cli/model_management/__init__.py
"""
Model management package for mcp-cli.

This package provides clean, type-safe LLM provider and model management:
- Pydantic models for provider configuration
- Model discovery from APIs
- Client creation and caching
- ModelManager for orchestrating everything

NO HARDCODED MODELS - All data comes from configuration or discovery.
"""

from mcp_cli.model_management.provider import (
    RuntimeProviderConfig,
    ProviderCapabilities,
)
from mcp_cli.model_management.discovery import DiscoveryResult
from mcp_cli.model_management.client_factory import ClientFactory
from mcp_cli.model_management.provider_discovery import ProviderDiscovery
from mcp_cli.model_management.model_manager import ModelManager

__all__ = [
    "RuntimeProviderConfig",
    "ProviderCapabilities",
    "DiscoveryResult",
    "ClientFactory",
    "ProviderDiscovery",
    "ModelManager",
]
