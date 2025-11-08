# src/mcp_cli/model_management/provider_discovery.py
"""
Provider and model discovery functionality.

This module handles discovering models from OpenAI-compatible APIs
and refreshing model lists for providers.
"""

import logging
from typing import Optional

from mcp_cli.model_management.discovery import DiscoveryResult
from mcp_cli.model_management.provider import RuntimeProviderConfig

logger = logging.getLogger(__name__)


class ProviderDiscovery:
    """
    Handles provider and model discovery operations.

    This class manages discovering available models from APIs and
    refreshing model lists for runtime providers.
    """

    @staticmethod
    def discover_models_from_api(
        api_base: str, api_key: str, provider_name: str = "unknown"
    ) -> DiscoveryResult:
        """
        Discover available models from an OpenAI-compatible API.

        Args:
            api_base: API base URL
            api_key: API key
            provider_name: Provider name for the result

        Returns:
            DiscoveryResult with models or error information
        """
        try:
            import httpx

            # Normalize the API base URL
            base_url = api_base.rstrip("/")

            # Don't add /v1 if it's already there
            if base_url.endswith("/v1"):
                models_url = f"{base_url}/models"
            else:
                # Try with /v1 suffix (OpenAI standard)
                models_url = f"{base_url}/v1/models"

            headers = {"Authorization": f"Bearer {api_key}"}
            logger.debug(f"Discovering models from {models_url}")

            with httpx.Client(timeout=10.0) as client:
                response = client.get(models_url, headers=headers)
                response.raise_for_status()

                data = response.json()
                if "data" in data and isinstance(data["data"], list):
                    models = [model["id"] for model in data["data"] if "id" in model]
                    if models:
                        logger.info(f"Discovered {len(models)} models")
                        return DiscoveryResult(
                            provider=provider_name,
                            models=models,
                            success=True,
                            error=None,
                        )

            return DiscoveryResult(
                provider=provider_name,
                models=[],
                success=False,
                error="No models found in API response",
            )

        except Exception as e:
            logger.debug(f"Model discovery failed: {e}")
            return DiscoveryResult(
                provider=provider_name,
                models=[],
                success=False,
                error=str(e),
            )

    @staticmethod
    def refresh_provider_models(config: RuntimeProviderConfig) -> Optional[int]:
        """
        Refresh models for a runtime provider by querying the API.

        Args:
            config: RuntimeProviderConfig to refresh

        Returns:
            Number of models discovered, or None if failed
        """
        api_key = config.api_key

        if not api_key or not config.api_base:
            logger.warning(f"Cannot refresh {config.name}: missing API key or base URL")
            return None

        try:
            discovery_result = ProviderDiscovery.discover_models_from_api(
                config.api_base, api_key, config.name
            )

            if discovery_result.success and discovery_result.has_models:
                # Update the provider's model list using Pydantic method
                old_count = len(config.models)
                config.set_models(discovery_result.models)

                logger.info(
                    f"Refreshed {config.name}: "
                    f"{discovery_result.discovered_count} models (was {old_count})"
                )
                return discovery_result.discovered_count

            return None

        except Exception as e:
            logger.error(f"Failed to refresh runtime provider {config.name}: {e}")
            return None
