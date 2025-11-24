# src/mcp_cli/core/resolver.py
"""Model and provider resolution utilities."""

from __future__ import annotations

import logging

from mcp_cli.model_management import ModelManager

logger = logging.getLogger(__name__)


class ModelResolver:
    """Handles provider and model resolution logic."""

    def __init__(self, model_manager: ModelManager | None = None):
        """
        Initialize resolver with optional model manager.

        Args:
            model_manager: ModelManager instance (creates one if not provided)
        """
        self.model_manager = model_manager or ModelManager()

    def resolve(
        self, provider: str | None = None, model: str | None = None
    ) -> tuple[str, str]:
        """
        Resolve effective provider and model from user input.

        When only a model is specified, attempts to detect the provider
        by searching all configured providers for that model.

        Args:
            provider: User-specified provider (optional)
            model: User-specified model (optional)

        Returns:
            Tuple of (effective_provider, effective_model)
        """
        if provider and model:
            # Both explicitly specified
            logger.debug(f"Using explicit provider/model: {provider}/{model}")
            return provider, model

        elif provider and not model:
            # Provider specified, get its default model
            default_model = self.model_manager.get_default_model(provider)
            logger.debug(
                f"Using provider with default model: {provider}/{default_model}"
            )
            return provider, default_model

        elif not provider and model:
            # Model specified, try to detect provider
            detected_provider = self.model_manager.detect_provider_for_model(model)

            if detected_provider:
                logger.debug(
                    f"Detected provider '{detected_provider}' for model '{model}'"
                )
                return detected_provider, model
            else:
                # Fall back to current provider if detection fails
                current_provider = self.model_manager.get_active_provider()
                logger.warning(
                    f"Could not detect provider for model '{model}', using current provider '{current_provider}'"
                )
                return current_provider, model

        else:
            # Neither specified, use active configuration
            active_provider = self.model_manager.get_active_provider()
            active_model = self.model_manager.get_active_model()
            logger.debug(
                f"Using active configuration: {active_provider}/{active_model}"
            )
            return active_provider, active_model

    def validate_provider(self, provider: str) -> bool:
        """
        Validate if a provider exists.

        Args:
            provider: Provider name to validate

        Returns:
            True if provider is valid, False otherwise
        """
        return self.model_manager.validate_provider(provider)

    def validate_model(self, model: str, provider: str | None = None) -> bool:
        """
        Validate if a model exists for a provider.

        Args:
            model: Model name to validate
            provider: Provider name (uses active if not specified)

        Returns:
            True if model is valid for the provider, False otherwise
        """
        return self.model_manager.validate_model(model, provider)

    def validate_and_print_error(self, provider: str) -> bool:
        """
        Validate provider and print helpful error message if invalid.

        Args:
            provider: Provider name to validate

        Returns:
            True if valid, False otherwise (with error printed)
        """
        if not self.validate_provider(provider):
            available = ", ".join(self.model_manager.get_available_providers())
            print(f"[red]Error:[/red] Unknown provider: {provider}")
            print(f"[yellow]Available providers:[/yellow] {available}")

            # Check if it might be a provider command
            if provider in ["list", "config", "diagnostic", "set"]:
                print(f"[yellow]Did you mean:[/yellow] mcp-cli provider {provider}")

            return False
        return True

    def get_available_providers(self) -> list[str]:
        """Get list of available providers."""
        return self.model_manager.get_available_providers()

    def get_available_models(self, provider: str | None = None) -> list[str]:
        """Get list of available models for a provider."""
        return self.model_manager.get_available_models(provider)

    def switch_to(
        self, provider: str | None = None, model: str | None = None
    ) -> tuple[str, str]:
        """
        Switch to a provider/model combination and return the result.

        Args:
            provider: Provider to switch to (optional)
            model: Model to switch to (optional)

        Returns:
            Tuple of (active_provider, active_model) after switch
        """
        if provider and model:
            self.model_manager.switch_model(provider, model)
        elif provider:
            self.model_manager.switch_provider(provider)
        elif model:
            # Switch model in current provider
            current_provider = self.model_manager.get_active_provider()
            self.model_manager.switch_model(current_provider, model)

        return (
            self.model_manager.get_active_provider(),
            self.model_manager.get_active_model(),
        )

    def configure_provider(
        self,
        provider: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        """
        Configure a provider with API settings.

        Args:
            provider: Provider name
            api_key: API key (optional)
            api_base: API base URL (optional)
        """
        self.model_manager.add_runtime_provider(
            name=provider, api_key=api_key, api_base=api_base or ""
        )

    def get_status(self) -> dict:
        """Get current resolver status."""
        return {
            "active_provider": self.model_manager.get_active_provider(),
            "active_model": self.model_manager.get_active_model(),
            "available_providers": self.get_available_providers(),
            "provider_model_counts": {
                provider: len(self.get_available_models(provider))
                for provider in self.get_available_providers()
            },
        }
