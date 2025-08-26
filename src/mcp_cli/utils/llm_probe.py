# src/mcp_cli/utils/llm_probe.py
"""
LLM Provider/Model availability testing utility.

This module provides utilities for testing whether a provider/model combination
is available and working before committing to configuration changes.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Any, Optional
from dataclasses import dataclass

from mcp_cli.model_manager import ModelManager  # ← CHANGED


@dataclass
class ProbeResult:
    """Result of a provider/model availability probe."""

    success: bool
    error_message: Optional[str] = None
    client: Optional[Any] = None
    response: Optional[Dict[str, Any]] = None


class LLMProbe:
    """Utility class for testing LLM provider/model availability."""

    def __init__(
        self, model_manager: ModelManager, suppress_logging: bool = True
    ):  # ← CHANGED
        """
        Initialize the probe utility.

        Args:
            model_manager: Model manager instance
            suppress_logging: Whether to suppress chuk_llm logging during probes
        """
        self.model_manager = model_manager  # ← CHANGED
        self.suppress_logging = suppress_logging
        self._original_log_level: Optional[int] = None

    def __enter__(self):
        """Context manager entry - suppress logging if requested."""
        if self.suppress_logging:
            chuk_logger = logging.getLogger("chuk_llm")
            self._original_log_level = chuk_logger.level
            chuk_logger.setLevel(logging.CRITICAL)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore logging level."""
        if self.suppress_logging and self._original_log_level is not None:
            chuk_logger = logging.getLogger("chuk_llm")
            chuk_logger.setLevel(self._original_log_level)

    async def __aenter__(self):
        """Async context manager entry - suppress logging if requested."""
        if self.suppress_logging:
            chuk_logger = logging.getLogger("chuk_llm")
            self._original_log_level = chuk_logger.level
            chuk_logger.setLevel(logging.CRITICAL)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - restore logging level."""
        if self.suppress_logging and self._original_log_level is not None:
            chuk_logger = logging.getLogger("chuk_llm")
            chuk_logger.setLevel(self._original_log_level)

    async def test_provider_model(
        self, provider: str, model: str, test_message: str = "ping"
    ) -> ProbeResult:
        """
        Test if a provider/model combination is available and working.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            model: Model name (e.g., 'gpt-4', 'claude-3-sonnet')
            test_message: Message to send for testing (default: "ping")

        Returns:
            ProbeResult with success status, error message, and client if successful
        """
        try:
            # Create client using ModelManager's client creation method
            client = self.model_manager.get_client_for_provider(provider, model)

            # Test with a simple completion
            response = await client.create_completion(
                [{"role": "user", "content": test_message}]
            )

            # Validate the response
            if self._is_valid_response(response):
                return ProbeResult(success=True, client=client, response=response)
            else:
                error_msg = self._extract_error_message(response)
                return ProbeResult(
                    success=False, error_message=error_msg, response=response
                )

        except Exception as exc:
            return ProbeResult(success=False, error_message=str(exc))

    async def test_model(self, model: str, test_message: str = "ping") -> ProbeResult:
        """
        Test if a model is available with the current active provider.

        Args:
            model: Model name to test
            test_message: Message to send for testing

        Returns:
            ProbeResult with success status and details
        """
        provider = self.model_manager.get_active_provider()  # ← CHANGED
        return await self.test_provider_model(provider, model, test_message)

    async def test_provider(
        self, provider: str, test_message: str = "ping"
    ) -> ProbeResult:
        """
        Test if a provider is available with its default model.

        Args:
            provider: Provider name to test
            test_message: Message to send for testing

        Returns:
            ProbeResult with success status and details
        """
        try:
            # Validate provider exists in configuration
            self.model_manager.get_provider_config(provider)  # ← CHANGED
            model = self.model_manager.get_default_model(provider)  # ← CHANGED
            return await self.test_provider_model(provider, model, test_message)
        except ValueError as e:
            return ProbeResult(success=False, error_message=str(e))

    def _is_valid_response(self, response: Any) -> bool:
        """
        Check if a response indicates successful communication.

        Args:
            response: Response from create_completion

        Returns:
            True if response indicates success, False otherwise
        """
        return (
            isinstance(response, dict)
            and not response.get("error", False)
            and isinstance(response.get("response"), str)
            and response["response"].strip()
            and not response["response"].strip().lower().startswith("error")
        )

    def _extract_error_message(self, response: Any) -> str:
        """
        Extract a clean, user-friendly error message from a failed response.

        Args:
            response: Failed response from create_completion

        Returns:
            Clean error message string
        """
        if not isinstance(response, dict):
            return "Invalid response format"

        response_text = response.get("response", "")
        if not response_text:
            return "Provider returned empty response"

        # Try to extract meaningful error from structured error responses
        if "Error code:" in response_text and "message" in response_text:
            try:
                # Extract error message from JSON-like structure
                match = re.search(r"'message': '([^']+)'", response_text)
                if match:
                    return match.group(1)

                # Fallback: extract error code
                code_match = re.search(r"Error code: (\d+)", response_text)
                if code_match:
                    return f"HTTP {code_match.group(1)} error - check model availability or authentication"

            except Exception:
                pass

        # Fallback: return the response text (might be verbose but informative)
        return response_text


# Convenience functions for common use cases
async def test_model_availability(
    model: str,
    model_manager: ModelManager,  # ← CHANGED
    suppress_logging: bool = True,
) -> ProbeResult:
    """
    Quick function to test if a model is available with current provider.

    Args:
        model: Model name to test
        model_manager: Model manager instance
        suppress_logging: Whether to suppress internal logging

    Returns:
        ProbeResult indicating success/failure
    """
    async with LLMProbe(model_manager, suppress_logging) as probe:
        return await probe.test_model(model)


async def test_provider_availability(
    provider: str,
    model_manager: ModelManager,  # ← CHANGED
    suppress_logging: bool = True,
) -> ProbeResult:
    """
    Quick function to test if a provider is available with its default model.

    Args:
        provider: Provider name to test
        model_manager: Model manager instance
        suppress_logging: Whether to suppress internal logging

    Returns:
        ProbeResult indicating success/failure
    """
    async with LLMProbe(model_manager, suppress_logging) as probe:
        return await probe.test_provider(provider)
