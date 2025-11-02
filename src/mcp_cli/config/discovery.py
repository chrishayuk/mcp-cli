"""
ChukLLM discovery and provider management.

This module handles the discovery and validation of ChukLLM providers and models.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Global flags to ensure we only set up once
_ENV_SETUP_COMPLETE = False
_DISCOVERY_TRIGGERED = False


def setup_chuk_llm_environment() -> None:
    """
    Set up environment variables for ChukLLM discovery.
    MUST be called before any chuk_llm imports.
    """
    global _ENV_SETUP_COMPLETE

    if _ENV_SETUP_COMPLETE:
        return

    # Set environment variables (only if not already set by user)
    env_vars = {
        "CHUK_LLM_DISCOVERY_ENABLED": "true",
        "CHUK_LLM_AUTO_DISCOVER": "true",
        "CHUK_LLM_DISCOVERY_ON_STARTUP": "true",
        "CHUK_LLM_DISCOVERY_TIMEOUT": "10",
        "CHUK_LLM_OLLAMA_DISCOVERY": "true",
        "CHUK_LLM_OPENAI_DISCOVERY": "true",
        "CHUK_LLM_OPENAI_TOOL_COMPATIBILITY": "true",
        "CHUK_LLM_UNIVERSAL_TOOLS": "true",
    }

    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value

    _ENV_SETUP_COMPLETE = True
    logger.debug("ChukLLM environment variables set")


def trigger_discovery_after_setup() -> int:
    """
    Trigger discovery after environment setup.
    Call this after setup_chuk_llm_environment() and before using models.

    Returns:
        Number of new functions discovered
    """
    global _DISCOVERY_TRIGGERED

    if _DISCOVERY_TRIGGERED:
        return 0

    try:
        # Import discovery functions
        from chuk_llm.api.providers import trigger_ollama_discovery_and_refresh

        logger.debug("Triggering Ollama discovery from cli_options...")

        # Trigger Ollama discovery to get all available models
        new_functions = trigger_ollama_discovery_and_refresh()

        _DISCOVERY_TRIGGERED = True

        if new_functions:
            logger.debug(f"CLI discovery: {len(new_functions)} new Ollama functions")
        else:
            logger.debug("CLI discovery: no new functions (may already be cached)")

        return len(new_functions)

    except Exception as e:
        logger.debug(f"CLI discovery failed: {e}")
        return 0


def get_available_models_quick(provider: str = "ollama") -> List[str]:
    """
    Quick function to get available models after discovery.

    Args:
        provider: Provider name (default: "ollama")

    Returns:
        List of available model names
    """
    try:
        from chuk_llm.llm.client import list_available_providers

        providers = list_available_providers()
        provider_info = providers.get(provider, {})
        models = provider_info.get("models", [])
        return list(models)  # Ensure it's a list
    except Exception as e:
        logger.debug(f"Could not get models for {provider}: {e}")
        return []


def validate_provider_exists(provider: str) -> bool:
    """
    Validate provider exists, potentially after discovery.

    Args:
        provider: Provider name to validate

    Returns:
        True if provider exists, False otherwise
    """
    try:
        from chuk_llm.configuration import get_config

        config = get_config()
        config.get_provider(provider)  # This will raise if not found
        return True
    except Exception:
        return False


def get_discovery_status() -> Dict[str, Any]:
    """
    Get discovery status for debugging.

    Returns:
        Dictionary with discovery status information
    """
    return {
        "env_setup_complete": _ENV_SETUP_COMPLETE,
        "discovery_triggered": _DISCOVERY_TRIGGERED,
        "discovery_enabled": os.getenv("CHUK_LLM_DISCOVERY_ENABLED", "false"),
        "ollama_discovery": os.getenv("CHUK_LLM_OLLAMA_DISCOVERY", "false"),
        "auto_discover": os.getenv("CHUK_LLM_AUTO_DISCOVER", "false"),
        "tool_compatibility": os.getenv("CHUK_LLM_OPENAI_TOOL_COMPATIBILITY", "false"),
        "universal_tools": os.getenv("CHUK_LLM_UNIVERSAL_TOOLS", "false"),
    }


def force_discovery_refresh() -> int:
    """
    Force a fresh discovery (useful for debugging).

    Returns:
        Number of new functions discovered
    """
    global _DISCOVERY_TRIGGERED
    _DISCOVERY_TRIGGERED = False

    # Set force refresh environment variable
    os.environ["CHUK_LLM_DISCOVERY_FORCE_REFRESH"] = "true"

    # Trigger discovery again
    return trigger_discovery_after_setup()
