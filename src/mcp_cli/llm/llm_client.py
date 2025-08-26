# mcp_cli/llm/llm_client.py
"""
LLM client interface - compatibility layer and stub for tests
"""

from typing import Any, Optional

try:
    # Try to import from the real chuk-llm if available
    from chuk_llm.llm.llm_client import get_llm_client as _real_get_llm_client

    _HAS_CHUK_LLM = True
except ImportError:
    _HAS_CHUK_LLM = False


class LLMClient:
    """Base LLM client interface."""

    async def create_completion(self, *args, **kwargs) -> str:
        """Create a completion using the LLM."""
        raise NotImplementedError("Subclasses must implement create_completion")


class StubLLMClient(LLMClient):
    """Stub LLM client for testing."""

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.provider = provider
        self.model = model

    async def create_completion(self, *args, **kwargs) -> str:
        """Return a test response."""
        return f"Test response from {self.provider} {self.model}"


def get_llm_client(
    provider: str = "openai", model: Optional[str] = None, **kwargs
) -> Any:
    """
    Get an LLM client instance.

    This function provides compatibility for tests while allowing real usage
    when chuk-llm is available.
    """
    if _HAS_CHUK_LLM:
        # Use real implementation if available
        return _real_get_llm_client(provider=provider, model=model, **kwargs)
    else:
        # Use stub for tests
        return StubLLMClient(provider=provider, model=model or "gpt-4o-mini")
