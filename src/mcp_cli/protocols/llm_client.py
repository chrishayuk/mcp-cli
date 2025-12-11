"""Protocol definitions for LLM clients - no more Any types!

Defines the interface that all LLM clients must implement.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from mcp_cli.chat.response_models import CompletionResponse


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM client interface.

    All LLM clients (OpenAI, Anthropic, Ollama, etc.) must implement this interface.
    """

    async def create_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Create a completion from messages.

        Args:
            messages: Conversation history in OpenAI format
            tools: Available tools for function calling
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters

        Returns:
            CompletionResponse with text and optional tool calls
        """
        ...

    def get_model_name(self) -> str:
        """Get the model name being used.

        Returns:
            Model identifier string
        """
        ...

    def get_provider_name(self) -> str:
        """Get the provider name.

        Returns:
            Provider identifier (e.g., 'openai', 'anthropic', 'ollama')
        """
        ...


@runtime_checkable
class StreamingLLMClient(LLMClient, Protocol):
    """Protocol for LLM clients that support streaming.

    Extends base LLMClient with streaming-specific methods.
    """

    async def create_streaming_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        """Create a streaming completion.

        Args:
            messages: Conversation history
            tools: Available tools
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            CompletionResponse with accumulated stream data
        """
        ...


__all__ = [
    "LLMClient",
    "StreamingLLMClient",
]
