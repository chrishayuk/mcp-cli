"""Protocol definitions for type safety."""

from mcp_cli.protocols.llm_client import LLMClient, StreamingLLMClient

__all__ = [
    "LLMClient",
    "StreamingLLMClient",
]
