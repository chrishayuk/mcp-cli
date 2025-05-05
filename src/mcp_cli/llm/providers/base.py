# mcp_cli/llm/providers/base.py
"""
Common abstract interface for every LLM adapter used by MCP-CLI.

* All implementations **must be asynchronous** – i.e. `await
  client.create_completion(...)`.
* The returned value is a dict with at least

      {
          "response": str | None,      # assistant message (None if only tool-calls)
          "tool_calls": list[dict],    # OpenAI-style tool-call objects
      }
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional


class BaseLLMClient(abc.ABC):
    """Abstract base class for LLM chat clients."""

    @abc.abstractmethod
    async def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generate (or continue) a chat conversation.

        Parameters
        ----------
        messages
            List of ChatML-style message dicts.
        tools
            Optional list of OpenAI-function-tool schemas.

        Returns
        -------
        Standardised payload with keys ``response`` and ``tool_calls``.
        """
        ...
