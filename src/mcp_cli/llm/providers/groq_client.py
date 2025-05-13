# mcp_cli/llm/providers/groq_client.py
"""
Groq chat-completion adapter for MCP-CLI.

Features
--------
* Shares sanitising / normalising helpers via OpenAIStyleMixin.
* `create_completion(..., stream=False)`  → same dict as before.
* `create_completion(..., stream=True)`   → **async iterator** yielding
  incremental deltas:

      async for chunk in llm.create_completion(msgs, tools, stream=True):
          # chunk = {"response": "...", "tool_calls":[...]}
          ...

  Works in chat UIs for live-token updates.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from groq import Groq

from mcp_cli.llm.openai_style_mixin import OpenAIStyleMixin
from mcp_cli.llm.providers.base import BaseLLMClient

log = logging.getLogger(__name__)


class GroqAILLMClient(OpenAIStyleMixin, BaseLLMClient):
    """
    Adapter around `groq` SDK compatible with MCP-CLI’s BaseLLMClient.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self.model = model
        self.client = (
            Groq(api_key=api_key, base_url=api_base)
            if api_base else
            Groq(api_key=api_key)
        )

    # ──────────────────────────────────────────────────────────────────
    # public API
    # ──────────────────────────────────────────────────────────────────
    async def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
    ) -> Dict[str, Any] | AsyncIterator[Dict[str, Any]]:
        """
        • stream=False → return single normalised dict
        • stream=True  → return async-iterator of delta dicts
        """
        tools = self._sanitize_tool_names(tools)

        if stream:
            return self._stream(messages, tools or [])

        # non-streaming path (unchanged)
        resp = await self._call_blocking(
            self.client.chat.completions.create,
            model=self.model,
            messages=messages,
            tools=tools or [],
        )
        return self._normalise_message(resp.choices[0].message)

    # ──────────────────────────────────────────────────────────────────
    # internal: streaming wrapper
    # ──────────────────────────────────────────────────────────────────
    async def _stream(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Wrap Groq’s blocking streaming generator in an async iterator and
        emit MCP-style deltas.
        """
        queue: asyncio.Queue = asyncio.Queue()

        def _producer() -> None:
            try:
                for chunk in self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    stream=True,
                ):
                    queue.put_nowait(chunk)
            finally:
                queue.put_nowait(None)  # sentinel

        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _producer)

        while True:
            chunk = await queue.get()
            if chunk is None:  # sentinel received
                break

            delta = chunk.choices[0].delta
            yield {
                "response": delta.content or "",
                # Groq includes tool_calls only in the *final* chunk
                "tool_calls": getattr(delta, "tool_calls", []),
            }
