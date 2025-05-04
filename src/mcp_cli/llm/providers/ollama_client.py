# mcp_cli/llm/providers/ollama_client.py
"""
Async Ollama LLM adapter for MCP-CLI.

* `ollama.chat()` is still a synchronous helper in the upstream
  library, so we run it in a background thread via
  ``asyncio.to_thread`` to avoid blocking the event-loop.
* Public contract matches the OpenAI adapter:
    await client.create_completion(messages, tools)
      →  {"response": str | None, "tool_calls": List[dict]}
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import ollama  # pip install ollama-python

from mcp_cli.llm.providers.base import BaseLLMClient

log = logging.getLogger(__name__)


class OllamaLLMClient(BaseLLMClient):
    """Non-blocking wrapper around ``ollama.chat``."""

    def __init__(self, model: str = "qwen2.5-coder") -> None:
        self.model = model

        # Fail fast if the runtime lacks the required function
        if not hasattr(ollama, "chat"):
            raise ValueError(
                "The installed ollama package does not expose 'chat'; "
                "check your ollama-python version."
            )

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    async def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Fire a chat request at Ollama **without** blocking the event-loop.

        Parameters
        ----------
        messages
            Standard ChatML message list.
        tools
            Optional OpenAI-style tool schema list.

        Returns
        -------
        Same dict structure as the OpenAI adapter.
        """
        # Convert to Ollama’s simple schema
        ollama_messages = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]

        try:
            # ------------------------------------------------------------------
            # Run the *synchronous* API in a background thread
            # ------------------------------------------------------------------
            response = await asyncio.to_thread(
                functools.partial(
                    ollama.chat,
                    model=self.model,
                    messages=ollama_messages,
                    stream=False,
                    tools=tools or [],
                )
            )
            log.debug("Ollama raw response: %s", response)

            # ------------------------------------------------------------------
            # Normalise the payload
            # ------------------------------------------------------------------
            main_msg = response.message
            main_content: Optional[str] = (
                main_msg.content if main_msg is not None else None
            )

            tool_calls: List[Dict[str, Any]] = []
            if hasattr(main_msg, "tool_calls") and main_msg.tool_calls:
                for tc in main_msg.tool_calls:
                    args_raw = tc.function.arguments

                    if isinstance(args_raw, dict):
                        args_str = json.dumps(args_raw)
                    elif isinstance(args_raw, str):
                        args_str = args_raw
                    else:
                        args_str = str(args_raw)

                    tc_id = getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:8]}"

                    tool_calls.append(
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": args_str,
                            },
                        }
                    )

            return {
                "response": main_content or "No response",
                "tool_calls": tool_calls,
            }

        except Exception as exc:  # noqa: BLE001
            log.error("Ollama API Error: %s", exc, exc_info=True)
            raise ValueError(f"Ollama API Error: {exc}") from exc
