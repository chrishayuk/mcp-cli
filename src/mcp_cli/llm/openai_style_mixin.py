# mcp_cli/llm/openai_style_mixin.py
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
)

Tool      = Dict[str, Any]
LLMResult = Dict[str, Any]          # {"response": str|None, "tool_calls":[...]}


class OpenAIStyleMixin:
    """
    Helper mix-in for providers that emit OpenAI-style messages
    (OpenAI, Groq, Anthropic, Azure OpenAI, etc.).
    Includes:

      • _sanitize_tool_names
      • _call_blocking          – run blocking SDK in thread
      • _normalise_message      – convert full message → MCP dict
      • _stream_from_blocking   – wrap *stream=True* SDK generators
    """

    # ------------------------------------------------------------------ sanitise
    _NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")

    @classmethod
    def _sanitize_tool_names(cls, tools: Optional[List[Tool]]) -> Optional[List[Tool]]:
        if not tools:
            return tools
        fixed: List[Tool] = []
        for t in tools:
            copy = dict(t)
            fn = copy.get("function", {})
            name = fn.get("name")
            if name and cls._NAME_RE.search(name):
                clean = cls._NAME_RE.sub("_", name)
                logging.debug("Sanitising tool name '%s' → '%s'", name, clean)
                fn["name"] = clean
                copy["function"] = fn
            fixed.append(copy)
        return fixed

    # ------------------------------------------------------------------ blocking
    @staticmethod
    async def _call_blocking(fn: Callable, *args, **kwargs):
        """Run a blocking SDK call in a background thread."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    # ------------------------------------------------------------------ normalise
    @staticmethod
    def _normalise_message(msg) -> LLMResult:
        """
        Convert `response.choices[0].message` (full) → MCP dict.
        """
        raw = getattr(msg, "tool_calls", None)
        calls: List[Dict[str, Any]] = []

        if raw:
            for c in raw:
                cid = getattr(c, "id", None) or f"call_{uuid.uuid4().hex[:8]}"
                try:
                    args   = c.function.arguments
                    args_j = (
                        json.dumps(json.loads(args))
                        if isinstance(args, str)
                        else json.dumps(args)
                    )
                except (TypeError, json.JSONDecodeError):
                    args_j = "{}"

                calls.append(
                    {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": c.function.name,
                            "arguments": args_j,
                        },
                    }
                )

        return {"response": msg.content if not calls else None, "tool_calls": calls}

    # ------------------------------------------------------------------ streaming
    @classmethod
    def _stream_from_blocking(
        cls,
        sdk_call: Callable[..., Any],
        /,
        **kwargs,
    ) -> AsyncIterator[LLMResult]:
        """
        Wrap a *blocking* SDK streaming generator (``stream=True``) and yield
        MCP-style *delta dictionaries* asynchronously.

        Usage inside provider adapter::

            return self._stream_from_blocking(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                tools=tools or [],
            )
        """
        queue: asyncio.Queue = asyncio.Queue()

        async def _aiter() -> AsyncIterator[LLMResult]:
            while True:
                chunk = await queue.get()
                if chunk is None:               # sentinel from worker
                    break
                delta = chunk.choices[0].delta
                yield {
                    "response": delta.content or "",
                    "tool_calls": getattr(delta, "tool_calls", []),
                }

        # run the blocking generator in a thread
        def _worker():
            try:
                for ch in sdk_call(stream=True, **kwargs):
                    queue.put_nowait(ch)
            finally:
                queue.put_nowait(None)

        asyncio.get_running_loop().run_in_executor(None, _worker)
        return _aiter()
