"""
Anthropic chat‑completion adapter for MCP‑CLI (Claude 3 family).

* Converts OpenAI‑style tool specs to Anthropic format (`input_schema`).
* Moves `system` role messages to the top‑level `system` arg (Claude 3 API requirement).
* Streams via the shared OpenAIStyleMixin helper.
* Provides default `max_tokens = 1024`.
* Detects `tool_use` blocks (from the SDK or raw dict) and surfaces them as
  MCP‑style `tool_calls` so the upstream handler can execute them.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from anthropic import Anthropic

from mcp_cli.llm.openai_style_mixin import OpenAIStyleMixin
from mcp_cli.llm.providers.base import BaseLLMClient


class AnthropicLLMClient(OpenAIStyleMixin, BaseLLMClient):
    """A very thin wrapper around the official *anthropic* SDK."""

    # --------------------------------------------------------------------- init
    def __init__(
        self,
        model: str = "claude-3-sonnet-20250219",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self.model = model
        kwargs: Dict[str, Any] = {"base_url": api_base} if api_base else {}
        if api_key:
            kwargs["api_key"] = api_key  # otherwise SDK falls back to env‑var
        self.client = Anthropic(**kwargs)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _convert_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Translate OpenAI function‑call schema → Anthropic `input_schema`."""
        if not tools:
            return None
        converted: List[Dict[str, Any]] = []
        for t in tools:
            fn = t.get("function", t)
            converted.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or fn.get("input_schema") or {},
                }
            )
        return converted

    @staticmethod
    def _split_for_anthropic(messages: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
        """Return `(system_text, filtered_messages)` removing unsupported roles."""
        system_parts: List[str] = []
        filtered: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if role == "system":
                system_parts.append(msg.get("content", ""))
            elif role in {"user", "assistant"}:  # Claude accepts only these two
                cont = msg.get("content")
                if cont is None:
                    # skip empty assistant placeholders (e.g. old OpenAI tool_call stubs)
                    continue
                # ensure proper list form
                if isinstance(cont, str):
                    msg = dict(msg)
                    msg["content"] = [{"type": "text", "text": cont}]
                filtered.append(msg)
            # ignore any other role (e.g. "tool")
        return "\n".join(system_parts).strip(), filtered

    # ------------------------------------------------------------------ public
    async def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        **extra,
    ) -> Dict[str, Any] | AsyncIterator[Dict[str, Any]]:
        # Sanitize + convert tool schemas
        tools = self._sanitize_tool_names(tools)
        anth_tools = self._convert_tools(tools)

        # Split out system prompt and strip unsupported roles
        system_text, msg_no_system = self._split_for_anthropic(messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "system": system_text or None,
            "messages": msg_no_system,
            "stream": stream,
            "tools": anth_tools,
            "max_tokens": max_tokens or 1024,
            **extra,
        }
        if anth_tools:
            payload["tool_choice"] = {"type": "auto"}

        # ---------------- streaming -----------------
        if stream:
            return self._stream_from_blocking(self.client.messages.create, **payload)

        # ---------------- one‑shot -------------------
        resp = await self._call_blocking(self.client.messages.create, **payload)

        # Detect tool_use blocks → convert to MCP tool_calls
        tool_calls: List[Dict[str, Any]] = []
        for blk in getattr(resp, "content", []):
            blk_type = blk.get("type") if isinstance(blk, dict) else getattr(blk, "type", None)
            if blk_type != "tool_use":
                continue
            blk_id = blk.get("id") if isinstance(blk, dict) else getattr(blk, "id", None)
            blk_name = blk.get("name") if isinstance(blk, dict) else getattr(blk, "name", None)
            blk_input = blk.get("input") if isinstance(blk, dict) else getattr(blk, "input", None)
            tool_calls.append(
                {
                    "id": blk_id or f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": blk_name,
                        "arguments": json.dumps(blk_input or {}),
                    },
                }
            )

        if tool_calls:
            return {"response": None, "tool_calls": tool_calls}

        assistant_text = resp.content[0].text if getattr(resp, "content", None) else ""
        return {"response": assistant_text, "tool_calls": []}
