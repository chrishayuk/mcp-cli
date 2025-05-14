# mcp_cli/llm/providers/anthropic_client.py
"""
Anthropic chat-completion adapter for MCP-CLI (Claude 3 family).

* Accepts OpenAI-style **or** bare function schemas.
* Converts those schemas to Claude’s `input_schema`.
* Lifts `system` messages to the top-level arg (Claude requirement).
* Converts:
    • assistant tool-calls  → `tool_use` blocks
    • tool results          → `tool_result` blocks
  so loops are avoided.
* Streams via the shared OpenAIStyleMixin helper.
* DEBUG logging honours the LOGLEVEL env-var.

Fixes
-----
* `tools` is always a list (never `null`).
* **NEW** 2025-05-14 – omit the `"system"` key entirely when there is no system
  prompt, preventing `system: Input should be a valid list` 400 errors.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from anthropic import Anthropic

from mcp_cli.llm.openai_style_mixin import OpenAIStyleMixin
from mcp_cli.llm.providers.base import BaseLLMClient

log = logging.getLogger(__name__)
if os.getenv("LOGLEVEL"):
    logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO").upper())


# ─────────────────────────── helpers
def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _parse_claude_response(resp) -> Dict[str, Any]:
    tool_calls: List[Dict[str, Any]] = []
    for blk in getattr(resp, "content", []):
        if _safe_get(blk, "type") != "tool_use":
            continue
        tool_calls.append(
            {
                "id": _safe_get(blk, "id") or f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": _safe_get(blk, "name"),
                    "arguments": json.dumps(_safe_get(blk, "input", {})),
                },
            }
        )

    if tool_calls:
        return {"response": None, "tool_calls": tool_calls}

    text = resp.content[0].text if getattr(resp, "content", None) else ""
    return {"response": text, "tool_calls": []}


# ─────────────────────────── client
class AnthropicLLMClient(OpenAIStyleMixin, BaseLLMClient):
    def __init__(
        self,
        model: str = "claude-3-sonnet-20250219",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        self.model = model
        kwargs: Dict[str, Any] = {"base_url": api_base} if api_base else {}
        if api_key:
            kwargs["api_key"] = api_key
        self.client = Anthropic(**kwargs)

    # ------------- helpers
    @staticmethod
    def _convert_tools(tools: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not tools:
            return []

        converted: List[Dict[str, Any]] = []
        for entry in tools:
            fn = entry.get("function", entry)
            try:
                converted.append(
                    {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters") or fn.get("input_schema") or {},
                    }
                )
            except Exception as exc:
                log.debug("Tool schema error (%s) – using open schema", exc)
                converted.append(
                    {
                        "name": fn.get("name", f"tool_{uuid.uuid4().hex[:6]}"),
                        "description": fn.get("description", ""),
                        "input_schema": {"type": "object", "additionalProperties": True},
                    }
                )
        return converted

    @staticmethod
    def _split_for_anthropic(messages: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
        sys_txt: List[str] = []
        out: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")

            if role == "system":
                sys_txt.append(msg.get("content", ""))
                continue

            if role == "assistant" and msg.get("tool_calls"):
                blocks = [
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"].get("arguments", "{}")),
                    }
                    for tc in msg["tool_calls"]
                ]
                out.append({"role": "assistant", "content": blocks})
                continue

            if role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id")
                                or msg.get("id", f"tr_{uuid.uuid4().hex[:8]}"),
                                "content": msg.get("content") or "",
                            }
                        ],
                    }
                )
                continue

            if role in {"user", "assistant"}:
                cont = msg.get("content")
                if cont is None:
                    continue
                if isinstance(cont, str):
                    msg = dict(msg)
                    msg["content"] = [{"type": "text", "text": cont}]
                out.append(msg)

        return "\n".join(sys_txt).strip(), out

    # ------------- public
    async def create_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        **extra,
    ) -> Dict[str, Any] | AsyncIterator[Dict[str, Any]]:

        tools = self._sanitize_tool_names(tools)
        anth_tools = self._convert_tools(tools)

        system_text, msg_no_system = self._split_for_anthropic(messages)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": msg_no_system,
            "stream": stream,
            "tools": anth_tools,
            "max_tokens": max_tokens or 1024,
            **extra,
        }
        if system_text:            # ← only send if non-empty
            payload["system"] = system_text
        if anth_tools:
            payload["tool_choice"] = {"type": "auto"}

        log.debug("Claude payload: %s", payload)

        if stream:
            return self._stream_from_blocking(self.client.messages.create, **payload)

        resp = await self._call_blocking(self.client.messages.create, **payload)
        log.debug("Claude raw response: %s", resp)
        return _parse_claude_response(resp)
