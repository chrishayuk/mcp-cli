"""
Anthropic chat-completion adapter for MCP-CLI (Claude 3 family).

* Accepts OpenAI-style *or* bare function schemas.
* Converts tool specs to Claude’s `input_schema`.
* Lifts `system` messages to the top-level arg (Claude 3 requirement).
* Converts:
    • assistant tool-calls  → MCP tool_calls
    • tool results          → Claude `tool_result` blocks
  so loops are avoided.
* Streams via the shared OpenAIStyleMixin helper.
* Adds verbose DEBUG logging (honours LOGLEVEL env-var).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from anthropic import Anthropic
from anthropic.types import ToolUseBlock  # SDK 0.25+

from mcp_cli.llm.openai_style_mixin import OpenAIStyleMixin
from mcp_cli.llm.providers.base import BaseLLMClient

log = logging.getLogger(__name__)
if os.getenv("LOGLEVEL"):
    logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO").upper())


# ───────────────────────────────────────────────────────────────────────── helpers
def _get(obj: Any, key: str, default: Any = None) -> Any:
    """
    Safe accessor that works for both dicts and pydantic objects.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_claude_response(resp) -> Dict[str, Any]:
    """
    Convert Claude Message → MCP dict (response/tool_calls).
    """
    tool_calls: List[Dict[str, Any]] = []

    for blk in getattr(resp, "content", []):
        # Claude 3 SDK returns ToolUseBlock objects
        blk_type = _get(blk, "type")

        if blk_type != "tool_use":
            continue

        blk_id = _get(blk, "id") or f"call_{uuid.uuid4().hex[:8]}"
        blk_name = _get(blk, "name")
        blk_input = _get(blk, "input", {})

        tool_calls.append(
            {
                "id": blk_id,
                "type": "function",
                "function": {
                    "name": blk_name,
                    "arguments": json.dumps(blk_input),
                },
            }
        )

    if tool_calls:
        return {"response": None, "tool_calls": tool_calls}

    # otherwise plain assistant text
    assistant_text = (
        resp.content[0].text
        if getattr(resp, "content", None)
        else ""
    )
    return {"response": assistant_text, "tool_calls": []}


# ───────────────────────────────────────────────────────────────────────── client
class AnthropicLLMClient(OpenAIStyleMixin, BaseLLMClient):
    """Thin wrapper around the official `anthropic` SDK."""

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
            kwargs["api_key"] = api_key
        self.client = Anthropic(**kwargs)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _convert_tools(
        tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Translate OpenAI-style function schemas → Claude `input_schema`.
        Accepts either wrapped {"type":"function", "function":{...}}
        or bare declarations.
        """
        if not tools:
            return None

        converted: List[Dict[str, Any]] = []
        for t in tools:
            fn = t.get("function", t)
            try:
                converted.append(
                    {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "input_schema": (
                            fn.get("parameters")
                            or fn.get("input_schema")
                            or {}
                        ),
                    }
                )
            except Exception as err:  # schema malformed?
                log.debug("Tool schema error (%s) – using open schema", err)
                converted.append(
                    {
                        "name": fn.get("name", f"tool_{uuid.uuid4().hex[:6]}"),
                        "description": fn.get("description", ""),
                        "input_schema": {
                            "type": "object",
                            "additionalProperties": True,
                        },
                    }
                )
        return converted

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _split_for_anthropic(
        messages: List[Dict[str, Any]]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Convert ChatML → Claude-3 Messages format.

        • returns (system_text, msg_list_without_system)
        • keeps only roles Claude accepts ("user", "assistant")
        • converts assistant tool-calls  → tool_use blocks
        • converts role="tool" results  → tool_result blocks
        *and* preserves the id pairing so loops disappear.
        """
        sys_parts: List[str] = []
        filtered:  List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")

            # ─────────────────────────── system ───────────────────────────
            if role == "system":
                sys_parts.append(msg.get("content", ""))
                continue

            # ────────────────── assistant tool_call → tool_use ────────────
            if role == "assistant" and msg.get("tool_calls"):
                blocks = []
                for tc in msg["tool_calls"]:
                    # tc["id"] is guaranteed by OpenAIStyleMixin
                    tc_id   = tc["id"]
                    fn_name = tc["function"]["name"]
                    args    = json.loads(tc["function"]["arguments"] or "{}")
                    blocks.append(
                        {
                            "type":  "tool_use",
                            "id":    tc_id,
                            "name":  fn_name,
                            "input": args,
                        }
                    )
                filtered.append({"role": "assistant", "content": blocks})
                continue  # do *not* fall through to normal assistant handling

            # ──────────────── role="tool" → Claude tool_result ────────────
            if role == "tool":
                tc_id = (
                    msg.get("tool_call_id")
                    or msg.get("id")                       # edge-case
                    or f"toolu_{uuid.uuid4().hex[:8]}"
                )
                filtered.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tc_id,
                                "content": msg.get("content") or "",
                            }
                        ],
                    }
                )
                continue

            # ─────────────── normal user / assistant text ────────────────
            if role in {"user", "assistant"}:
                cont = msg.get("content")
                if cont is None:
                    # skip placeholder rows (e.g. assistant stub before tool_call)
                    continue
                if isinstance(cont, str):                    # wrap plain text
                    msg = dict(msg)
                    msg["content"] = [{"type": "text", "text": cont}]
                filtered.append(msg)

            # ignore any other role (e.g. "function")

        return "\n".join(sys_parts).strip(), filtered


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

        # 1. sanitise + convert tool schemas
        tools = self._sanitize_tool_names(tools)
        anth_tools = self._convert_tools(tools)

        # 2. split messages for Claude
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

        log.debug("Claude payload: %s", payload)

        # ───────────── streaming ─────────────
        if stream:
            # _stream_from_blocking handles chunk→delta conversion
            return self._stream_from_blocking(
                self.client.messages.create, **payload
            )

        # ───────────── one-shot ─────────────
        resp = await self._call_blocking(
            self.client.messages.create, **payload
        )
        log.debug("Claude raw response: %s", resp)

        return _parse_claude_response(resp)
