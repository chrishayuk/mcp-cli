# mcp_cli/llm/providers/gemini_client.py
"""
Google Gemini chat-completion adapter for MCP-CLI **(debug-enhanced)**.

Key features
------------
* Fully asynchronous `create_completion` matching `BaseLLMClient`.
* Optional **streaming** via `stream=True`.
* ChatML → Gemini `types.Content` conversion (multimodal-aware).
* Accepts **both** OpenAI-style and bare function declarations.
* ⚡ **Extensive DEBUG logging** – enable with e.g. `export LOGLEVEL=DEBUG`.

Implementation details
---------------------
* Blocking SDK calls are wrapped in `asyncio.to_thread`.
* All crucial transformation steps now emit `log.debug(...)` lines so you can
  trace exactly what goes to / comes from the Gemini backend.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types as gtypes

from mcp_cli.llm.providers.base import BaseLLMClient

log = logging.getLogger(__name__)

# Honour LOGLEVEL env-var for quick local tweaks
if "LOGLEVEL" in os.environ:
    log.setLevel(os.environ["LOGLEVEL"].upper())

# ───────────────────────────────────────────────────────── helpers ──────────

def _convert_messages(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[gtypes.Content]]:
    """Convert ChatML list → Gemini contents list.

    Returns
    -------
    system_instruction : Optional[str]
    contents           : list[google.genai.types.Content]
    """
    system_txt: Optional[str] = None
    gem_contents: List[gtypes.Content] = []

    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        log.debug("↻ msg[%d] role=%s keys=%s", i, role, list(msg.keys()))

        # ---------------- system -----------------------------------------
        if role == "system":
            if system_txt is None:  # Gemini supports only one system prompt
                system_txt = content if isinstance(content, str) else str(content)
            continue

        # ---------------- tool response ----------------------------------
        if role == "tool":
            fn_name = msg.get("name") or "tool"
            try:
                payload = json.loads(content) if isinstance(content, str) else content
            except Exception:
                payload = {"response": content}
            part = gtypes.Part.from_function_response(name=fn_name, response=payload)
            gem_contents.append(gtypes.Content(role="tool", parts=[part]))
            continue

        # ---------------- assistant function-calls ------------------------
        if role == "assistant" and msg.get("tool_calls"):
            parts: List[gtypes.Part] = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name")
                args_raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {}
                if name:
                    parts.append(gtypes.Part.from_function_call(name=name, args=args))
            if parts:
                gem_contents.append(gtypes.Content(role="model", parts=parts))
            continue

        # ---------------- normal / multimodal ----------------------------
        if role in {"user", "assistant"}:
            g_role = "user" if role == "user" else "model"
            if isinstance(content, str):
                gem_contents.append(gtypes.Content(role=g_role, parts=[gtypes.Part.from_text(text=content)]))
            elif isinstance(content, list):
                gem_contents.append(gtypes.Content(role=g_role, parts=content))
            elif content is not None:
                gem_contents.append(gtypes.Content(role=g_role, parts=[content]))
        else:
            log.debug("Skipping unsupported message: %s", msg)

    log.debug("System-instruction: %s", system_txt)
    log.debug("Gemini contents prepared: %s", gem_contents)
    return system_txt, gem_contents


def _convert_tools(tools: Optional[List[Dict[str, Any]]]) -> Tuple[List[gtypes.Tool], Optional[gtypes.ToolConfig]]:
    """Translate mixed tool formats → Gemini Tool objects."""
    if not tools:
        log.debug("No tools supplied")
        return [], None

    fn_decls: List[gtypes.FunctionDeclaration] = []

    for entry in tools:
        if entry.get("type") == "function":
            fn = entry.get("function", {})
        else:
            fn = entry

        name = fn.get("name")
        description = fn.get("description", "")
        params = fn.get("parameters", {})
        if not name:
            log.warning("Skipping tool without name: %s", entry)
            continue

        try:
            schema = params if isinstance(params, gtypes.Schema) else gtypes.Schema(**params)
        except Exception as exc:
            log.error("Invalid schema for tool '%s' (%s) → using permissive schema", name, exc)
            schema = gtypes.Schema(type="object", additionalProperties=True)

        decl = gtypes.FunctionDeclaration(name=name, description=description, parameters=schema)
        fn_decls.append(decl)
        log.debug("Tool registered: %s", decl)

    if not fn_decls:
        return [], None

    tools_list = [gtypes.Tool(function_declarations=fn_decls)]
    tool_cfg = gtypes.ToolConfig(
        function_calling_config=gtypes.FunctionCallingConfig(mode=gtypes.FunctionCallingConfigMode.AUTO)
    )
    return tools_list, tool_cfg


# ─────────────────────────────────────────────────── main adapter ───────────

class GeminiLLMClient(BaseLLMClient):
    """`google-genai` wrapper with MCP-CLI interface."""

    def __init__(self, model: str = "gemini-2.0-flash", *, api_key: Optional[str] = None) -> None:
        load_dotenv()
        api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY / GEMINI_API_KEY env var not set")

        self.model = model
        self.client = genai.Client(api_key=api_key)
        log.info("GeminiLLMClient initialised with model '%s'", model)

    # ---------------------------------------------------------------- sync (non-streaming)
    def _create_sync(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        system_txt, contents = _convert_messages(messages)
        gem_tools, tool_cfg = _convert_tools(tools)

        cfg = gtypes.GenerateContentConfig(system_instruction=system_txt or None, tools=gem_tools or None, tool_config=tool_cfg)
        log.debug("GenerateContentConfig: %s", cfg)
        log.debug("Sending request to Gemini…")

        resp = self.client.models.generate_content(model=self.model, contents=contents, config=cfg)
        log.debug("Raw Gemini response: %s", resp)
        return _parse_final_response(resp)

    # ---------------------------------------------------------------- streaming helpers
    def _stream(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]) -> AsyncIterator[Dict[str, Any]]:
        system_txt, contents = _convert_messages(messages)
        gem_tools, tool_cfg = _convert_tools(tools)
        cfg = gtypes.GenerateContentConfig(system_instruction=system_txt or None, tools=gem_tools or None, tool_config=tool_cfg)

        queue: asyncio.Queue = asyncio.Queue()

        def _producer():
            try:
                for chunk in self.client.models.generate_content_stream(model=self.model, contents=contents, config=cfg):
                    queue.put_nowait(chunk)
            finally:
                queue.put_nowait(None)

        asyncio.get_running_loop().run_in_executor(None, _producer)

        async def _aiter():
            while True:
                chunk = await queue.get()
                if chunk is None:
                    break
                text_piece, t_calls = _parse_stream_chunk(chunk)
                yield {"response": text_piece, "tool_calls": t_calls}

        return _aiter()

    # ---------------------------------------------------------------- async facade
    async def create_completion(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, *, stream: bool = False) -> Dict[str, Any] | AsyncIterator[Dict[str, Any]]:
        log.debug("create_completion called – stream=%s", stream)
        if stream:
            return self._stream(messages, tools)
        return await asyncio.to_thread(self._create_sync, messages, tools)


# ───────────────────────────────────────── parse helpers ──────────

def _parse_stream_chunk(chunk) -> Tuple[str, List[Dict[str, Any]]]:
    delta_text: str = getattr(chunk, "text", "") or ""
    calls: List[Dict[str, Any]] = []

    if not delta_text and getattr(chunk, "candidates", None):
        cand = chunk.candidates[0]
        for part in cand.content.parts:
            if hasattr(part, "text") and part.text:
                delta_text += part.text
            elif hasattr(part, "function_call"):
                fc = part.function_call
                calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function", "function": {"name": fc.name, "arguments": json.dumps(dict(fc.args))}})

    log.debug("Stream chunk parsed – text='%s…', tool_calls=%d", delta_text[:40], len(calls))
    return delta_text, calls


def _parse_final_response(resp) -> Dict[str, Any]:
    main_text: str = ""
    tool_calls: List[Dict[str, Any]] = []

    cand = resp.candidates[0]
    for part in cand.content.parts:
        if hasattr(part, "text") and part.text:
            main_text += part.text
        elif hasattr(part, "function_call"):
            fc = part.function_call
            tool_calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function", "function": {"name": fc.name, "arguments": json.dumps(dict(fc.args))}})

    if not tool_calls and getattr(resp, "functionCalls", None):  # type: ignore[attr-defined]
        for fc in resp.functionCalls:  # type: ignore[attr-defined]
            tool_calls.append({"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function", "function": {"name": fc.name, "arguments": json.dumps(dict(fc.args))}})

    log.debug("Parsed text='%s…', tool_calls=%d", main_text[:60], len(tool_calls))

    if tool_calls and not main_text.strip():
        return {"response": None, "tool_calls": tool_calls}
    return {"response": main_text.strip(), "tool_calls": tool_calls}
