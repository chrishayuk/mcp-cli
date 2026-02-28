# mcp_cli/dashboard/bridge.py
"""Dashboard bridge — the integration layer between mcp-cli's chat engine
and the browser dashboard.

The bridge is stored on ``ChatContext.dashboard_bridge``. All hooks are no-ops
when the bridge is ``None`` (i.e., ``--dashboard`` was not set), so there is
zero performance impact on normal usage.

Hook call sites:
  tool_processor._on_tool_result()    → bridge.on_tool_result()
  conversation.process_user_input()   → bridge.on_agent_state(), bridge.on_message()
  streaming_handler._process_chunk()  → bridge.on_token()          (Phase 4)
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from mcp_cli.dashboard.server import DashboardServer

logger = logging.getLogger(__name__)

_PROTOCOL = "mcp-dashboard"
_VERSION = 1


def _envelope(msg_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": _PROTOCOL,
        "version": _VERSION,
        "type": msg_type,
        "payload": payload,
    }


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class DashboardBridge:
    """Routes chat-engine events to connected browser dashboard clients."""

    def __init__(self, server: DashboardServer) -> None:
        self.server = server
        self._turn_number: int = 0
        # Set this to inject user messages from the browser back into the chat engine
        self._user_message_callback: Callable[[str], None] | None = None
        # Queue for injecting browser messages into the chat loop
        self._input_queue: asyncio.Queue[str] | None = None
        # Callback to execute a tool requested by the browser (REQUEST_TOOL)
        self._tool_call_callback: (
            Callable[[str, dict[str, Any]], Awaitable[Any]] | None
        ) = None
        # View registry discovered from _meta.ui fields in tool results
        self._view_registry: list[dict[str, Any]] = []
        self._seen_view_ids: set[str] = set()
        # Wire server callbacks
        server.on_browser_message = self._on_browser_message
        server.on_client_connected = self._on_client_connected

    # ------------------------------------------------------------------ #
    #  Outbound hooks (chat engine → browser)                            #
    # ------------------------------------------------------------------ #

    async def on_tool_result(
        self,
        tool_name: str,
        server_name: str,
        result: Any,
        success: bool,
        error: str | None = None,
        duration_ms: int | None = None,
        meta_ui: Any = None,
        call_id: str | None = None,
    ) -> None:
        """Called after every tool execution completes."""
        payload: dict[str, Any] = {
            "tool_name": tool_name,
            "server_name": server_name,
            "agent_id": "default",
            "call_id": call_id or "",
            "timestamp": _now(),
            "duration_ms": duration_ms,
            "result": self._serialise(result),
            "error": error,
            "success": success,
        }
        if meta_ui is not None:
            payload["meta_ui"] = self._serialise(meta_ui)
            # Discover new views declared in _meta.ui before broadcasting
            if isinstance(meta_ui, dict) and meta_ui.get("view"):
                await self._discover_view(meta_ui, server_name)
        await self.server.broadcast(_envelope("TOOL_RESULT", payload))

    async def on_agent_state(
        self,
        status: Literal["thinking", "tool_calling", "idle"],
        current_tool: str | None = None,
        turn_number: int | None = None,
        tokens_used: int = 0,
    ) -> None:
        """Called when the agent's status changes."""
        if turn_number is not None:
            self._turn_number = turn_number
        payload: dict[str, Any] = {
            "agent_id": "default",
            "status": status,
            "current_tool": current_tool,
            "turn_number": self._turn_number,
            "tokens_used": tokens_used,
            "budget_remaining": None,
        }
        await self.server.broadcast(_envelope("AGENT_STATE", payload))

    async def on_message(
        self,
        role: Literal["user", "assistant", "tool"],
        content: str,
        streaming: bool = False,
        reasoning: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Called when a complete conversation message is emitted."""
        payload: dict[str, Any] = {
            "role": role,
            "content": content,
            "streaming": streaming,
            "tool_calls": tool_calls,
            "reasoning": reasoning,
        }
        await self.server.broadcast(_envelope("CONVERSATION_MESSAGE", payload))

    async def on_token(self, token: str, done: bool = False) -> None:
        """Called for each streamed LLM token (high-volume — only used by agent terminal)."""
        payload: dict[str, Any] = {"token": token, "done": done}
        await self.server.broadcast(_envelope("CONVERSATION_TOKEN", payload))

    async def on_view_registry_update(self, views: list[dict[str, Any]]) -> None:
        """Called when the set of available views changes (server connect/disconnect)."""
        await self.server.broadcast({"type": "VIEW_REGISTRY", "views": views})

    async def _discover_view(self, meta_ui: dict[str, Any], server_name: str) -> None:
        """Register a new view from a _meta.ui block and broadcast VIEW_REGISTRY."""
        view_id: str = meta_ui["view"]
        if view_id in self._seen_view_ids:
            return
        self._seen_view_ids.add(view_id)
        entry: dict[str, Any] = {
            "id": view_id,
            "name": meta_ui.get("name") or view_id.replace(":", " ").title(),
            "icon": meta_ui.get("icon") or "◻",
            "source": server_name,
            "type": meta_ui.get("type") or "tool",
            "url": meta_ui.get("url") or f"/views/{view_id}.html",
        }
        self._view_registry.append(entry)
        logger.debug(
            "Dashboard: discovered new view %s from server %s", view_id, server_name
        )
        await self.on_view_registry_update(self._view_registry)

    async def _on_client_connected(self, ws: Any) -> None:
        """Send current VIEW_REGISTRY to a newly connected browser client."""
        if not self._view_registry:
            return
        import json as _json

        try:
            await ws.send(
                _json.dumps({"type": "VIEW_REGISTRY", "views": self._view_registry})
            )
        except Exception as exc:
            logger.debug("Error sending VIEW_REGISTRY to new client: %s", exc)

    # ------------------------------------------------------------------ #
    #  Inbound messages (browser → chat engine)                          #
    # ------------------------------------------------------------------ #

    def set_input_queue(self, queue: asyncio.Queue[str]) -> None:
        """Register the asyncio.Queue that the chat loop reads from.

        Browser messages (USER_MESSAGE / USER_COMMAND) are put on this queue
        so the chat loop picks them up alongside terminal input.
        """
        self._input_queue = queue

    def set_tool_call_callback(
        self, fn: Callable[[str, dict[str, Any]], Awaitable[Any]]
    ) -> None:
        """Register a callback to execute a browser-requested tool call.

        The callback receives (tool_name, arguments) and should execute the
        tool.  It is responsible for broadcasting the result back (e.g. by
        calling ``on_tool_result()`` internally) or returning the result.
        """
        self._tool_call_callback = fn

    async def _on_browser_message(self, msg: dict[str, Any]) -> None:
        msg_type = msg.get("type")
        if msg_type in ("USER_MESSAGE", "USER_COMMAND"):
            content = msg.get("content") or msg.get("command", "")
            if content and self._input_queue is not None:
                try:
                    await self._input_queue.put(content)
                except Exception as exc:
                    logger.warning("Error queuing browser message: %s", exc)
            elif content:
                logger.debug(
                    "Dashboard received %s but no input queue registered", msg_type
                )
        elif msg_type == "REQUEST_TOOL":
            tool_name = msg.get("tool_name") or msg.get("tool") or ""
            arguments = msg.get("arguments") or msg.get("args") or {}
            call_id = msg.get("call_id") or ""
            if tool_name and self._tool_call_callback is not None:
                try:
                    await self._tool_call_callback(tool_name, arguments)
                except Exception as exc:
                    logger.warning(
                        "Error executing REQUEST_TOOL %s: %s", tool_name, exc
                    )
                    await self.on_tool_result(
                        tool_name=tool_name,
                        server_name="",
                        result=None,
                        success=False,
                        error=str(exc),
                        call_id=call_id,
                    )
            else:
                logger.debug(
                    "Dashboard REQUEST_TOOL: %s (no callback registered)",
                    tool_name or "(missing tool_name)",
                )
        elif msg_type == "USER_ACTION":
            action = msg.get("action") or ""
            content = msg.get("content") or ""
            # Prefer explicit content; fall back to slash-command form of action name
            text = content or (f"/{action}" if action else "")
            if text and self._input_queue is not None:
                try:
                    await self._input_queue.put(text)
                except Exception as exc:
                    logger.warning("Error queuing USER_ACTION: %s", exc)
            else:
                logger.debug("Dashboard USER_ACTION not routed: %s", msg)
        else:
            logger.debug("Dashboard received unknown message type: %s", msg_type)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _serialise(value: Any) -> Any:
        """Convert a value to a JSON-safe representation."""
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [DashboardBridge._serialise(v) for v in value]
        if isinstance(value, dict):
            return {k: DashboardBridge._serialise(v) for k, v in value.items()}
        # Objects with a to_dict / model_dump / __dict__
        if hasattr(value, "to_dict"):
            try:
                return DashboardBridge._serialise(value.to_dict())
            except Exception:
                pass
        if hasattr(value, "model_dump"):
            try:
                return DashboardBridge._serialise(value.model_dump())
            except Exception:
                pass
        return str(value)
