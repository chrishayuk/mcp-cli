# mcp_cli/apps/bridge.py
"""MCP Apps bridge — handles protocol between browser WebSocket and MCP servers.

This is the Python-side protocol handler.  It receives JSON-RPC messages
from the browser host page via WebSocket and routes them to the
appropriate MCP server via ToolManager.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TYPE_CHECKING

from mcp_cli.apps.models import AppInfo, AppState

if TYPE_CHECKING:
    from mcp_cli.tools.manager import ToolManager

log = logging.getLogger(__name__)


class AppBridge:
    """Bridges WebSocket messages from browser to MCP server tool calls."""

    def __init__(self, app_info: AppInfo, tool_manager: ToolManager) -> None:
        self.app_info = app_info
        self.tool_manager = tool_manager
        self._ws: Any = None
        self._model_context: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    #  WebSocket lifecycle                                                #
    # ------------------------------------------------------------------ #

    def set_ws(self, ws: Any) -> None:
        """Attach the active WebSocket connection."""
        self._ws = ws

    # ------------------------------------------------------------------ #
    #  Inbound: browser -> Python                                        #
    # ------------------------------------------------------------------ #

    async def handle_message(self, raw: str) -> str | None:
        """Handle a JSON-RPC message from the browser.

        Returns a JSON-RPC response string, or *None* for notifications.
        """
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("Invalid JSON from browser: %s", raw[:200])
            return None

        method = msg.get("method")
        msg_id = msg.get("id")
        params = msg.get("params", {})

        if method == "tools/call":
            return await self._handle_tool_call(msg_id, params)

        if method == "resources/read":
            return await self._handle_resource_read(msg_id, params)

        if method == "ui/message":
            return self._handle_ui_message(msg_id, params)

        if method == "ui/update-model-context":
            return self._handle_model_context_update(msg_id, params)

        if method == "ui/notifications/initialized":
            self.app_info.state = AppState.READY
            log.info("App %s initialized", self.app_info.tool_name)
            return None

        # Unknown notification — ignore silently
        if msg_id is None:
            return None

        # Unknown request — return error
        return json.dumps({
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        })

    # ------------------------------------------------------------------ #
    #  Handler: tools/call                                                #
    # ------------------------------------------------------------------ #

    async def _handle_tool_call(
        self, msg_id: Any, params: dict[str, Any]
    ) -> str:
        """Proxy a tool call from the app to the MCP server."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        log.info(
            "App %s calling tool %s with %s",
            self.app_info.tool_name, tool_name, arguments,
        )

        try:
            result = await self.tool_manager.execute_tool(
                tool_name,
                arguments,
                namespace=self.app_info.server_name,
            )

            if result.success:
                # Format result in MCP content structure
                result_content = self._format_tool_result(result.result)
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": result_content,
                })
            else:
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32000,
                        "message": result.error or "Tool execution failed",
                    },
                })

        except Exception as e:
            log.error("Tool call failed: %s", e)
            return json.dumps({
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32000, "message": str(e)},
            })

    # ------------------------------------------------------------------ #
    #  Handler: resources/read                                            #
    # ------------------------------------------------------------------ #

    async def _handle_resource_read(
        self, msg_id: Any, params: dict[str, Any]
    ) -> str:
        """Proxy a resource read from the app to the MCP server."""
        uri = params.get("uri", "")

        try:
            result = await self.tool_manager.read_resource(
                uri, server_name=self.app_info.server_name
            )
            return json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result})
        except Exception as e:
            log.error("Resource read failed: %s", e)
            return json.dumps({
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32000, "message": str(e)},
            })

    # ------------------------------------------------------------------ #
    #  Handler: ui/message                                                #
    # ------------------------------------------------------------------ #

    def _handle_ui_message(
        self, msg_id: Any, params: dict[str, Any]
    ) -> str:
        """Handle a message from the app to be added to conversation."""
        content = params.get("content", {})
        log.info("App %s sent message: %s", self.app_info.tool_name, content)
        return json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": {}})

    # ------------------------------------------------------------------ #
    #  Handler: ui/update-model-context                                   #
    # ------------------------------------------------------------------ #

    def _handle_model_context_update(
        self, msg_id: Any, params: dict[str, Any]
    ) -> str:
        """Store updated model context from the app."""
        self._model_context = params
        log.info("App %s updated model context", self.app_info.tool_name)
        return json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": {}})

    # ------------------------------------------------------------------ #
    #  Outbound: Python -> browser                                        #
    # ------------------------------------------------------------------ #

    async def push_tool_result(self, result: Any) -> None:
        """Push a tool result notification to the app."""
        if not self._ws:
            return

        notification = json.dumps({
            "jsonrpc": "2.0",
            "method": "ui/notifications/tool-result",
            "params": self._format_tool_result(result),
        })

        try:
            await self._ws.send(notification)
        except Exception as e:
            log.warning("Failed to push tool result: %s", e)

    async def push_tool_input(self, arguments: dict[str, Any]) -> None:
        """Push tool input to the app (sent after initialization)."""
        if not self._ws:
            return

        notification = json.dumps({
            "jsonrpc": "2.0",
            "method": "ui/notifications/tool-input",
            "params": {"arguments": arguments},
        })

        try:
            await self._ws.send(notification)
        except Exception as e:
            log.warning("Failed to push tool input: %s", e)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @property
    def model_context(self) -> dict[str, Any] | None:
        """Get the latest model context from the app (if any)."""
        return self._model_context

    @staticmethod
    def _extract_raw_result(result: Any) -> Any:
        """Unwrap middleware/ToolCallResult wrappers to get the raw MCP result."""
        # Unwrap objects that have a .result attribute (ToolExecutionResult, etc.)
        seen: set[int] = set()
        while hasattr(result, "result") and not isinstance(result, (dict, str)):
            rid = id(result)
            if rid in seen:
                break
            seen.add(rid)
            result = result.result
        return result

    @staticmethod
    def _to_serializable(obj: Any) -> Any:
        """Convert an object to a JSON-serializable form."""
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return {k: AppBridge._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [AppBridge._to_serializable(v) for v in obj]
        # Pydantic models
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        # Dataclass / namedtuple fallback
        if hasattr(obj, "__dict__"):
            return {k: AppBridge._to_serializable(v)
                    for k, v in obj.__dict__.items() if not k.startswith("_")}
        return str(obj)

    @staticmethod
    def _extract_structured_content(out: dict[str, Any]) -> dict[str, Any]:
        """Extract structuredContent from text blocks per MCP spec.

        The MCP spec says servers SHOULD include structuredContent serialised
        as JSON inside a text content block for backwards compatibility.
        When the upstream transport loses the top-level structuredContent
        (e.g. CTP normalisation), we recover it from that text block.
        """
        if "structuredContent" in out:
            return out  # already present

        content = out.get("content")
        if not isinstance(content, list) or len(content) != 1:
            return out

        block = content[0]
        if not isinstance(block, dict) or block.get("type") != "text":
            return out

        text = block.get("text", "")
        if not isinstance(text, str) or not text.startswith("{"):
            return out

        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return out

        if not isinstance(parsed, dict):
            return out

        # Hoist structuredContent to the result level
        if "structuredContent" in parsed:
            out["structuredContent"] = parsed["structuredContent"]
            # Replace content with the inner content array if present,
            # otherwise keep the original text block
            if "content" in parsed and isinstance(parsed["content"], list):
                out["content"] = parsed["content"]

        return out

    @staticmethod
    def _format_tool_result(result: Any) -> dict[str, Any]:
        """Normalise a tool result into MCP CallToolResult structure.

        Returns ``{"content": [...], "structuredContent": {...}}`` matching
        the MCP spec's CallToolResult schema.
        """
        # Unwrap any middleware/result wrappers
        result = AppBridge._extract_raw_result(result)

        out: dict[str, Any]

        # Pydantic model with content attr — extract the content list directly
        if not isinstance(result, (dict, str)) and hasattr(result, "content"):
            content = result.content
            if isinstance(content, list):
                out = {"content": AppBridge._to_serializable(content)}
                # Preserve structuredContent / isError if present
                if hasattr(result, "structuredContent") and result.structuredContent:
                    out["structuredContent"] = AppBridge._to_serializable(
                        result.structuredContent
                    )
                if hasattr(result, "isError") and result.isError:
                    out["isError"] = True
                return AppBridge._extract_structured_content(out)

        if isinstance(result, dict):
            # If content value is an MCP SDK object, extract its content list
            content_val = result.get("content")
            if content_val is not None and not isinstance(content_val, (list, str)):
                if hasattr(content_val, "content") and isinstance(content_val.content, list):
                    result = dict(result)
                    result["content"] = content_val.content
                    # Copy structuredContent if present
                    if hasattr(content_val, "structuredContent") and content_val.structuredContent:
                        result["structuredContent"] = content_val.structuredContent
            # Make all nested values JSON-serializable
            result = AppBridge._to_serializable(result)
            if "content" in result:
                return AppBridge._extract_structured_content(result)
            return {"content": [{"type": "text", "text": json.dumps(result)}]}

        if isinstance(result, str):
            return {"content": [{"type": "text", "text": result}]}

        # Fallback
        if hasattr(result, "model_dump"):
            return {"content": [{"type": "text", "text": json.dumps(result.model_dump())}]}
        return {"content": [{"type": "text", "text": str(result)}]}
