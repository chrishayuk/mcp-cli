#!/usr/bin/env python
"""
Demonstrates the AppBridge JSON-RPC protocol handler.

Shows how messages from the browser are routed to MCP server
tool calls, and how results flow back. No aiohttp or browser needed.

Usage:
    uv run python examples/apps/bridge_protocol_demo.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from mcp_cli.apps.bridge import AppBridge
from mcp_cli.apps.models import AppInfo


def make_mock_tool_manager() -> MagicMock:
    """Create a mock ToolManager."""
    tm = MagicMock()

    class FakeResult:
        def __init__(self, result: Any) -> None:
            self.success = True
            self.result = result
            self.error = None

    async def mock_execute(
        tool_name: str, arguments: dict[str, Any], namespace: str = ""
    ) -> FakeResult:
        if tool_name == "echo":
            return FakeResult({"echoed": arguments.get("message", "")})
        return FakeResult({"error": f"Unknown tool: {tool_name}"})

    tm.execute_tool = AsyncMock(side_effect=mock_execute)
    return tm


async def main() -> None:
    print()
    print("=" * 60)
    print("  AppBridge Protocol Demo")
    print("=" * 60)
    print()

    app_info = AppInfo(
        tool_name="demo-app",
        resource_uri="ui://demo/app.html",
        server_name="demo-server",
        port=9470,
    )
    tool_manager = make_mock_tool_manager()
    bridge = AppBridge(app_info, tool_manager)

    # ── 1. tools/call request ──
    print("  1. Handling tools/call request:")
    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "echo",
                "arguments": {"message": "Hello from the app!"},
            },
        }
    )
    print(f"     Request:  {request}")
    response = await bridge.handle_message(request)
    parsed = json.loads(response)
    print(f"     Response: {json.dumps(parsed, indent=2)}")
    print()

    # ── 2. ui/message notification ──
    print("  2. Handling ui/message request:")
    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "ui/message",
            "params": {"content": {"text": "User selected option A"}},
        }
    )
    response = await bridge.handle_message(request)
    parsed = json.loads(response)
    print(f"     Response: {json.dumps(parsed, indent=2)}")
    print()

    # ── 3. ui/update-model-context ──
    print("  3. Handling ui/update-model-context request:")
    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "ui/update-model-context",
            "params": {"selectedItems": ["chart-1", "chart-2"]},
        }
    )
    response = await bridge.handle_message(request)
    print(f"     Stored model context: {bridge.model_context}")
    print()

    # ── 4. ui/notifications/initialized (notification, no id) ──
    print("  4. Handling ui/notifications/initialized notification:")
    print(f"     State before: {app_info.state.value}")
    notification = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "ui/notifications/initialized",
            "params": {},
        }
    )
    response = await bridge.handle_message(notification)
    print(f"     Response: {response} (None = notification, no reply)")
    print(f"     State after: {app_info.state.value}")
    print()

    # ── 5. Unknown method (request with id → error response) ──
    print("  5. Handling unknown method (error response):")
    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "unknown/method",
            "params": {},
        }
    )
    response = await bridge.handle_message(request)
    parsed = json.loads(response)
    print(f"     Response: {json.dumps(parsed, indent=2)}")
    print()

    # ── 6. Invalid JSON ──
    print("  6. Handling invalid JSON:")
    response = await bridge.handle_message("this is not json")
    print(f"     Response: {response} (None = silently ignored)")
    print()

    print("=" * 60)
    print("  All protocol paths demonstrated.")
    print("=" * 60)
    print()


if __name__ == "__main__":
    asyncio.run(main())
