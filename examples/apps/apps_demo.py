#!/usr/bin/env python
"""
MCP Apps demo — launches a self-contained MCP App in the browser.

This example demonstrates:
1. How _meta.ui is preserved on tool definitions
2. How the AppHostServer serves an MCP App
3. How the WebSocket bridge connects browser to backend
4. The full JSON-RPC handshake (ui/initialize, tools/call)

No real MCP server is needed — the demo uses mocks to simulate
the tool pipeline end-to-end.

Usage:
    uv run python examples/apps/apps_demo.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# ── Sample app HTML ──────────────────────────────────────────────────────
# A minimal MCP App that implements the required handshake and shows
# a simple interactive UI.  It can call `tools/call` back to the host.

SAMPLE_APP_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Demo App</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      background: #f8f9fa; color: #212529;
      padding: 24px; max-width: 640px; margin: 0 auto;
    }
    h1 { font-size: 22px; margin-bottom: 8px; }
    p { margin-bottom: 12px; color: #666; }
    .card {
      background: #fff; border-radius: 8px; padding: 16px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.12); margin-bottom: 16px;
    }
    .card h2 { font-size: 16px; margin-bottom: 8px; color: #333; }
    button {
      background: #0d6efd; color: #fff; border: none;
      padding: 8px 16px; border-radius: 6px; cursor: pointer;
      font-size: 14px; margin-right: 8px; margin-bottom: 8px;
    }
    button:hover { background: #0b5ed7; }
    button:disabled { background: #6c757d; cursor: not-allowed; }
    #output {
      background: #1a1a2e; color: #4ecca3; padding: 12px;
      border-radius: 6px; font-family: monospace; font-size: 13px;
      white-space: pre-wrap; min-height: 80px; margin-top: 12px;
    }
    #status {
      display: inline-block; padding: 4px 10px; border-radius: 12px;
      font-size: 12px; background: #dc3545; color: #fff; margin-bottom: 12px;
    }
    #status.ok { background: #198754; }
  </style>
</head>
<body>
  <h1>MCP Apps Demo</h1>
  <p>This app runs in a sandboxed iframe and communicates with
     the host via JSON-RPC over postMessage.</p>
  <div id="status">Initializing...</div>

  <div class="card">
    <h2>Call a tool</h2>
    <p>Click a button to call a tool on the MCP server via the host bridge:</p>
    <button id="btn-echo" disabled>Echo</button>
    <button id="btn-time" disabled>Get Time</button>
    <button id="btn-calc" disabled>Calculate 6 &times; 7</button>
  </div>

  <div class="card">
    <h2>Output</h2>
    <div id="output">Waiting for initialization...</div>
  </div>

<script>
(function() {
  "use strict";

  var output  = document.getElementById("output");
  var status  = document.getElementById("status");
  var nextId  = 1;
  var pending = {};
  var buttons = {
    echo: document.getElementById("btn-echo"),
    time: document.getElementById("btn-time"),
    calc: document.getElementById("btn-calc"),
  };

  function log(msg) {
    output.textContent += msg + "\n";
  }

  function sendRequest(method, params) {
    return new Promise(function(resolve) {
      var id = nextId++;
      pending[id] = resolve;
      window.parent.postMessage(
        { jsonrpc: "2.0", id: id, method: method, params: params },
        "*"
      );
    });
  }

  function sendNotification(method, params) {
    window.parent.postMessage(
      { jsonrpc: "2.0", method: method, params: params || {} },
      "*"
    );
  }

  // Listen for messages from host
  window.addEventListener("message", function(ev) {
    var msg = ev.data;
    if (!msg || msg.jsonrpc !== "2.0") return;

    // Response to our request
    if (msg.id != null && pending[msg.id]) {
      pending[msg.id](msg);
      delete pending[msg.id];
      return;
    }

    // Notification from host
    if (msg.method === "ui/notifications/tool-result") {
      log("[notification] tool-result: " + JSON.stringify(msg.params, null, 2));
    }
    if (msg.method === "ui/notifications/tool-input") {
      log("[notification] tool-input: " + JSON.stringify(msg.params, null, 2));
    }
  });

  // ── Initialize ──
  async function init() {
    log("Sending ui/initialize...");
    var resp = await sendRequest("ui/initialize", {
      protocolVersion: "2026-01-26",
      appInfo: { name: "demo-app", version: "1.0" },
      capabilities: {}
    });
    log("Host responded: " + resp.result.hostInfo.name +
        " v" + resp.result.hostInfo.version);
    status.textContent = "Connected";
    status.className = "ok";

    // Notify host we're initialized
    sendNotification("ui/notifications/initialized");

    // Enable buttons
    buttons.echo.disabled = false;
    buttons.time.disabled = false;
    buttons.calc.disabled = false;
    log("Ready — click a button to call a tool.\n");
  }

  // ── Button handlers ──
  buttons.echo.onclick = async function() {
    log("Calling tools/call echo...");
    var resp = await sendRequest("tools/call", {
      name: "echo", arguments: { message: "Hello from MCP App!" }
    });
    if (resp.error) {
      log("Error: " + resp.error.message);
    } else {
      log("Result: " + JSON.stringify(resp.result, null, 2));
    }
  };

  buttons.time.onclick = async function() {
    log("Calling tools/call get_time...");
    var resp = await sendRequest("tools/call", {
      name: "get_time", arguments: {}
    });
    if (resp.error) {
      log("Error: " + resp.error.message);
    } else {
      log("Result: " + JSON.stringify(resp.result, null, 2));
    }
  };

  buttons.calc.onclick = async function() {
    log("Calling tools/call calculate...");
    var resp = await sendRequest("tools/call", {
      name: "calculate", arguments: { operation: "multiply", a: 6, b: 7 }
    });
    if (resp.error) {
      log("Error: " + resp.error.message);
    } else {
      log("Result: " + JSON.stringify(resp.result, null, 2));
    }
  };

  // Boot
  init();
})();
</script>
</body>
</html>"""


# ── Mock tool manager ────────────────────────────────────────────────────

def make_mock_tool_manager() -> MagicMock:
    """Create a mock ToolManager that simulates tool execution."""
    tm = MagicMock()

    # Simulate read_resource returning our sample HTML
    async def mock_read_resource(uri: str, server_name: str | None = None) -> dict[str, Any]:
        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": "text/html",
                    "text": SAMPLE_APP_HTML,
                }
            ]
        }

    tm.read_resource = AsyncMock(side_effect=mock_read_resource)

    # Simulate tool execution
    class FakeResult:
        def __init__(self, result: Any) -> None:
            self.success = True
            self.result = result
            self.error = None

    async def mock_execute(
        tool_name: str, arguments: dict[str, Any], namespace: str = ""
    ) -> FakeResult:
        import datetime

        if tool_name == "echo":
            return FakeResult({"echoed": arguments.get("message", "")})
        if tool_name == "get_time":
            return FakeResult({"time": datetime.datetime.now().isoformat()})
        if tool_name == "calculate":
            a = arguments.get("a", 0)
            b = arguments.get("b", 0)
            op = arguments.get("operation", "add")
            ops = {"add": a + b, "subtract": a - b, "multiply": a * b}
            return FakeResult({"result": ops.get(op, 0), "operation": op})
        return FakeResult({"error": f"Unknown tool: {tool_name}"})

    tm.execute_tool = AsyncMock(side_effect=mock_execute)

    return tm


# ── Main ─────────────────────────────────────────────────────────────────

async def main() -> None:
    from mcp_cli.apps.host import AppHostServer

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    tool_manager = make_mock_tool_manager()
    host = AppHostServer(tool_manager)

    print()
    print("=" * 60)
    print("  MCP Apps Demo")
    print("=" * 60)
    print()
    print("  Launching a sample MCP App in your browser.")
    print("  The app connects via WebSocket and can call tools")
    print("  back through the bridge to the (mock) MCP server.")
    print()
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60)
    print()

    try:
        app_info = await host.launch_app(
            tool_name="demo-app",
            resource_uri="ui://demo-app/index.html",
            server_name="demo-server",
        )
        print(f"  App running at: {app_info.url}")
        print()

        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n  Shutting down...")
    finally:
        await host.close_all()
        print("  Done.")


if __name__ == "__main__":
    asyncio.run(main())
