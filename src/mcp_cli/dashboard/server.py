# mcp_cli/dashboard/server.py
"""Dashboard HTTP + WebSocket server.

Serves the shell.html and static assets on a single port using the
``websockets`` library's ``process_request`` hook — the same pattern used
by ``mcp_cli.apps.host``.

WebSocket endpoint: /ws
Static files:       / → shell.html
                    /views/<name>.html
                    /themes/themes.json
"""

from __future__ import annotations

import asyncio
import http
import json
import logging
import mimetypes
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import websockets
    from websockets.asyncio.server import serve as ws_serve, ServerConnection
    from websockets.http11 import Request, Response
except ImportError:  # pragma: no cover
    raise ImportError(
        "Dashboard support requires websockets. Install with:  pip install mcp-cli[dashboard]"
    )

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


class DashboardServer:
    """Local HTTP + WebSocket server for the mcp-cli dashboard shell."""

    def __init__(self) -> None:
        self._clients: set[ServerConnection] = set()
        self._server: Any = None
        # Called when a browser user sends USER_MESSAGE / USER_ACTION / REQUEST_TOOL
        self.on_browser_message: Callable[[dict[str, Any]], Any] | None = None
        # Called when a new WebSocket client connects (before message loop starts)
        self.on_client_connected: Callable[[Any], Any] | None = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    async def start(self, port: int = 0) -> int:
        """Find an available port and start the server. Returns the bound port."""
        bound_port = await self._find_port(port)

        self._server = await ws_serve(
            self._ws_handler,
            "localhost",
            bound_port,
            process_request=self._process_request,
        )
        logger.info("Dashboard server started on port %d", bound_port)
        return bound_port

    async def stop(self) -> None:
        """Shut down the server and close all client connections."""
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception as exc:
                logger.debug("Error during dashboard server shutdown: %s", exc)
            self._server = None
        self._clients.clear()

    async def broadcast(self, msg: dict[str, Any]) -> None:
        """Send a JSON message to all connected WebSocket clients."""
        if not self._clients:
            return
        payload = json.dumps(msg)
        dead: list[ServerConnection] = []
        for client in list(self._clients):
            try:
                await client.send(payload)
            except Exception:
                dead.append(client)
        for c in dead:
            self._clients.discard(c)

    # ------------------------------------------------------------------ #
    #  WebSocket handler                                                  #
    # ------------------------------------------------------------------ #

    async def _ws_handler(self, ws: ServerConnection) -> None:
        self._clients.add(ws)
        logger.debug(
            "Dashboard WebSocket client connected (%d total)", len(self._clients)
        )
        if self.on_client_connected is not None:
            try:
                result = self.on_client_connected(ws)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.debug("Error in client connected callback: %s", exc)
        try:
            async for raw in ws:
                if isinstance(raw, str):
                    await self._handle_browser_message(raw)
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            logger.debug(
                "Dashboard WebSocket client disconnected (%d remain)",
                len(self._clients),
            )

    async def _handle_browser_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Dashboard received invalid JSON: %.200s", raw)
            return
        if self.on_browser_message:
            try:
                result = self.on_browser_message(msg)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                logger.warning("Error in dashboard browser message handler: %s", exc)

    # ------------------------------------------------------------------ #
    #  HTTP request handler                                               #
    # ------------------------------------------------------------------ #

    def _process_request(
        self, connection: ServerConnection, request: Request
    ) -> Response | None:
        path = request.path.split("?")[0]  # strip query string

        # WebSocket upgrade — let the library handle it
        if path == "/ws":
            return None

        # Serve static files
        file_path = self._resolve_static(path)
        if file_path is None:
            body = b"Not Found"
            return Response(
                http.HTTPStatus.NOT_FOUND,
                "Not Found",
                websockets.Headers({"Content-Length": str(len(body))}),
                body,
            )

        try:
            data = file_path.read_bytes()
        except OSError as exc:
            logger.warning("Could not read static file %s: %s", file_path, exc)
            body = b"Internal Server Error"
            return Response(
                http.HTTPStatus.INTERNAL_SERVER_ERROR,
                "Internal Server Error",
                websockets.Headers({"Content-Length": str(len(body))}),
                body,
            )

        mime, _ = mimetypes.guess_type(str(file_path))
        content_type = mime or "application/octet-stream"
        if content_type.startswith("text/"):
            content_type += "; charset=utf-8"

        return Response(
            http.HTTPStatus.OK,
            "OK",
            websockets.Headers(
                {
                    "Content-Type": content_type,
                    "Content-Length": str(len(data)),
                    "Cache-Control": "no-cache",
                }
            ),
            data,
        )

    def _resolve_static(self, path: str) -> Path | None:
        """Map a URL path to a file in the static directory. Returns None if not found."""
        if path in ("", "/"):
            candidate = _STATIC_DIR / "shell.html"
        elif path.startswith("/views/"):
            name = path[len("/views/") :]
            candidate = _STATIC_DIR / "views" / name
        elif path.startswith("/themes/"):
            name = path[len("/themes/") :]
            candidate = _STATIC_DIR / "themes" / name
        else:
            # Reject unknown paths
            return None

        # Safety: ensure the resolved path is inside _STATIC_DIR
        try:
            candidate.resolve().relative_to(_STATIC_DIR.resolve())
        except ValueError:
            return None

        if candidate.exists() and candidate.is_file():
            return candidate
        return None

    # ------------------------------------------------------------------ #
    #  Port selection                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    async def _find_port(preferred: int) -> int:
        from mcp_cli.config.defaults import DEFAULT_DASHBOARD_PORT_START

        start = preferred if preferred > 0 else DEFAULT_DASHBOARD_PORT_START
        for port in range(start, start + 20):
            try:
                server = await asyncio.start_server(
                    lambda r, w: None, "localhost", port
                )
                server.close()
                await server.wait_closed()
                return port
            except OSError:
                continue
        raise RuntimeError(
            f"Could not find an available port in range {start}–{start + 19}"
        )
