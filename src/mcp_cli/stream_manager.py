"""mcp_cli.stream_manager
========================
Centralised management of server connections, tool discovery and
remote‑tool execution with built‑in resilience:

* **AsyncExitStack** – automatic cleanup of every stdio client context.
* **Timeouts + exponential back‑off** for `initialize`, `tools_list` and
  every tool call.
* **Graceful cancellation** – UI can call :pyfunc:`request_cancel` to abort
  the next in‑flight tool invocation.
* **Heartbeat loop** – background ping task marks servers *Degraded* if they
  stop responding after initialisation.
* Automatic namespacing for duplicate tool names across servers.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from contextlib import AsyncExitStack
from subprocess import Popen
from typing import Any, Dict, List, Optional, Tuple

from chuk_mcp.mcp_client.messages.initialize.send_messages import send_initialize
from chuk_mcp.mcp_client.messages.ping.send_messages import send_ping
from chuk_mcp.mcp_client.messages.tools.send_messages import (
    send_tools_call,
    send_tools_list,
)
from chuk_mcp.mcp_client.transport.stdio.stdio_client import stdio_client

from mcp_cli.config import load_config

# ---------------------------------------------------------------------------
# Reliability tuning (override via env‑vars if desired)
# ---------------------------------------------------------------------------
INIT_TIMEOUT   = 5      # seconds
TOOLS_TIMEOUT  = 5
CALL_TIMEOUT   = 30
MAX_RETRIES    = 3
BACKOFF_FACTOR = 1.7
HEARTBEAT_EVERY = 30    # seconds
HEARTBEAT_TIMEOUT = 3   # seconds
# ---------------------------------------------------------------------------


class StreamManager:
    """Handle multiple server stdio connections and tool routing."""

    # ---------------------------------------------------------------------
    # construction / teardown
    # ---------------------------------------------------------------------

    def __init__(self) -> None:
        # live streams and contexts
        self.streams: List[Tuple[Any, Any]] = []            # (read, write)
        self.client_contexts: List[Any] = []
        self._exit_stack: Optional[AsyncExitStack] = None

        # metadata
        self.server_info: List[Dict[str, Any]] = []
        self.server_names: Dict[int, str] = {}
        self.server_streams_map: Dict[str, int] = {}        # server ➜ index

        # tool maps
        self.tools: List[Dict[str, Any]] = []               # for UI (orig)
        self.internal_tools: List[Dict[str, Any]] = []       # namespaced
        self.tool_to_server_map: Dict[str, str] = {}
        self.namespaced_tool_map: Dict[str, str] = {}        # namespaced ➜ orig
        self.original_to_namespaced: Dict[str, List[str]] = {}
        self.original_to_default: Dict[str, str] = {}

        # runtime helpers
        self.active_subprocesses: set[Popen] = set()
        self._cancel_event: asyncio.Event = asyncio.Event()
        self._hb_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # factory
    # ------------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        config_file: str,
        servers: List[str],
        server_names: Optional[Dict[int, str]] = None,
    ) -> "StreamManager":
        mgr = cls()
        if await mgr._initialize_servers(config_file, servers, server_names):
            mgr._hb_task = asyncio.create_task(mgr._heartbeat_loop())
        return mgr

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def call_tool(
        self,
        tool_name: str,
        arguments: Any,
        server_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Route a tool call to the appropriate server.

        * Handles namespacing & default‑server resolution.
        * Enforces CALL_TIMEOUT.
        * Aborts early if :pyfunc:`request_cancel` was triggered.
        """
        # cancellation check (edge triggered)
        if self._cancel_event.is_set():
            self._cancel_event.clear()
            return {
                "isError": True,
                "error": "Cancelled by user",
                "content": "Tool execution was interrupted by user",
            }

        resolved, srv = self._resolve_tool_name(tool_name, server_name)
        if srv == "Unknown":
            return {
                "isError": True,
                "error": f"Tool '{tool_name}' not found on any server",
                "content": f"Error: Tool '{tool_name}' not found on any server",
            }

        try:
            idx = self.server_streams_map[srv]
            read_stream, write_stream = self.streams[idx]
        except (KeyError, IndexError):
            return {
                "isError": True,
                "error": "Invalid server index",
                "content": "Error: Invalid server index",
            }

        # ensure args are JSON‑able dict if string provided
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass

        to_call = self.namespaced_tool_map.get(resolved, tool_name)

        try:
            result = await asyncio.wait_for(
                send_tools_call(
                    read_stream=read_stream,
                    write_stream=write_stream,
                    name=to_call,
                    arguments=arguments,
                ),
                timeout=CALL_TIMEOUT,
            )
            return result
        except asyncio.TimeoutError:
            logging.warning("Tool '%s' timed out after %ss", to_call, CALL_TIMEOUT)
            return {
                "isError": True,
                "error": "Timeout",
                "content": f"Error: Tool '{to_call}' timed out ({CALL_TIMEOUT}s)",
            }
        except Exception as exc:
            logging.error("Exception calling tool %s: %s", to_call, exc)
            return {"isError": True, "error": str(exc), "content": f"Error: {exc}"}

    async def request_cancel(self) -> None:
        """Signal the next \*in‑flight* tool call to abort as soon as possible."""
        self._cancel_event.set()

    async def close(self) -> None:
        """Close streams, heartbeat task and subprocesses."""
        if self._hb_task:
            self._hb_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._hb_task

        if self._exit_stack is not None:
            await self._exit_stack.aclose()

        # kill stray subprocesses
        for proc in list(self.active_subprocesses):
            if proc.poll() is None:
                proc.terminate()
                try:
                    await asyncio.to_thread(proc.wait, timeout=0.5)
                except Exception:
                    proc.kill()
            self.active_subprocesses.discard(proc)

        # purge references
        self.streams.clear()
        self.server_streams_map.clear()
        self.client_contexts.clear()

    # ------------------------------------------------------------------
    # initialisation helpers
    # ------------------------------------------------------------------

    async def _initialize_servers(
        self,
        config_file: str,
        servers: List[str],
        server_names: Optional[Dict[int, str]],
    ) -> bool:
        self.server_names = server_names or {}
        self._exit_stack = AsyncExitStack()
        await self._exit_stack.__aenter__()

        tool_index = 0
        for i, srv_name in enumerate(servers):
            display_name = self._display_name(i, srv_name)
            try:
                server_params = await load_config(config_file, srv_name)
                ctx = stdio_client(server_params)
                self.client_contexts.append(ctx)
                r, w = await self._exit_stack.enter_async_context(ctx)

                # ------------- initialize with retries ---------------
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        ok = await asyncio.wait_for(
                            send_initialize(r, w), timeout=INIT_TIMEOUT
                        )
                        if ok:
                            break
                    except asyncio.TimeoutError:
                        logging.warning(
                            "%s: initialize timeout – retry %d/%d",
                            display_name, attempt, MAX_RETRIES,
                        )
                        await asyncio.sleep(BACKOFF_FACTOR ** attempt)
                else:
                    raise RuntimeError("Timeout waiting for initialization response")

                # ------------- fetch tools ---------------------------
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        fetched = await asyncio.wait_for(
                            send_tools_list(r, w), timeout=TOOLS_TIMEOUT
                        )
                        break
                    except asyncio.TimeoutError:
                        logging.warning(
                            "%s: tools_list timeout – retry %d/%d",
                            display_name, attempt, MAX_RETRIES,
                        )
                        await asyncio.sleep(BACKOFF_FACTOR ** attempt)
                else:
                    raise RuntimeError("Timeout waiting for tools list")

                tools = fetched.get("tools", [])
                self._register_tools(display_name, tools, tool_index)
                tool_index += len(tools)

                self.server_streams_map[display_name] = len(self.streams)
                self.streams.append((r, w))
                self.server_info.append({
                    "id": i + 1,
                    "name": display_name,
                    "tools": len(tools),
                    "status": "Connected",
                    "tool_start_index": tool_index,
                })
                logging.info("Successfully initialised %s", display_name)
            except Exception as exc:
                logging.error("Error initialising %s: %s", display_name, exc)
                self.server_info.append({
                    "id": i + 1,
                    "name": display_name,
                    "tools": 0,
                    "status": f"Error: {exc}",
                    "tool_start_index": tool_index,
                })
        return bool(self.streams)

    # ------------------------------------------------------------------
    # helper registration
    # ------------------------------------------------------------------

    def _register_tools(self, srv_display: str, tools: List[Dict[str, Any]], start_idx: int) -> None:
        disp, internal = [], []
        for tool in tools:
            orig = tool["name"]
            namespaced = f"{srv_display}_{orig}"

            disp.append(tool.copy())
            internal_tool = tool.copy(); internal_tool["name"] = namespaced
            internal.append(internal_tool)

            self.tool_to_server_map[orig] = srv_display
            self.namespaced_tool_map[namespaced] = orig
            self.original_to_namespaced.setdefault(orig, []).append(namespaced)
            self.original_to_default.setdefault(orig, namespaced)

        self.tools.extend(disp)
        self.internal_tools.extend(internal)

    # ------------------------------------------------------------------
    # heart‑beat
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self):
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_EVERY)
                for srv, idx in list(self.server_streams_map.items()):
                    r, w = self.streams[idx]
                    try:
                        ok = await asyncio.wait_for(send_ping(r, w), timeout=HEARTBEAT_TIMEOUT)
                        self._mark_status(srv, "Connected" if ok else "Degraded")
                    except Exception:
                        self._mark_status(srv, "Degraded")
        except asyncio.CancelledError:
            pass

    def _mark_status(self, srv: str, status: str):
        for entry in self.server_info:
            if entry["name"] == srv and entry["status"] != status:
                entry["status"] = status
                logging.info("Server %s → %s", srv, status)
                break

    # ------------------------------------------------------------------
    # name helpers
    # ------------------------------------------------------------------

    def _display_name(self, idx: int, srv_name: str) -> str:
        if isinstance(self.server_names, dict):
            return self.server_names.get(idx, srv_name)
        if isinstance(self.server_names, list) and idx < len(self.server_names):
            return self.server_names[idx]
        return srv_name

    def _resolve_tool_name(self, tool: str, forced_srv: Optional[str] = None):
        if forced_srv and forced_srv != "Unknown":
            candidate = f"{forced_srv}_{tool}"
            if candidate in self.namespaced_tool_map:
                return candidate, forced_srv

        if tool in self.namespaced_tool_map:
            srv = tool.split("_", 1)[0]
            return tool, srv

        if tool in self.original_to_namespaced:
            ns_list = self.original_to_namespaced[tool]
            chosen = self.original_to_default[tool]
            if len(ns_list) > 1:
                logging.debug("%s on multiple servers; default → %s", tool, chosen)
            return chosen, chosen.split("_", 1)[0]

        return tool, "Unknown"

    # ------------------------------------------------------------------
    # convenience accessors (unchanged)
    # ------------------------------------------------------------------

    def get_all_tools(self):
        return self.tools

    def get_internal_tools(self):
        return self.internal_tools

    def get_server_info(self):
        return self.server_info

    def get_server_for_tool(self, tool_name: str) -> str:
        _, srv = self._resolve_tool_name(tool_name)
        return srv

    def has_tools(self):
        return bool(self.tools)

    def to_dict(self):
        return {
            "server_info": self.server_info,
            "tools": self.tools,
            "internal_tools": self.internal_tools,
            "tool_to_server_map": self.tool_to_server_map,
            "namespaced_tool_map": self.namespaced_tool_map,
            "original_to_namespaced": self.original_to_namespaced,
            "original_to_default": self.original_to_default,
            "server_names": self.server_names,
        }
