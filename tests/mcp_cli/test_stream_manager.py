"""
StreamManager module for centralized management of server streams.

This module provides a centralized way to:
1. Initialize and maintain server connections
2. Fetch tools and resources
3. Ensure proper cleanup of streams and resources
4. Handle connection errors gracefully
5. Handle duplicate tool names across servers automatically
6. Provide cancellation and heartbeat-based resilience
"""
import asyncio
import logging
import json
from contextlib import AsyncExitStack
from typing import Dict, List, Tuple, Any, Optional

# Heartbeat interval in seconds (can be adjusted for testing)
HEARTBEAT_EVERY = 30.0

# mcp imports
from chuk_mcp.mcp_client.transport.stdio.stdio_client import stdio_client
from chuk_mcp.mcp_client.messages.initialize.send_messages import send_initialize
from chuk_mcp.mcp_client.messages.tools.send_messages import send_tools_list, send_tools_call

# Use our own config loader
from mcp_cli.config import load_config


async def send_ping(read_stream: Any, write_stream: Any) -> bool:
    """
    Default ping function to check server health. Returns True if healthy.
    """
    return True


class StreamManager:
    """
    Centralized manager for server streams and connections.
    """
    def __init__(self) -> None:
        self.streams: List[Tuple[Any, Any]] = []
        self.client_contexts: List[Any] = []
        self.server_info: List[Dict[str, Any]] = []
        self.tools: List[Dict[str, Any]] = []           # Display tools (original names)
        self.internal_tools: List[Dict[str, Any]] = []  # Namespaced tools for LLM
        self.tool_to_server_map: Dict[str, str] = {}
        self.namespaced_tool_map: Dict[str, str] = {}
        self.original_to_namespaced: Dict[str, List[str]] = {}
        self.original_to_default: Dict[str, str] = {}
        self.server_names: Dict[int, str] = {}
        self.server_streams_map: Dict[str, int] = {}
        self.active_subprocesses: set = set()
        self._exit_stack: Optional[AsyncExitStack] = None
        self._cancel_requested: bool = False
        self._hb_task: Optional[asyncio.Task] = None

    @classmethod
    async def create(
        cls, config_file: str, servers: List[str],
        server_names: Optional[Dict[int, str]] = None
    ) -> 'StreamManager':
        manager = cls()
        await manager.initialize_servers(config_file, servers, server_names)
        # Start heartbeat monitoring if any streams are active
        if manager.streams:
            manager._hb_task = asyncio.create_task(manager._heartbeat_loop())
        return manager

    async def initialize_servers(
        self,
        config_file: str,
        servers: List[str],
        server_names: Optional[Dict[int, str]] = None
    ) -> bool:
        """
        Initialize connections to the specified servers using AsyncExitStack.
        """
        self.server_names = server_names or {}
        tool_index = 0

        # Prepare an exit stack for all client contexts
        stack = AsyncExitStack()
        self._exit_stack = stack
        await stack.__aenter__()

        for i, server_name in enumerate(servers):
            display = self._get_server_display_name(i, server_name)
            logging.info(f"Initializing server: {display}")
            try:
                params = await load_config(config_file, server_name)
                client_ctx = stdio_client(params)
                self.client_contexts.append(client_ctx)
                read_stream, write_stream = await stack.enter_async_context(client_ctx)

                ok = await send_initialize(read_stream, write_stream)
                if not ok:
                    logging.error(f"Failed to initialize server {display}")
                    self.server_info.append({
                        "id": i+1,
                        "name": display,
                        "tools": 0,
                        "status": "Failed to initialize",
                        "tool_start_index": tool_index
                    })
                    continue

                fetched = await send_tools_list(read_stream, write_stream)
                tools = fetched.get("tools", [])
                display_tools: List[Dict[str, Any]] = []
                namespaced_tools: List[Dict[str, Any]] = []

                for tool in tools:
                    original = tool["name"]
                    # display copy
                    display_tools.append(tool.copy())
                    # namespaced copy
                    ns_tool = tool.copy()
                    ns_name = f"{display}_{original}"
                    ns_tool["name"] = ns_name

                    # register maps
                    self.tool_to_server_map[original] = display
                    self.namespaced_tool_map[ns_name] = original
                    if original in self.original_to_namespaced:
                        self.original_to_namespaced[original].append(ns_name)
                    else:
                        self.original_to_namespaced[original] = [ns_name]
                        self.original_to_default[original] = ns_name

                    namespaced_tools.append(ns_tool)

                self.server_streams_map[display] = len(self.streams)
                self.streams.append((read_stream, write_stream))
                self.tools.extend(display_tools)
                self.internal_tools.extend(namespaced_tools)

                self.server_info.append({
                    "id": i+1,
                    "name": display,
                    "tools": len(tools),
                    "status": "Connected",
                    "tool_start_index": tool_index
                })
                tool_index += len(tools)
                logging.info(f"Successfully initialized server: {display}")
            except Exception as e:
                logging.error(f"Error initializing server {display}: {e}")
                self.server_info.append({
                    "id": i+1,
                    "name": display,
                    "tools": 0,
                    "status": f"Error: {e}",
                    "tool_start_index": tool_index
                })

        return bool(self.streams)

    def _get_server_display_name(self, index: int, server_name: str) -> str:
        if isinstance(self.server_names, dict) and index in self.server_names:
            return self.server_names[index]
        if isinstance(self.server_names, list) and index < len(self.server_names):
            return self.server_names[index]
        return server_name or f"Server {index+1}"

    async def _heartbeat_loop(self) -> None:
        """
        Periodically ping servers to update health status.
        """
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_EVERY)
                for idx, (r, w) in enumerate(self.streams):
                    try:
                        alive = await send_ping(r, w)
                        self.server_info[idx]["status"] = "Connected" if alive else "Degraded"
                    except Exception:
                        self.server_info[idx]["status"] = "Degraded"
        except asyncio.CancelledError:
            pass

    async def request_cancel(self) -> None:
        """Request cancellation of the next tool call."""
        self._cancel_requested = True

    async def call_tool(
        self,
        tool_name: str,
        arguments: Any,
        server_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call a tool with cancellation and resolution logic.
        """
        # handle cancellation
        if self._cancel_requested:
            self._cancel_requested = False
            return {"isError": True, "error": "Cancelled by user", "content": "Cancelled by user"}

        original = tool_name
        # resolve namespaced
        if server_name and server_name != "Unknown":
            candidate = f"{server_name}_{tool_name}"
            if candidate in self.namespaced_tool_map:
                tool_name = candidate
            else:
                tool_name, server_name = self._resolve_tool_name(tool_name)
        else:
            tool_name, server_name = self._resolve_tool_name(tool_name)

        logging.debug(f"Resolved '{original}' to '{tool_name}' on '{server_name}'")

        # fallback to first server if unknown
        if server_name == "Unknown":
            if self.streams:
                first = next(iter(self.server_streams_map))
                server_name = first
            else:
                return {"isError": True, "error": f"Tool '{original}' not found", "content": f"Error: '{original}' not found"}

        idx = self.server_streams_map.get(server_name)
        if idx is None or idx >= len(self.streams):
            return {"isError": True, "error": "Invalid server index", "content": "Error: Invalid server index"}

        read_stream, write_stream = self.streams[idx]
        try:
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except Exception:
                    pass

            to_call = self.namespaced_tool_map.get(tool_name, original)
            result = await send_tools_call(read_stream, write_stream, to_call, arguments)
            if result.get("isError"):
                logging.error(f"Error calling {to_call}: {result.get('error')}")
            return result
        except Exception as e:
            logging.error(f"Exception calling tool {original}: {e}")
            return {"isError": True, "error": str(e), "content": f"Error: {e}"}

    async def close(self) -> None:
        """
        Close all streams and clean up resources.
        """
        logging.debug("Closing StreamManager resources")

        # Cancel heartbeat
        if self._hb_task:
            self._hb_task.cancel()
            try:
                await self._hb_task
            except Exception:
                pass

        # Close all client contexts via the exit stack
        if self._exit_stack is not None:
            await self._exit_stack.aclose()

        # Terminate subprocesses
        for proc in list(self.active_subprocesses):
            try:
                if proc.poll() is None:
                    proc.terminate()
                    await asyncio.to_thread(proc.wait, timeout=0.5)
                self.active_subprocesses.discard(proc)
            except Exception:
                proc.kill()

        # Clear resources
        self.streams.clear()
        self.server_streams_map.clear()
        self.client_contexts.clear()

    def _resolve_tool_name(self, tool_name: str) -> Tuple[str, str]:
        """
        Resolve a tool name to its namespaced version and server.
        """
        # Already namespaced
        if tool_name in self.namespaced_tool_map:
            srv = tool_name.split('_', 1)[0]
            return tool_name, srv

        # Original mapped
        if tool_name in self.original_to_namespaced:
            versions = self.original_to_namespaced[tool_name]
            if len(versions) > 1:
                default = self.original_to_default[tool_name]
                srv = default.split('_', 1)[0]
                logging.debug(f"Tool '{tool_name}' on multiple servers; using default {default}")
                return default, srv
            only = versions[0]
            srv = only.split('_', 1)[0]
            return only, srv

        # Not known
        return tool_name, "Unknown"

    def get_all_tools(self) -> List[Dict[str, Any]]:
        return self.tools

    def get_internal_tools(self) -> List[Dict[str, Any]]:
        return self.internal_tools

    def get_server_info(self) -> List[Dict[str, Any]]:
        return self.server_info

    def get_server_for_tool(self, tool_name: str) -> str:
        _, srv = self._resolve_tool_name(tool_name)
        return srv

    def has_tools(self) -> bool:
        return bool(self.tools)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "server_info": self.server_info,
            "tools": self.tools,
            "internal_tools": self.internal_tools,
            "tool_to_server_map": self.tool_to_server_map,
            "namespaced_tool_map": self.namespaced_tool_map,
            "original_to_namespaced": self.original_to_namespaced,
            "original_to_default": self.original_to_default,
            "server_names": self.server_names
        }