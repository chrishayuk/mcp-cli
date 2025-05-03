# mcp_cli/tools/manager.py
"""
Centralized tool management using CHUK Tool Processor.

This module provides a unified interface for all tool-related operations in MCP CLI,
abstracting away the underlying CHUK Tool Processor implementation details.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Any

from chuk_tool_processor.mcp import setup_mcp_stdio
from chuk_tool_processor.core.processor import ToolProcessor
from chuk_tool_processor.registry import ToolRegistryProvider
from chuk_tool_processor.mcp.stream_manager import StreamManager

from mcp_cli.tools.models import ToolInfo, ServerInfo, ToolCallResult


class ToolManager:
    """
    Central interface for all tool operations in MCP CLI.

    This class wraps CHUK Tool Processor and provides a clean API for:
    - Tool discovery and listing
    - Tool execution
    - Server management
    - Tool conversion for LLMs
    """

    # ------------------------------------------------------------------ #
    # construction / initialisation
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        config_file: str,
        servers: List[str],
        server_names: Optional[Dict[int, str]] = None,
    ):
        self.config_file = config_file
        self.servers = servers
        self.server_names = server_names or {}

        self.processor: Optional[ToolProcessor] = None
        self.stream_manager: Optional[StreamManager] = None
        self._registry = None

    async def initialize(self, namespace: str = "stdio") -> bool:
        """Connect to the MCP servers and populate the tool registry."""
        try:
            self.processor, self.stream_manager = await setup_mcp_stdio(
                config_file=self.config_file,
                servers=self.servers,
                server_names=self.server_names,
                namespace=namespace,
                enable_caching=True,
                enable_retries=True,
                max_retries=3,
            )
            self._registry = ToolRegistryProvider.get_registry()
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"[red]Error initializing tool manager: {exc}[/red]")
            return False

    async def close(self):
        if self.stream_manager:
            await self.stream_manager.close()

    # ------------------------------------------------------------------ #
    # discovery helpers
    # ------------------------------------------------------------------ #
    def _all_registry_items(self):
        return self._registry.list_tools() if self._registry else []

    def _metadata(self, name: str, ns: str):
        return self._registry.get_metadata(name, ns) if self._registry else None

    def get_all_tools(self) -> List[ToolInfo]:
        """Return *every* tool (including duplicates)."""
        tools: List[ToolInfo] = []
        for ns, name in self._all_registry_items():
            md = self._metadata(name, ns)
            if md:
                tools.append(
                    ToolInfo(
                        name=name,
                        namespace=ns,
                        description=md.description,
                        parameters=md.argument_schema,
                        is_async=md.is_async,
                        tags=list(md.tags),
                    )
                )
        return tools

    def get_unique_tools(self) -> List[ToolInfo]:
        """Return tools without duplicates from the default namespace."""
        seen, unique = set(), []
        for ns, name in self._all_registry_items():
            if ns == "default" or name in seen:
                continue
            seen.add(name)
            md = self._metadata(name, ns)
            if md:
                unique.append(
                    ToolInfo(
                        name=name,
                        namespace=ns,
                        description=md.description,
                        parameters=md.argument_schema,
                        is_async=md.is_async,
                        tags=list(md.tags),
                    )
                )
        return unique

    def get_tool_by_name(self, tool_name: str, namespace: str | None = None) -> Optional[ToolInfo]:
        if not self._registry:
            return None

        if namespace:  # explicit ns
            md = self._metadata(tool_name, namespace)
            if md:
                return ToolInfo(
                    name=tool_name,
                    namespace=namespace,
                    description=md.description,
                    parameters=md.argument_schema,
                    is_async=md.is_async,
                    tags=list(md.tags),
                )

        # otherwise search all non-default namespaces
        for ns, name in self._all_registry_items():
            if name == tool_name and ns != "default":
                return self.get_tool_by_name(name, ns)
        return None

    # ------------------------------------------------------------------ #
    # execution helper
    # ------------------------------------------------------------------ #
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCallResult:
        """Run *tool_name* with *arguments* and return a ToolCallResult."""
        if not self.processor:
            return ToolCallResult(tool_name, False, error="Tool manager not initialized")

        # Resolve the fully-qualified call name expected by CHUK
        if self._registry and self._metadata(tool_name, "stdio"):
            call_name = f"stdio.{tool_name}"
        else:
            call_name = tool_name

        try:
            llm_text = f'<tool name="{call_name}" args=\'{arguments}\'/>'
            results = await self.processor.process_text(llm_text)
        except Exception as exc:  # noqa: BLE001
            return ToolCallResult(tool_name, False, error=str(exc))

        if not results:
            return ToolCallResult(tool_name, False, error="No result returned")

        r = results[0]
        return ToolCallResult(
            tool_name=tool_name,
            success=not bool(r.error),
            result=r.result,
            error=r.error,
            execution_time=(r.end_time - r.start_time).total_seconds(),
        )

    # ------------------------------------------------------------------ #
    # server / stream helpers
    # ------------------------------------------------------------------ #
    def _extract_namespace(self, server_name: str) -> str:
        return server_name.split("_", 1)[0] if "_" in server_name else server_name

    def get_server_info(self) -> List[ServerInfo]:
        if not self.stream_manager:
            return []
        infos = []
        for raw in self.stream_manager.get_server_info():
            infos.append(
                ServerInfo(
                    id=raw.get("id", 0),
                    name=raw.get("name", "Unknown"),
                    status=raw.get("status", "Unknown"),
                    tool_count=raw.get("tools", 0),
                    namespace=self._extract_namespace(raw.get("name", "")),
                )
            )
        return infos

    def get_server_for_tool(self, tool_name: str) -> Optional[str]:
        if "." in tool_name:
            return tool_name.split(".", 1)[0]
        # look up via registry
        for ns, name in self._all_registry_items():
            if name == tool_name and ns != "default":
                return ns
        # fallback to stream-manager map
        if self.stream_manager:
            return self.stream_manager.get_server_for_tool(tool_name)
        return None

    # ------------------------------------------------------------------ #
    # LLM helpers
    # ------------------------------------------------------------------ #
    def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        if not self._registry:
            return []
        unique = self.get_unique_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.parameters or {},
                },
            }
            for t in unique
        ]

    # ------------------------------------------------------------------ #
    # compatibility shim  (NEW)
    # ------------------------------------------------------------------ #
    def get_streams(self):
        """
        Legacy helper so commands like **/resources** and **/prompts** that
        expect the low-level *StreamManager* continue to work when they
        receive a *ToolManager* instance.
        """
        if self.stream_manager and hasattr(self.stream_manager, "get_streams"):
            return self.stream_manager.get_streams()
        return []

    # ------------------------------------------------------------------ #
    # convenience for NL queries
    # ------------------------------------------------------------------ #
    def parse_natural_language_tool(self, text: str) -> Optional[str]:
        conversions = {
            "list tables": "list_tables",
            "read query": "read_query",
            "write query": "write_query",
            "create table": "create_table",
            "describe table": "describe_table",
            "append insight": "append_insight",
        }
        lowered = text.lower().strip()
        for phrase, tool in conversions.items():
            if phrase in lowered or phrase.replace(" ", "") in lowered:
                return tool
        return None


# ---------------------------------------------------------------------- #
# global singleton accessor (unchanged)
# ---------------------------------------------------------------------- #
_tool_manager: Optional[ToolManager] = None


def get_tool_manager() -> Optional[ToolManager]:
    return _tool_manager


def set_tool_manager(manager: ToolManager):
    global _tool_manager
    _tool_manager = manager
