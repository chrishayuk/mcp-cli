# mcp_cli/tools/manager.py
"""
ToolManager - orchestrates chuk-tool-processor with mcp-cli features.

Responsibilities:
1. Parse MCP config files and detect server types (HTTP/SSE/STDIO)
2. Integrate with mcp-cli's OAuth TokenManager
3. Filter and validate tools for LLM compatibility
4. Convert between chuk and mcp-cli data models

For direct tool operations, use the exposed stream_manager property.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from chuk_tool_processor import StreamManager, ToolProcessor
from chuk_tool_processor import ToolCall as CTPToolCall
from chuk_tool_processor import ToolResult as CTPToolResult

from mcp_cli.config import RuntimeConfig, TimeoutType, load_runtime_config
from mcp_cli.constants import ServerStatus
from mcp_cli.llm.content_models import ContentBlockType
from mcp_cli.tools.config_loader import ConfigLoader
from mcp_cli.tools.dynamic_tools import DynamicToolProvider
from mcp_cli.tools.execution import (
    execute_tools_parallel as _execute_tools_parallel,
    stream_execute_tools as _stream_execute_tools,
)
from mcp_cli.tools.filter import DisabledReason, ToolFilter
from mcp_cli.tools.models import (
    ServerInfo,
    ToolCallResult,
    ToolDefinitionInput,
    ToolInfo,
    TransportType,
)

logger = logging.getLogger(__name__)


class ToolManager:
    """
    Facade over chuk-tool-processor with mcp-cli specific features.

    Responsibilities:
    1. Parse MCP config files and detect server types (HTTP/SSE/STDIO)
    2. Integrate with mcp-cli's OAuth TokenManager
    3. Filter and validate tools for LLM compatibility
    4. Convert between chuk and mcp-cli data models

    For direct tool operations, use the exposed stream_manager property.
    """

    def __init__(
        self,
        config_file: str,
        servers: list[str],
        server_names: dict[int, str | None] | None = None,
        tool_timeout: float | None = None,
        max_concurrency: int = 4,
        initialization_timeout: float = 120.0,
        runtime_config: RuntimeConfig | None = None,
    ):
        self.config_file = config_file
        self.servers = servers
        self.server_names = server_names or {}

        # Use runtime config for timeout management (type-safe!)
        self.runtime_config = runtime_config or load_runtime_config()

        # Tool timeout with priority: param > runtime_config
        if tool_timeout is not None:
            self.tool_timeout = tool_timeout
        else:
            self.tool_timeout = self.runtime_config.get_timeout(
                TimeoutType.TOOL_EXECUTION
            )

        # Initialization timeout with priority: param > runtime_config
        if initialization_timeout != 120.0:  # User provided non-default
            self.initialization_timeout = initialization_timeout
        else:
            self.initialization_timeout = self.runtime_config.get_timeout(
                TimeoutType.SERVER_INIT
            )

        self.max_concurrency = max_concurrency

        logger.debug(
            f"ToolManager initialized with timeouts: "
            f"tool={self.tool_timeout}s, init={self.initialization_timeout}s"
        )

        # chuk-tool-processor components (publicly accessible)
        self.stream_manager: StreamManager | None = None
        self.processor: ToolProcessor | None = None
        self._registry = None

        # Config loader handles parsing and OAuth
        self._config_loader = ConfigLoader(config_file, servers)

        # mcp-cli features
        self.tool_filter = ToolFilter()

        # Setup dynamic tool provider for on-demand tool discovery
        self.dynamic_tool_provider = DynamicToolProvider(self)

    # ================================================================
    # Initialization
    # ================================================================

    async def initialize(self, namespace: str = "stdio") -> bool:
        """Initialize by parsing config, setting up OAuth, and creating StreamManager."""
        try:
            from chuk_term.ui import output

            # Load config and detect server types
            config = self._config_loader.load()
            if not config:
                output.warning("No config found, initializing with empty toolset")
                return await self._setup_empty_toolset()

            self._config_loader.detect_server_types(config)

            # Initialize StreamManager based on detected types
            success = await self._initialize_stream_manager(namespace)

            if success and self.stream_manager:
                # Get registry/processor from StreamManager if available
                if hasattr(self.stream_manager, "registry"):
                    self._registry = self.stream_manager.registry
                if hasattr(self.stream_manager, "processor"):
                    self.processor = self.stream_manager.processor

                logger.info("ToolManager initialized successfully")

            return success

        except Exception as e:
            logger.error(f"Failed to initialize ToolManager: {e}", exc_info=True)
            return False

    async def _initialize_stream_manager(self, namespace: str) -> bool:
        """Initialize StreamManager with detected transport type.

        ENHANCED: Initializes different transport types in parallel for faster startup.
        """
        self.stream_manager = StreamManager()

        http_servers = self._config_loader.http_servers
        sse_servers = self._config_loader.sse_servers
        stdio_servers = self._config_loader.stdio_servers

        if not (http_servers or sse_servers or stdio_servers):
            logger.info("No servers detected")
            return True

        try:
            # Build initialization tasks for parallel execution
            init_tasks: list[asyncio.Task[None]] = []
            task_names: list[str] = []

            # Create OAuth callback once (shared by HTTP and SSE)
            oauth_callback = self._config_loader.create_oauth_refresh_callback(
                http_servers, sse_servers
            )

            if http_servers:
                logger.info(f"Preparing {len(http_servers)} HTTP servers for init")
                http_dicts = [
                    {"name": s.name, "url": s.url, "headers": s.headers or {}}
                    for s in http_servers
                ]
                task = asyncio.create_task(
                    self.stream_manager.initialize_with_http_streamable(
                        servers=http_dicts,
                        server_names=self.server_names,
                        initialization_timeout=self.initialization_timeout,
                        oauth_refresh_callback=oauth_callback,
                    ),
                    name="init_http",
                )
                init_tasks.append(task)
                task_names.append("HTTP")

            if sse_servers:
                logger.info(f"Preparing {len(sse_servers)} SSE servers for init")
                sse_dicts = [
                    {"name": s.name, "url": s.url, "headers": s.headers or {}}
                    for s in sse_servers
                ]
                task = asyncio.create_task(
                    self.stream_manager.initialize_with_sse(
                        servers=sse_dicts,
                        server_names=self.server_names,
                        initialization_timeout=self.initialization_timeout,
                        oauth_refresh_callback=oauth_callback,
                    ),
                    name="init_sse",
                )
                init_tasks.append(task)
                task_names.append("SSE")

            if stdio_servers:
                logger.info(f"Preparing {len(stdio_servers)} STDIO servers for init")
                stdio_dicts = [
                    {
                        "name": s.name,
                        "command": s.command,
                        "args": s.args,
                        "env": s.env,
                    }
                    for s in stdio_servers
                ]
                task = asyncio.create_task(
                    self.stream_manager.initialize_with_stdio(
                        servers=stdio_dicts,
                        server_names=self.server_names,
                        initialization_timeout=self.initialization_timeout,
                    ),
                    name="init_stdio",
                )
                init_tasks.append(task)
                task_names.append("STDIO")

            # Run all transport initializations in parallel
            if init_tasks:
                logger.info(
                    f"Starting parallel initialization of {len(init_tasks)} transport types: {', '.join(task_names)}"
                )
                results = await asyncio.gather(*init_tasks, return_exceptions=True)

                # Check for errors
                errors = []
                for name, result in zip(task_names, results):
                    if isinstance(result, Exception):
                        errors.append(f"{name}: {result}")
                        logger.error(f"{name} initialization failed: {result}")

                if errors:
                    # Log errors but don't fail if at least one transport succeeded
                    logger.warning(
                        f"Some transports failed to initialize: {'; '.join(errors)}"
                    )

                logger.info("Parallel server initialization complete")

            return True

        except Exception as e:
            logger.error(f"StreamManager initialization failed: {e}", exc_info=True)
            return False

    async def _setup_empty_toolset(self) -> bool:
        """Setup empty toolset when no servers configured."""
        logger.info("Setting up empty toolset")
        self.stream_manager = None
        self._registry = None
        self.processor = None
        return True

    async def close(self) -> None:
        """Close StreamManager and cleanup."""
        if self.stream_manager:
            try:
                await self.stream_manager.close()
            except Exception as e:
                logger.warning(f"Error closing stream_manager: {e}")

    # ================================================================
    # Tool Access (with ToolInfo conversion)
    # ================================================================

    async def get_all_tools(self) -> list[ToolInfo]:
        """Get all tools, converted to mcp-cli ToolInfo model."""
        # Support direct registry access for tests (when stream_manager is None)
        if self._registry is not None and self.stream_manager is None:  # type: ignore[unreachable]
            try:  # type: ignore[unreachable]
                registry_items = await self._registry.list_tools()
                tools = []
                for tool_info in registry_items:
                    ns = tool_info.namespace
                    name = tool_info.name
                    try:
                        metadata = await self._registry.get_metadata(name, ns)
                    except Exception as e:
                        logger.debug(f"Failed to get metadata for {name}: {e}")
                        metadata = None

                    tools.append(
                        ToolInfo(
                            name=name,
                            namespace=ns,
                            description=metadata.description if metadata else "",
                            parameters=metadata.argument_schema if metadata else {},
                            is_async=metadata.is_async if metadata else False,
                            tags=list(metadata.tags) if metadata else [],
                        )
                    )
                return tools
            except Exception as e:
                logger.error(f"Error getting tools from registry: {e}")
                return []

        if not self.stream_manager:
            return []

        try:
            tools_dict = self.stream_manager.get_all_tools()
            # Get tool→server mapping for correct namespace
            tool_to_server = getattr(self.stream_manager, "tool_to_server_map", {})
            tools = []
            for t in tools_dict:
                tool_info = self._convert_to_tool_info(t)
                # Override namespace with server name from tool_to_server_map
                server_name = tool_to_server.get(tool_info.name)
                if server_name:
                    tool_info = ToolInfo(
                        name=tool_info.name,
                        namespace=server_name,
                        description=tool_info.description,
                        parameters=tool_info.parameters,
                        is_async=tool_info.is_async,
                        tags=tool_info.tags,
                        supports_streaming=tool_info.supports_streaming,
                    )
                tools.append(tool_info)
            return tools
        except Exception as e:
            logger.error(f"Error getting tools: {e}")
            return []

    async def get_unique_tools(self) -> list[ToolInfo]:
        """Get tools with duplicates removed (by name)."""
        all_tools = await self.get_all_tools()

        # De-duplicate by name, preferring non-default namespaces
        seen = {}
        for tool in all_tools:
            if tool.name not in seen or tool.namespace != "default":
                seen[tool.name] = tool

        return list(seen.values())

    async def get_tool_by_name(
        self, tool_name: str, namespace: str | None = None
    ) -> ToolInfo | None:
        """Get a tool by name, optionally filtering by namespace."""
        all_tools = await self.get_all_tools()

        if namespace:
            for tool in all_tools:
                if tool.name == tool_name and tool.namespace == namespace:
                    return tool
        else:
            for tool in all_tools:
                if tool.name == tool_name:
                    return tool

        return None

    @staticmethod
    def format_tool_response(response: Any) -> str:
        """Format a tool response for display."""
        import json

        # Handle list of text records (MCP format)
        if isinstance(response, list):
            from mcp_cli.llm.content_models import TextContent

            try:
                text_blocks = [
                    TextContent.model_validate(item)
                    for item in response
                    if isinstance(item, dict)
                    and item.get("type") == ContentBlockType.TEXT.value
                ]
                if text_blocks:
                    return "\n".join(block.text for block in text_blocks)
            except Exception:
                pass

            if all(
                isinstance(item, dict)
                and item.get("type") == ContentBlockType.TEXT.value
                for item in response
            ):
                return "\n".join(item.get("text", "") for item in response)

            return json.dumps(response, indent=2)

        if isinstance(response, dict):
            return json.dumps(response, indent=2)

        return str(response)

    def _convert_to_tool_info(self, tool_dict: dict[str, Any]) -> ToolInfo:
        """Convert chuk tool dict to mcp-cli ToolInfo."""
        tool_input = ToolDefinitionInput.model_validate(tool_dict)

        return ToolInfo(
            name=tool_input.name,
            namespace=tool_input.namespace,
            description=tool_input.description,
            parameters=tool_input.inputSchema,
            is_async=tool_input.is_async,
            tags=tool_input.tags,
        )

    # ================================================================
    # Tool Execution
    # ================================================================

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        namespace: str | None = None,
        timeout: float | None = None,
    ) -> ToolCallResult:
        """Execute tool and return ToolCallResult with automatic recovery on transport errors."""
        # Check if this is a dynamic tool
        if self.dynamic_tool_provider.is_dynamic_tool(tool_name):
            logger.info(f"Executing dynamic tool: {tool_name}")
            try:
                result = await self.dynamic_tool_provider.execute_dynamic_tool(
                    tool_name, arguments
                )
                return ToolCallResult(tool_name=tool_name, success=True, result=result)
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Dynamic tool execution failed: {error_msg}")
                return ToolCallResult(
                    tool_name=tool_name, success=False, error=error_msg
                )

        # Regular MCP tool execution
        if not self.stream_manager:
            return ToolCallResult(
                tool_name=tool_name, success=False, error="ToolManager not initialized"
            )

        try:
            result = await self.stream_manager.call_tool(
                tool_name=tool_name,
                arguments=arguments,
                server_name=namespace,
                timeout=timeout or self.tool_timeout,
            )
            return ToolCallResult(tool_name=tool_name, success=True, result=result)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Tool execution failed: {error_msg}")

            # Check if this is a transport error that might be recoverable
            if (
                "Transport not initialized" in error_msg
                or "transport" in error_msg.lower()
            ):
                logger.warning(
                    f"Transport error detected for tool {tool_name}, attempting recovery..."
                )
                recovery_result = await self._attempt_transport_recovery(
                    tool_name, arguments, namespace, timeout
                )
                if recovery_result:
                    return recovery_result

            return ToolCallResult(tool_name=tool_name, success=False, error=error_msg)

    async def stream_execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        namespace: str | None = None,
        timeout: float | None = None,
    ):
        """Stream tool execution results (for tools that support streaming)."""
        result = await self.execute_tool(tool_name, arguments, namespace, timeout)
        yield result

    async def execute_tools_parallel(
        self,
        calls: list[CTPToolCall],
        timeout: float | None = None,
        on_tool_start: Callable[[CTPToolCall], Awaitable[None]] | None = None,
        on_tool_result: Callable[[CTPToolResult], Awaitable[None]] | None = None,
        max_concurrency: int = 4,
    ) -> list[CTPToolResult]:
        """Execute multiple tool calls in parallel with optional callbacks."""
        return await _execute_tools_parallel(
            self, calls, timeout, on_tool_start, on_tool_result, max_concurrency
        )

    async def stream_execute_tools(
        self,
        calls: list[CTPToolCall],
        timeout: float | None = None,
        on_tool_start: Callable[[CTPToolCall], Awaitable[None]] | None = None,
        max_concurrency: int = 4,
    ) -> AsyncIterator[CTPToolResult]:
        """Execute multiple tool calls in parallel, yielding results as they complete."""
        async for result in _stream_execute_tools(
            self, calls, timeout, on_tool_start, max_concurrency
        ):
            yield result

    async def _attempt_transport_recovery(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        namespace: str | None = None,
        timeout: float | None = None,
    ) -> ToolCallResult | None:
        """Attempt to recover from transport errors by reconnecting to the server."""
        import asyncio

        try:
            server_name = namespace
            if not server_name:
                tools = await self.get_all_tools()
                for tool in tools:
                    if tool.name == tool_name:
                        server_name = tool.server_name
                        break

            if not server_name:
                logger.warning(f"Could not identify server for tool {tool_name}")
                return None

            logger.info(
                f"Attempting to reconnect to server '{server_name}' for tool '{tool_name}'"
            )

            if self.stream_manager is None:
                logger.warning("StreamManager is None, cannot attempt recovery")
                return None

            if hasattr(self.stream_manager, "reconnect_server"):
                await self.stream_manager.reconnect_server(server_name)
            elif hasattr(self.stream_manager, "restart_server"):
                await self.stream_manager.restart_server(server_name)
            else:
                logger.warning(
                    f"StreamManager doesn't support reconnection - "
                    f"server {server_name} may remain in bad state"
                )
                return None

            await asyncio.sleep(0.5)

            logger.info(f"Retrying tool {tool_name} after transport recovery")
            result = await self.stream_manager.call_tool(
                tool_name=tool_name,
                arguments=arguments,
                server_name=namespace,
                timeout=timeout or self.tool_timeout,
            )

            logger.info(f"Tool {tool_name} succeeded after recovery")
            return ToolCallResult(tool_name=tool_name, success=True, result=result)

        except Exception as recovery_error:
            logger.error(f"Transport recovery failed: {recovery_error}")
            return None

    # ================================================================
    # LLM Integration (filtering + adaptation)
    # ================================================================

    async def get_tools_for_llm(self, provider: str = "openai") -> list[dict[str, Any]]:
        """Get tools filtered and validated for LLM."""
        try:
            # Check if dynamic tools mode is enabled
            dynamic_mode = os.environ.get("MCP_CLI_DYNAMIC_TOOLS") == "1"

            if dynamic_mode:
                dynamic_tools = self.dynamic_tool_provider.get_dynamic_tools()
                logger.info(
                    f"Dynamic tools mode: Returning {len(dynamic_tools)} dynamic tools only"
                )
                return dynamic_tools

            # Static mode: load all tools upfront
            all_tools = await self.get_all_tools()

            raw_tools: list[dict[str, Any]] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "No description provided",
                        "parameters": t.parameters
                        or {"type": "object", "properties": {}},
                    },
                }
                for t in all_tools
            ]

            # Apply include/exclude filtering from environment variables
            include_tools = os.environ.get("MCP_CLI_INCLUDE_TOOLS")
            exclude_tools = os.environ.get("MCP_CLI_EXCLUDE_TOOLS")

            if include_tools:
                include_set = {name.strip() for name in include_tools.split(",")}
                raw_tools = [
                    tool
                    for tool in raw_tools
                    if tool["function"]["name"] in include_set
                ]
                logger.info(
                    f"Filtered to {len(raw_tools)} tools using include list: {include_set}"
                )

            if exclude_tools:
                exclude_set = {name.strip() for name in exclude_tools.split(",")}
                raw_tools = [
                    tool
                    for tool in raw_tools
                    if tool["function"]["name"] not in exclude_set
                ]
                logger.info(
                    f"Filtered to {len(raw_tools)} tools using exclude list: {exclude_set}"
                )

            # Filter and validate for provider
            valid_tools, _ = self.tool_filter.filter_tools(raw_tools, provider=provider)

            logger.info(
                f"Returning {len(valid_tools)} tools for LLM after all filtering"
            )
            if len(valid_tools) <= 10:
                for tool in valid_tools:
                    logger.info(f"  - {tool['function']['name']}")

            return valid_tools

        except Exception as e:
            logger.error(f"Error getting LLM tools: {e}")
            return []

    async def get_adapted_tools_for_llm(
        self,
        provider: str = "openai",
        name_mapping: dict[str, str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Get tools adapted for LLM with name mapping."""
        tools = await self.get_tools_for_llm(provider)

        if name_mapping is None:
            mapping = {
                tool["function"]["name"]: tool["function"]["name"] for tool in tools
            }
        else:
            mapping = name_mapping

        return tools, mapping

    # ================================================================
    # Tool Filtering API (delegates to ToolFilter)
    # ================================================================

    def disable_tool(
        self, tool_name: str, reason: DisabledReason = DisabledReason.USER
    ) -> None:
        """Disable a tool."""
        self.tool_filter.disable_tool(tool_name, reason)

    def enable_tool(self, tool_name: str) -> None:
        """Enable a previously disabled tool."""
        self.tool_filter.enable_tool(tool_name)

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled."""
        return self.tool_filter.is_tool_enabled(tool_name)

    def get_disabled_tools(self) -> dict[str, str]:
        """Get all disabled tools with their reasons."""
        return self.tool_filter.get_disabled_tools()

    def set_auto_fix_enabled(self, enabled: bool) -> None:
        """Enable/disable auto-fix for tools."""
        self.tool_filter.set_auto_fix_enabled(enabled)

    def is_auto_fix_enabled(self) -> bool:
        """Check if auto-fix is enabled."""
        return self.tool_filter.is_auto_fix_enabled()

    def clear_validation_disabled_tools(self) -> None:
        """Clear all validation-disabled tools."""
        self.tool_filter.clear_validation_disabled()

    def get_validation_summary(self) -> dict[str, Any]:
        """Get validation summary."""
        return self.tool_filter.get_validation_summary()

    async def validate_single_tool(
        self, tool_name: str, provider: str = "openai"
    ) -> tuple[bool, str | None]:
        """Validate a single tool for LLM compatibility."""
        try:
            tools = await self.get_all_tools()
            tool = next((t for t in tools if t.name == tool_name), None)
            if not tool:
                return False, f"Tool '{tool_name}' not found"

            tool_dict = {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters,
            }

            valid_tools, invalid_tools = self.tool_filter.filter_tools(
                [tool_dict], provider=provider
            )
            if valid_tools:
                return True, None
            elif invalid_tools:
                return False, invalid_tools[0].get("error", "Unknown validation error")
            return False, "Tool validation failed"

        except Exception as e:
            return False, str(e)

    async def revalidate_tools(self, provider: str = "openai") -> dict[str, Any]:
        """Revalidate all tools and return summary."""
        try:
            tools = await self.get_all_tools()
            tool_dicts = [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.parameters,
                }
                for t in tools
            ]

            valid_tools, invalid_tools = self.tool_filter.filter_tools(
                tool_dicts, provider=provider
            )

            return {
                "total": len(tools),
                "valid": len(valid_tools),
                "invalid": len(invalid_tools),
                "invalid_tools": [t.get("name", "unknown") for t in invalid_tools],
            }

        except Exception as e:
            logger.error(f"Error revalidating tools: {e}")
            return {"total": 0, "valid": 0, "invalid": 0, "invalid_tools": []}

    def get_tool_validation_details(self, tool_name: str) -> dict[str, Any] | None:
        """Get validation details for a specific tool."""
        return {"name": tool_name, "status": "unknown"}

    # ================================================================
    # Server Info
    # ================================================================

    async def get_server_info(self) -> list[ServerInfo]:
        """Get information about connected servers."""
        if not self.stream_manager:
            return []

        try:
            servers = []
            server_id = 0

            http_servers = self._config_loader.http_servers
            sse_servers = self._config_loader.sse_servers
            stdio_servers = self._config_loader.stdio_servers
            all_servers = http_servers + sse_servers + stdio_servers

            # Get tool counts per server from StreamManager's tool_to_server_map
            # This is the authoritative source for tool→server mapping
            tool_counts: dict[str, int] = {}
            if hasattr(self.stream_manager, "tool_to_server_map"):
                for server_name in self.stream_manager.tool_to_server_map.values():
                    tool_counts[server_name] = tool_counts.get(server_name, 0) + 1

            for server in all_servers:
                server_name = server.name

                # Determine transport type and get transport-specific fields
                from mcp_cli.config.server_models import (
                    HTTPServerConfig,
                    STDIOServerConfig,
                )

                command: str | None = None
                url: str | None = None
                args: list[str] = []
                env: dict[str, str] = {}

                if server in http_servers:
                    transport = TransportType.HTTP
                    if isinstance(server, HTTPServerConfig):
                        url = server.url
                elif server in sse_servers:
                    transport = TransportType.SSE
                    if isinstance(server, HTTPServerConfig):
                        url = server.url
                else:
                    transport = TransportType.STDIO
                    if isinstance(server, STDIOServerConfig):
                        command = server.command
                        args = list(server.args)
                        env = dict(server.env)

                servers.append(
                    ServerInfo(
                        id=server_id,
                        name=server_name,
                        status=ServerStatus.CONNECTED.value,
                        tool_count=tool_counts.get(server_name, 0),
                        namespace=server_name,
                        enabled=not server.disabled,
                        connected=True,
                        transport=transport,
                        capabilities={},
                        command=command,
                        url=url,
                        args=args,
                        env=env,
                    )
                )
                server_id += 1

            return servers

        except Exception as e:
            logger.error(f"Error getting server info: {e}")
            return []

    async def get_server_for_tool(self, tool_name: str) -> str | None:
        """Get the server name for a given tool."""
        if not self.stream_manager:
            return None

        try:
            tools = await self.get_all_tools()
            tool = next((t for t in tools if t.name == tool_name), None)
            return tool.namespace if tool else None

        except Exception as e:
            logger.error(f"Error getting server for tool: {e}")
            return None

    def get_streams(self):
        """Get active streams from StreamManager."""
        if not self.stream_manager:
            return []

        try:
            if hasattr(self.stream_manager, "get_streams"):
                return self.stream_manager.get_streams()
            return []
        except Exception as e:
            logger.error(f"Error getting streams: {e}")
            return []

    def list_resources(self):
        """List available resources from servers."""
        if not self.stream_manager:
            return []

        try:
            if hasattr(self.stream_manager, "list_resources"):
                return self.stream_manager.list_resources()
            return []
        except Exception as e:
            logger.error(f"Error listing resources: {e}")
            return []

    def list_prompts(self):
        """List available prompts from servers."""
        if not self.stream_manager:
            return []

        try:
            if hasattr(self.stream_manager, "list_prompts"):
                return self.stream_manager.list_prompts()
            return []
        except Exception as e:
            logger.error(f"Error listing prompts: {e}")
            return []


# ====================================================================
# Global instance management
# ====================================================================

_GLOBAL_TOOL_MANAGER: ToolManager | None = None


def get_tool_manager() -> ToolManager | None:
    """Get the global tool manager instance."""
    return _GLOBAL_TOOL_MANAGER


def set_tool_manager(manager: ToolManager) -> None:
    """Set the global tool manager instance."""
    global _GLOBAL_TOOL_MANAGER
    _GLOBAL_TOOL_MANAGER = manager
