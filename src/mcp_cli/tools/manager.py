# mcp_cli/tools/manager.py
"""
Slim ToolManager - orchestrates chuk-tool-processor with mcp-cli features.

Slimmed from 2000+ lines to ~600 lines by:
1. Delegating to StreamManager for all tool operations
2. Keeping only value-add: config parsing, OAuth, filtering, LLM adaptation
3. Removing unused methods and pure pass-through wrappers

For direct StreamManager access: tool_manager.stream_manager.method()
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

from chuk_tool_processor import StreamManager, ToolProcessor

from mcp_cli.auth import TokenManager, TokenStoreBackend, OAuthHandler
from mcp_cli.constants import NAMESPACE
from mcp_cli.tools.filter import ToolFilter
from mcp_cli.tools.models import ServerInfo, ToolCallResult, ToolInfo, TransportType

logger = logging.getLogger(__name__)


class ToolManager:
    """
    Slim facade over chuk-tool-processor with mcp-cli specific features.

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
    ):
        self.config_file = config_file
        self.servers = servers
        self.server_names = server_names or {}
        self.tool_timeout = tool_timeout or 30.0
        self.max_concurrency = max_concurrency
        self.initialization_timeout = initialization_timeout

        # chuk-tool-processor components (publicly accessible)
        self.stream_manager: StreamManager | None = None
        self.processor: ToolProcessor | None = None
        self._registry = None

        # mcp-cli features
        self.tool_filter = ToolFilter()
        self._token_manager: TokenManager
        self._config_cache: dict[str, Any] | None = None

        # Server detection results
        self._http_servers: list[dict[str, Any]] = []
        self._sse_servers: list[dict[str, Any]] = []
        self._stdio_servers: list[dict[str, Any]] = []

        # Setup OAuth
        self._token_manager = TokenManager(
            backend=TokenStoreBackend.AUTO,
            namespace=NAMESPACE,
            service_name="mcp-cli",
        )
        self.oauth_handler = OAuthHandler(token_manager=self._token_manager)

    # ================================================================
    # Initialization
    # ================================================================

    async def initialize(self, namespace: str = "stdio") -> bool:
        """
        Initialize by parsing config, setting up OAuth, and creating StreamManager.
        """
        try:
            from chuk_term.ui import output

            # Load config and detect server types
            config = self._load_config()
            if not config:
                output.warning("No config found, initializing with empty toolset")
                return await self._setup_empty_toolset()

            self._detect_server_types(config)

            # Process OAuth for HTTP/SSE servers before connecting
            if self._http_servers:
                await self._process_oauth_for_servers(self._http_servers)
            if self._sse_servers:
                await self._process_oauth_for_servers(self._sse_servers)

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
        """Initialize StreamManager with detected transport type."""
        self.stream_manager = StreamManager()

        try:
            # Initialize all server types (not mutually exclusive)
            # Changed from elif to if statements to support multiple transport types
            if self._http_servers:
                logger.info(f"Initializing {len(self._http_servers)} HTTP servers")
                await self.stream_manager.initialize_with_http_streamable(
                    servers=self._http_servers,
                    server_names=self.server_names,
                    initialization_timeout=self.initialization_timeout,
                    oauth_refresh_callback=self._create_oauth_refresh_callback(),
                )
            if self._sse_servers:
                logger.info(f"Initializing {len(self._sse_servers)} SSE servers")
                await self.stream_manager.initialize_with_sse(
                    servers=self._sse_servers,
                    server_names=self.server_names,
                    initialization_timeout=self.initialization_timeout,
                    oauth_refresh_callback=self._create_oauth_refresh_callback(),
                )
            if self._stdio_servers:
                logger.info(f"Initializing {len(self._stdio_servers)} STDIO servers")
                await self.stream_manager.initialize_with_stdio(
                    servers=self._stdio_servers,
                    server_names=self.server_names,
                    initialization_timeout=self.initialization_timeout,
                )

            if not (self._http_servers or self._sse_servers or self._stdio_servers):
                logger.info("No servers detected")
                return True

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
    # Config Parsing
    # ================================================================

    def _load_config(self) -> dict[str, Any]:
        """Load and parse MCP config file with token resolution."""
        if self._config_cache:
            return self._config_cache

        try:
            with open(self.config_file) as f:
                config = cast(dict[str, Any], json.load(f))

            # Resolve {{token:provider}} placeholders
            self._resolve_token_placeholders(config)

            self._config_cache = config
            return config

        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_file}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def _resolve_token_placeholders(self, config: dict[str, Any]) -> None:
        """Replace {{token:provider}} with actual OAuth tokens."""

        # Recursive function to process nested dicts
        def process_value(value: Any) -> Any:
            if isinstance(value, str) and value.startswith("{{token:"):
                # Extract provider name
                provider = value[8:-2]  # Remove {{token: and }}
                try:
                    token = self._token_manager.get_token(provider)
                    if token:
                        return f"Bearer {token.access_token}"
                except Exception as e:
                    logger.warning(f"Failed to get token for {provider}: {e}")
            elif isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            return value

        # Process entire config
        if "mcpServers" in config:
            config["mcpServers"] = process_value(config["mcpServers"])

    def _detect_server_types(self, config: dict[str, Any]) -> None:
        """Detect HTTP/SSE/STDIO servers from config."""
        mcp_servers = config.get("mcpServers", {})

        for server_name in self.servers:
            if server_name not in mcp_servers:
                logger.warning(f"Server '{server_name}' not found in config")
                continue

            server_cfg = mcp_servers[server_name]

            # Detect transport type
            if "url" in server_cfg:
                transport = server_cfg.get("transport", "").lower()
                server_config = {
                    "name": server_name,
                    "url": server_cfg["url"],
                    "headers": server_cfg.get("headers", {}),
                }

                if "sse" in transport:
                    self._sse_servers.append(server_config)
                else:
                    self._http_servers.append(server_config)
            else:
                # STDIO server
                self._stdio_servers.append(
                    {
                        "name": server_name,
                        "command": server_cfg.get("command"),
                        "args": server_cfg.get("args", []),
                        "env": server_cfg.get("env", {}),
                    }
                )

    # ================================================================
    # OAuth Integration
    # ================================================================

    async def _process_oauth_for_servers(self, servers: list[dict[str, Any]]) -> None:
        """Process OAuth authentication for servers that require it."""
        from pathlib import Path
        from mcp_cli.config.config_manager import initialize_config

        # Initialize config manager if not already initialized
        config = initialize_config(Path(self.config_file))

        for server_entry in servers:
            server_name = server_entry["name"]
            server_config = config.get_server(server_name)

            if not server_config:
                continue

            # Remote MCP servers (with URL) automatically use MCP OAuth
            # Servers with explicit oauth config use that config
            # Skip servers that already have Authorization headers (they don't need OAuth)
            is_remote_mcp = server_config.url and not server_config.command
            has_explicit_oauth = server_config.oauth is not None
            has_auth_header = (
                server_config.headers and "Authorization" in server_config.headers
            )

            # Skip servers that:
            # 1. Don't need OAuth (not remote MCP and no explicit OAuth config)
            # 2. Already have Authorization headers (pre-configured auth)
            if (not is_remote_mcp and not has_explicit_oauth) or has_auth_header:
                # Skip servers that don't need OAuth
                continue

            try:
                # Perform OAuth and get authorization header
                headers = await self.oauth_handler.prepare_server_headers(server_config)

                # IMPORTANT: HTTP Streamable transport prioritizes configured_headers over api_key
                # To avoid conflicts, we should set EITHER headers OR api_key, not both
                # For OAuth, we use api_key field (which gets converted to Authorization header)
                if "Authorization" in headers:
                    # Extract just the token value (remove "Bearer " prefix if present)
                    # The http_streamable_transport will add "Bearer " prefix automatically
                    auth_value = headers["Authorization"]
                    if auth_value.startswith("Bearer "):
                        auth_value = auth_value[7:]  # Remove "Bearer " prefix
                    server_entry["api_key"] = auth_value
                    logger.debug(f"Set api_key for {server_name}: {auth_value[:20]}...")

                # Merge any other headers (non-Authorization)
                other_headers = {
                    k: v for k, v in headers.items() if k != "Authorization"
                }
                if other_headers:
                    server_entry.setdefault("headers", {}).update(other_headers)

            except Exception as e:
                error_msg = str(e)

                # Check if this is a "server doesn't support OAuth" error (404 on .well-known)
                if "404" in error_msg and ".well-known/oauth-authorization-server" in error_msg:
                    logger.debug(f"Server {server_name} does not support OAuth (no .well-known endpoint), continuing without authentication")
                    # This is fine - not all remote servers require OAuth
                    continue

                # Check for timeout errors (user might need to authenticate)
                if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    logger.warning(f"OAuth setup timed out for {server_name}. This might indicate:")
                    logger.warning("  - Network connectivity issues")
                    logger.warning("  - OAuth server being slow")
                    logger.warning("  - Need to complete authentication in browser")
                    # Don't fail - let the connection attempt continue
                    continue

                # For other errors, log but continue (don't block initialization)
                logger.warning(f"OAuth setup failed for {server_name}: {e}")
                logger.debug("Continuing without OAuth for this server")
                # Don't raise - allow connection without OAuth
                continue

    def _create_oauth_refresh_callback(self):
        """Create OAuth token refresh callback for StreamManager."""

        async def refresh_oauth_token():
            # Simplified - can be expanded with actual refresh logic
            logger.debug("OAuth refresh callback triggered")

        return refresh_oauth_token

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

                    # Create ToolInfo even if metadata is missing
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
            # Get tools from StreamManager
            tools_dict = self.stream_manager.get_all_tools()

            # Convert to ToolInfo
            return [self._convert_to_tool_info(t) for t in tools_dict]
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
            # Filter by namespace first
            for tool in all_tools:
                if tool.name == tool_name and tool.namespace == namespace:
                    return tool
        else:
            # Return first match
            for tool in all_tools:
                if tool.name == tool_name:
                    return tool

        return None

    @staticmethod
    def format_tool_response(response: Any) -> str:
        """
        Format a tool response for display.

        Handles MCP text records, JSON data, dicts, and scalars.
        """
        import json

        # Handle list of text records (MCP format)
        if isinstance(response, list):
            # Check if it's a list of text records
            if all(
                isinstance(item, dict) and item.get("type") == "text"
                for item in response
            ):
                return "\n".join(item.get("text", "") for item in response)

            # Otherwise serialize as JSON
            return json.dumps(response, indent=2)

        # Handle dict
        if isinstance(response, dict):
            return json.dumps(response, indent=2)

        # Handle scalar values
        return str(response)

    def _convert_to_tool_info(self, tool_dict: dict[str, Any]) -> ToolInfo:
        """Convert chuk tool dict to mcp-cli ToolInfo."""
        return ToolInfo(
            name=tool_dict.get("name", ""),
            namespace=tool_dict.get("namespace", "default"),
            description=tool_dict.get("description"),
            parameters=tool_dict.get("inputSchema", {}),
            is_async=tool_dict.get("is_async", False),
            tags=tool_dict.get("tags", []),
        )

    # ================================================================
    # Tool Execution (wraps StreamManager)
    # ================================================================

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        namespace: str | None = None,
        timeout: float | None = None,
    ) -> ToolCallResult:
        """Execute tool and return ToolCallResult with automatic recovery on transport errors."""
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
            if "Transport not initialized" in error_msg or "transport" in error_msg.lower():
                logger.warning(f"Transport error detected for tool {tool_name}, attempting recovery...")

                # Attempt to recover by reconnecting to the affected server
                recovery_result = await self._attempt_transport_recovery(
                    tool_name, arguments, namespace, timeout
                )
                if recovery_result:
                    return recovery_result

            return ToolCallResult(tool_name=tool_name, success=False, error=error_msg)

    # ================================================================
    # LLM Integration (filtering + adaptation)
    # ================================================================

    async def get_tools_for_llm(self, provider: str = "openai") -> list[dict[str, Any]]:
        """Get tools filtered and validated for LLM."""
        try:
            # Get all tools first (handles both stream_manager and registry paths)
            all_tools = await self.get_all_tools()

            # Convert ToolInfo to LLM format for filter
            raw_tools = [
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

            # Filter and validate for provider
            valid_tools, _ = self.tool_filter.filter_tools(raw_tools, provider=provider)

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

        # Create identity mapping if not provided
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

    def disable_tool(self, tool_name: str, reason: str = "user") -> None:
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

            # Convert to dict for validation
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

    async def _attempt_transport_recovery(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        namespace: str | None = None,
        timeout: float | None = None,
    ) -> ToolCallResult | None:
        """
        Attempt to recover from transport errors by reconnecting to the server.

        This handles cases where the MCP server transport gets into a bad state
        after timeouts or concurrent requests.

        Returns:
            ToolCallResult if recovery succeeded and tool was executed, None otherwise
        """
        try:
            # First, try to identify which server this tool belongs to
            server_name = namespace
            if not server_name:
                # Try to find the server by looking at available tools
                tools = await self.get_all_tools()
                for tool in tools:
                    if tool.name == tool_name:
                        server_name = tool.server_name
                        break

            if not server_name:
                logger.warning(f"Could not identify server for tool {tool_name}")
                return None

            logger.info(f"Attempting to reconnect to server '{server_name}' for tool '{tool_name}'")

            # Try to reconnect the specific server through StreamManager
            if hasattr(self.stream_manager, 'reconnect_server'):
                await self.stream_manager.reconnect_server(server_name)
            elif hasattr(self.stream_manager, 'restart_server'):
                await self.stream_manager.restart_server(server_name)
            else:
                # If no specific reconnect method, log warning
                logger.warning(
                    f"StreamManager doesn't support reconnection - server {server_name} may remain in bad state"
                )
                return None

            # Wait a moment for reconnection
            import asyncio
            await asyncio.sleep(0.5)

            # Retry the tool call once
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
        # For now, return basic info - can be expanded if validation cache is needed
        return {"name": tool_name, "status": "unknown"}

    async def get_server_info(self) -> list[ServerInfo]:
        """Get information about connected servers."""
        if not self.stream_manager:
            return []

        try:
            # Construct ServerInfo from detected servers
            servers = []
            server_id = 0

            all_servers = self._http_servers + self._sse_servers + self._stdio_servers

            # Get tool counts per server if available
            tools = await self.get_all_tools()
            tool_counts: dict[str, int] = {}
            for tool in tools:
                namespace = tool.namespace or "default"
                tool_counts[namespace] = tool_counts.get(namespace, 0) + 1

            for server in all_servers:
                server_name = server.get("name", "unknown")

                # Determine transport type
                if server in self._http_servers:
                    transport = TransportType.HTTP
                elif server in self._sse_servers:
                    transport = TransportType.SSE
                else:
                    transport = TransportType.STDIO

                servers.append(
                    ServerInfo(
                        id=server_id,
                        name=server_name,
                        status="connected",
                        tool_count=tool_counts.get(server_name, 0),
                        namespace=server_name,
                        enabled=True,
                        connected=True,
                        transport=transport,
                        capabilities={},
                        command=server.get("command"),
                        args=server.get("args", []),
                        env=server.get("env", {}),
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

    async def stream_execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        namespace: str | None = None,
        timeout: float | None = None,
    ):
        """Stream tool execution results (for tools that support streaming)."""
        # For now, fall back to regular execution and yield once
        # Can be enhanced when StreamManager supports streaming
        result = await self.execute_tool(tool_name, arguments, namespace, timeout)
        yield result

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
