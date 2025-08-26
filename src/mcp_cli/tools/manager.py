# mcp_cli/tools/manager.py
"""
Centralized tool management using CHUK Tool Processor.

This module provides a unified interface for all tool-related operations in MCP CLI,
leveraging the async-native capabilities of CHUK Tool Processor.

Supports STDIO, HTTP, and SSE servers with automatic detection.
ENHANCED: Now includes schema validation and tool filtering capabilities.
CLEAN: Tool name resolution using registry lookup without sanitization.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncIterator
from pathlib import Path

from chuk_tool_processor.core.processor import ToolProcessor
from chuk_tool_processor.registry import ToolRegistryProvider
from chuk_tool_processor.mcp.stream_manager import StreamManager
from chuk_tool_processor.models.tool_result import ToolResult
from chuk_tool_processor.models.tool_call import ToolCall
from chuk_tool_processor.execution.strategies.inprocess_strategy import (
    InProcessStrategy,
)
from chuk_tool_processor.execution.tool_executor import ToolExecutor

from mcp_cli.tools.models import ServerInfo, ToolCallResult, ToolInfo
from mcp_cli.tools.validation import ToolSchemaValidator
from mcp_cli.tools.filter import ToolFilter

logger = logging.getLogger(__name__)


class ToolManager:
    """
    Central interface for all tool operations in MCP CLI.

    This class wraps CHUK Tool Processor and provides a clean API for:
    - Tool discovery and listing
    - Tool execution with streaming support
    - Server management (STDIO, HTTP, and SSE)
    - LLM-compatible tool conversion
    - Schema validation and tool filtering (ENHANCED)
    """

    def __init__(
        self,
        config_file: str,
        servers: List[str],
        server_names: Optional[Dict[int, str]] = None,
        tool_timeout: Optional[float] = None,
        max_concurrency: int = 4,
        initialization_timeout: float = 120.0,
        server_connection_timeout: float = 30.0,
    ):
        self.config_file = config_file
        self.servers = servers
        self.server_names = server_names or {}
        self.tool_timeout = self._determine_timeout(tool_timeout)
        self.max_concurrency = max_concurrency
        self.initialization_timeout = initialization_timeout
        self.server_connection_timeout = server_connection_timeout

        # CHUK components
        self.processor: Optional[ToolProcessor] = None
        self.stream_manager: Optional[StreamManager] = None
        self._registry = None
        self._executor: Optional[ToolExecutor] = None

        # Server type detection
        self._http_servers = []
        self._stdio_servers = []
        self._sse_servers = []
        self._config_cache = None

        # ENHANCED: Tool validation and filtering
        self.tool_filter = ToolFilter()
        self.validation_results: Dict[str, Dict[str, Any]] = {}
        self.last_validation_provider: Optional[str] = None

    def _determine_timeout(self, explicit_timeout: Optional[float]) -> float:
        """Determine timeout with environment variable fallback."""
        if explicit_timeout is not None:
            return explicit_timeout

        # Check environment variables in order of preference
        for env_var in [
            "MCP_TOOL_TIMEOUT",
            "CHUK_TOOL_TIMEOUT",
            "MCP_CLI_INIT_TIMEOUT",
        ]:
            env_timeout = os.getenv(env_var)
            if env_timeout:
                try:
                    return float(env_timeout)
                except ValueError:
                    logger.warning(f"Invalid timeout in {env_var}: {env_timeout}")

        return 120.0  # Default 2 minutes

    def _load_config(self) -> Dict[str, Any]:
        """Load and cache the configuration file."""
        if self._config_cache is not None:
            return self._config_cache

        try:
            config_path = Path(self.config_file)
            if config_path.exists():
                with open(config_path, "r") as f:
                    self._config_cache = json.load(f)
                    return self._config_cache
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")

        self._config_cache = {}
        return self._config_cache

    def _detect_server_types(self):
        """Detect server transport types from configuration."""
        config = self._load_config()
        mcp_servers = config.get("mcpServers", {})

        if not mcp_servers:
            # No config, assume all are STDIO
            self._stdio_servers = self.servers.copy()
            logger.debug("No config found, assuming all servers are STDIO")
            return

        for server in self.servers:
            server_config = mcp_servers.get(server, {})
            transport_type = server_config.get("transport", "").lower()

            if transport_type == "sse":
                self._sse_servers.append(
                    {
                        "name": server,
                        "url": server_config["url"],
                        "headers": server_config.get("headers", {}),
                    }
                )
                logger.debug(f"Detected SSE server: {server}")
            elif "url" in server_config:
                self._http_servers.append(
                    {
                        "name": server,
                        "url": server_config["url"],
                        "headers": server_config.get("headers", {}),
                    }
                )
                logger.debug(f"Detected HTTP server: {server}")
            else:
                self._stdio_servers.append(server)
                logger.debug(f"Detected STDIO server: {server}")

        logger.info(
            f"Detected {len(self._http_servers)} HTTP, {len(self._sse_servers)} SSE, {len(self._stdio_servers)} STDIO servers"
        )

    async def initialize(self, namespace: str = "stdio") -> bool:
        """Connect to MCP servers and initialize the tool registry."""
        try:
            logger.info(f"Initializing ToolManager with {len(self.servers)} servers")

            self._detect_server_types()

            # Try transports in priority order: SSE > HTTP > STDIO
            success = False

            if self._sse_servers:
                logger.info("Setting up SSE servers")
                success = await self._setup_sse_servers(self._sse_servers[0]["name"])
            elif self._http_servers:
                logger.info("Setting up HTTP servers")
                success = await self._setup_http_servers(self._http_servers[0]["name"])
            elif self._stdio_servers:
                logger.info("Setting up STDIO servers")
                success = await self._setup_stdio_servers(namespace)
            else:
                logger.info("No servers configured - initializing with empty tool list")
                success = await self._setup_empty_toolset()

            if not success:
                logger.error("Server setup failed")
                return False

            await self._setup_common_components()
            logger.info("ToolManager initialized successfully")
            return True

        except asyncio.TimeoutError:
            logger.error(
                f"Initialization timed out after {self.initialization_timeout}s"
            )
            return False
        except Exception as exc:
            logger.error(f"Error initializing tool manager: {exc}")
            return False

    async def _setup_sse_servers(self, namespace: str) -> bool:
        """Setup SSE servers."""
        try:
            from chuk_tool_processor.mcp.setup_mcp_sse import setup_mcp_sse

            self.processor, self.stream_manager = await asyncio.wait_for(
                setup_mcp_sse(
                    servers=self._sse_servers,
                    server_names=self.server_names,
                    namespace=namespace,
                    connection_timeout=self.server_connection_timeout,
                    default_timeout=self.tool_timeout,
                ),
                timeout=self.initialization_timeout,
            )
            return True

        except ImportError as e:
            logger.error(f"SSE transport not available: {e}")
            return False
        except Exception as e:
            logger.error(f"SSE server setup failed: {e}")
            return False

    async def _setup_http_servers(self, namespace: str) -> bool:
        """Setup HTTP servers."""
        try:
            from chuk_tool_processor.mcp.setup_mcp_http_streamable import (
                setup_mcp_http_streamable,
            )

            self.processor, self.stream_manager = await asyncio.wait_for(
                setup_mcp_http_streamable(
                    servers=self._http_servers,
                    server_names=self.server_names,
                    namespace=namespace,
                    connection_timeout=self.server_connection_timeout,
                    default_timeout=self.tool_timeout,
                ),
                timeout=self.initialization_timeout,
            )
            return True

        except ImportError as e:
            logger.error(f"HTTP transport not available: {e}")
            return False
        except Exception as e:
            logger.error(f"HTTP server setup failed: {e}")
            return False

    async def _setup_stdio_servers(self, namespace: str) -> bool:
        """Setup STDIO servers."""
        try:
            from chuk_tool_processor.mcp.setup_mcp_stdio import setup_mcp_stdio

            self.processor, self.stream_manager = await asyncio.wait_for(
                setup_mcp_stdio(
                    config_file=str(self.config_file),
                    servers=self._stdio_servers,
                    server_names=self.server_names,
                    namespace=namespace,
                    enable_caching=True,
                    enable_retries=True,
                    max_retries=2,
                ),
                timeout=self.initialization_timeout,
            )
            return True

        except Exception as e:
            logger.error(f"STDIO server setup failed: {e}")
            return False

    async def _setup_empty_toolset(self) -> bool:
        """Setup an empty tool processor when no servers are configured."""
        try:
            # Create a minimal mock processor that implements the required interface
            class EmptyToolProcessor:
                def __init__(self):
                    self.tools = {}

                async def execute_tool(self, *args, **kwargs):
                    return {"error": "No tools available"}

                def list_tools(self):
                    return []

                def get_tool(self, name):
                    return None

            class EmptyStreamManager:
                def __init__(self):
                    pass

                async def stream(self, *args, **kwargs):
                    yield {"error": "No streaming available"}

                async def close(self):
                    """No-op close method for compatibility."""
                    pass

            # Create minimal processor and stream manager with no tools
            self.processor = EmptyToolProcessor()
            self.stream_manager = EmptyStreamManager()

            logger.info(
                "Initialized with empty tool set - chat mode available without tools"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to setup empty toolset: {e}")
            return False

    async def _setup_common_components(self):
        """Setup components common to all transport types."""
        self._registry = await asyncio.wait_for(
            ToolRegistryProvider.get_registry(), timeout=30.0
        )

        strategy = InProcessStrategy(
            self._registry,
            max_concurrency=self.max_concurrency,
            default_timeout=self.tool_timeout,
        )

        self._executor = ToolExecutor(
            registry=self._registry,
            strategy=strategy,
            default_timeout=self.tool_timeout,
        )

    async def close(self):
        """Close all resources and connections."""
        try:
            if self.stream_manager:
                await self.stream_manager.close()
            if self._executor:
                await self._executor.shutdown()
        except Exception as exc:
            logger.warning(f"Error during shutdown: {exc}")

    # Tool discovery
    async def get_all_tools(self) -> List[ToolInfo]:
        """Return all available tools."""
        if not self._registry:
            return []

        tools = []
        try:
            registry_items = await asyncio.wait_for(
                self._registry.list_tools(), timeout=30.0
            )

            for ns, name in registry_items:
                try:
                    metadata = await asyncio.wait_for(
                        self._registry.get_metadata(name, ns), timeout=5.0
                    )

                    tools.append(
                        ToolInfo(
                            name=name,
                            namespace=ns,
                            description=metadata.description if metadata else "",
                            parameters=metadata.argument_schema if metadata else {},
                            is_async=metadata.is_async if metadata else False,
                            tags=list(metadata.tags) if metadata else [],
                            supports_streaming=getattr(
                                metadata, "supports_streaming", False
                            )
                            if metadata
                            else False,
                        )
                    )

                except asyncio.TimeoutError:
                    logger.warning(f"Timeout getting metadata for {ns}.{name}")
                except Exception as e:
                    logger.warning(f"Error getting metadata for {ns}.{name}: {e}")

        except Exception as exc:
            logger.error(f"Error discovering tools: {exc}")

        return tools

    async def get_unique_tools(self) -> List[ToolInfo]:
        """Return tools without duplicates from the default namespace."""
        seen = set()
        unique = []

        for tool in await self.get_all_tools():
            if tool.namespace == "default" or tool.name in seen:
                continue
            seen.add(tool.name)
            unique.append(tool)

        return unique

    async def get_tool_by_name(
        self, tool_name: str, namespace: str = None
    ) -> Optional[ToolInfo]:
        """Get tool info by name and optional namespace."""
        if not self._registry:
            return None

        if namespace:
            try:
                metadata = await asyncio.wait_for(
                    self._registry.get_metadata(tool_name, namespace), timeout=5.0
                )
                if metadata:
                    return ToolInfo(
                        name=tool_name,
                        namespace=namespace,
                        description=metadata.description,
                        parameters=metadata.argument_schema,
                        is_async=metadata.is_async,
                        tags=list(metadata.tags),
                        supports_streaming=getattr(
                            metadata, "supports_streaming", False
                        ),
                    )
            except Exception:
                pass

        # Search all non-default namespaces
        for tool in await self.get_all_tools():
            if tool.name == tool_name and tool.namespace != "default":
                return tool

        return None

    # Tool execution
    async def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any], timeout: Optional[float] = None
    ) -> ToolCallResult:
        """Execute a tool and return the result."""
        if not isinstance(arguments, dict):
            return ToolCallResult(tool_name, False, error="Arguments must be a dict")

        # Check if tool is enabled
        if not self.tool_filter.is_tool_enabled(tool_name):
            disabled_reason = self.tool_filter.get_disabled_tools().get(
                tool_name, "unknown"
            )
            return ToolCallResult(
                tool_name, False, error=f"Tool disabled ({disabled_reason})"
            )

        # CLEAN: Just look up the tool directly in the registry
        namespace, base_name = await self._find_tool_in_registry(tool_name)

        if not namespace:
            return ToolCallResult(
                tool_name, False, error=f"Tool '{tool_name}' not found in registry"
            )

        logger.info(
            f"EXECUTION: Found tool '{tool_name}' -> namespace='{namespace}', base_name='{base_name}'"
        )
        logger.info(f"EXECUTION: Arguments = {arguments}")
        logger.info(
            f"EXECUTION: Creating ToolCall with tool='{base_name}', namespace='{namespace}'"
        )
        logger.info(f"EXECUTION: Tool timeout = {timeout or self.tool_timeout}")

        call = ToolCall(
            tool=base_name,
            namespace=namespace,
            arguments=arguments,
            timeout=timeout or self.tool_timeout,
        )

        logger.info(
            f"EXECUTION: ToolCall created: tool='{call.tool}', namespace='{call.namespace}'"
        )
        logger.info(f"EXECUTION: ToolCall arguments: {call.arguments}")

        try:
            import time

            logger.info("EXECUTION: Calling executor.execute() with call")
            start_time = time.time()
            results = await self._executor.execute([call])
            elapsed = time.time() - start_time
            logger.info(
                f"EXECUTION: Executor completed in {elapsed:.2f}s, returned {len(results) if results else 0} results"
            )

            if not results:
                logger.error("EXECUTION: No results returned from executor")
                return ToolCallResult(tool_name, False, error="No result returned")

            result = results[0]
            logger.info(
                f"EXECUTION: First result: success={not bool(result.error)}, error='{result.error}', result_type={type(result.result)}"
            )

            if result.result:
                result_str = str(result.result)[:200]
                logger.info(f"EXECUTION: Result content: {result_str}...")

            return ToolCallResult(
                tool_name=tool_name,
                success=not bool(result.error),
                result=result.result,
                error=result.error,
                execution_time=(
                    (result.end_time - result.start_time).total_seconds()
                    if hasattr(result, "end_time") and hasattr(result, "start_time")
                    else None
                ),
            )
        except Exception as exc:
            logger.error(f"EXECUTION: Exception during execution: {exc}")
            import traceback

            traceback.print_exc()
            return ToolCallResult(tool_name, False, error=str(exc))

    async def _find_tool_in_registry(self, tool_name: str) -> Tuple[str, str]:
        """
        Find a tool in the registry exactly as it exists.
        Resolve namespace for the tool name.

        Returns:
            Tuple of (namespace, tool_name) or ("", "") if not found
        """
        if not self._registry:
            logger.debug("No registry available")
            return "", ""

        try:
            # Get all available tools from registry
            registry_items = await asyncio.wait_for(
                self._registry.list_tools(), timeout=10.0
            )

            logger.info(
                f"Looking for tool '{tool_name}' in {len(registry_items)} registry entries"
            )

            # Find the tool and its namespace
            for namespace, name in registry_items:
                if name == tool_name:
                    logger.info(f"Found tool '{tool_name}' in namespace '{namespace}'")
                    return namespace, name

            logger.error(f"Tool '{tool_name}' not found in any namespace")
            logger.debug("Available tools:")
            for namespace, name in registry_items[:10]:  # Show first 10 for debugging
                logger.debug(f"  {namespace}/{name}")

            return "", ""

        except Exception as e:
            logger.error(f"Error looking up tool '{tool_name}' in registry: {e}")
            return "", ""

    async def stream_execute_tool(
        self, tool_name: str, arguments: Dict[str, Any], timeout: Optional[float] = None
    ) -> AsyncIterator[ToolResult]:
        """Execute a tool with streaming support."""
        # Check if tool is enabled
        if not self.tool_filter.is_tool_enabled(tool_name):
            from chuk_tool_processor.models.tool_result import ToolResult
            from chuk_tool_processor.models.tool_call import ToolCall

            disabled_reason = self.tool_filter.get_disabled_tools().get(
                tool_name, "unknown"
            )
            dummy_call = ToolCall(tool=tool_name, namespace="", arguments={})
            error_result = ToolResult(
                tool_call=dummy_call,
                result=None,
                error=f"Tool disabled ({disabled_reason})",
            )
            yield error_result
            return

        # CLEAN: Same direct lookup
        namespace, base_name = await self._find_tool_in_registry(tool_name)

        if not namespace:
            dummy_call = ToolCall(tool=tool_name, namespace="", arguments={})
            error_result = ToolResult(
                tool_call=dummy_call,
                result=None,
                error=f"Tool '{tool_name}' not found in registry",
            )
            yield error_result
            return

        call = ToolCall(
            tool=base_name,
            namespace=namespace,
            arguments=arguments,
            timeout=timeout or self.tool_timeout,
        )

        async for result in self._executor.stream_execute([call]):
            yield result

    async def process_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        name_mapping: Dict[str, str],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ToolResult]:
        """Process tool calls from an LLM."""
        chuk_calls = []
        call_mapping = {}

        for tc in tool_calls:
            if not (tc.get("function") and "name" in tc.get("function", {})):
                continue

            # LLM tool name (possibly sanitized by LLM provider)
            llm_tool_name = tc["function"]["name"]
            tool_call_id = (
                tc.get("id") or f"call_{llm_tool_name}_{uuid.uuid4().hex[:8]}"
            )

            # Get the original tool name from the mapping provided by LLM provider
            # This mapping should handle all sanitization/conversion
            original_tool_name = name_mapping.get(llm_tool_name, llm_tool_name)

            logger.debug(
                f"Tool call: LLM name='{llm_tool_name}' -> Original name='{original_tool_name}'"
            )

            # Check if tool is enabled
            if not self.tool_filter.is_tool_enabled(original_tool_name):
                logger.warning(f"Skipping disabled tool: {original_tool_name}")
                continue

            # Parse arguments
            args_str = tc["function"].get("arguments", "{}")
            try:
                args_dict = (
                    json.loads(args_str)
                    if isinstance(args_str, str) and args_str.strip()
                    else args_str or {}
                )
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool arguments for {llm_tool_name}: {e}")
                args_dict = {}

            # CLEAN: Direct registry lookup using the original name
            namespace, base_name = await self._find_tool_in_registry(original_tool_name)

            # Skip if we can't find the tool
            if not namespace:
                logger.error(f"Tool not found in registry: {original_tool_name}")
                continue

            call = ToolCall(
                tool=base_name,
                namespace=namespace,
                arguments=args_dict,
                metadata={"call_id": tool_call_id, "original_name": original_tool_name},
            )

            chuk_calls.append(call)
            call_mapping[id(call)] = {
                "id": tool_call_id,
                "name": llm_tool_name,
            }  # Use LLM name for conversation

            # Add to conversation history using LLM tool name (for consistency with LLM)
            if conversation_history is not None:
                conversation_history.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": llm_tool_name,  # Use LLM name in conversation
                                    "arguments": json.dumps(args_dict),
                                },
                            }
                        ],
                    }
                )

        # Execute tool calls
        results = await self._executor.execute(chuk_calls)

        # Process results
        for result in results:
            call_info = call_mapping.get(
                id(result.tool_call),
                {"id": f"call_{uuid.uuid4().hex[:8]}", "name": result.tool},
            )

            if conversation_history is not None:
                content = (
                    f"Error: {result.error}"
                    if result.error
                    else self.format_tool_response(result.result)
                )
                conversation_history.append(
                    {
                        "role": "tool",
                        "name": call_info["name"],  # Use LLM name for consistency
                        "content": content,
                        "tool_call_id": call_info["id"],
                    }
                )

        return results

    # Server helpers
    async def get_server_info(self) -> List[ServerInfo]:
        """Get information about all connected servers."""
        if not self.stream_manager:
            return []

        try:
            if hasattr(self.stream_manager, "get_server_info"):
                server_info_result = self.stream_manager.get_server_info()

                if hasattr(server_info_result, "__await__"):
                    raw_infos = await asyncio.wait_for(server_info_result, timeout=10.0)
                else:
                    raw_infos = server_info_result

                return [
                    ServerInfo(
                        id=raw.get("id", 0),
                        name=raw.get("name", "Unknown"),
                        status=raw.get("status", "Unknown"),
                        tool_count=raw.get("tools", 0),
                        namespace=raw.get("name", "").split("_")[0]
                        if "_" in raw.get("name", "")
                        else raw.get("name", ""),
                    )
                    for raw in raw_infos
                ]

        except Exception as exc:
            logger.error(f"Error getting server info: {exc}")

        # Fallback to basic info
        return [
            ServerInfo(
                id=i,
                name=server,
                status="Unknown",
                tool_count=0,
                namespace=server.split("_")[0] if "_" in server else server,
            )
            for i, server in enumerate(self.servers)
        ]

    async def get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Get the server name for a tool."""
        # CLEAN: Just get namespace from registry lookup
        namespace, _ = await self._find_tool_in_registry(tool_name)
        return namespace if namespace else None

    # LLM helpers
    async def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool definitions (validated)."""
        valid_tools, _ = await self.get_adapted_tools_for_llm("openai")
        return valid_tools

    async def get_adapted_tools_for_llm(
        self, provider: str = "openai"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Get tools adapted for the specified LLM provider with validation."""
        # Get raw tools first
        raw_tools, raw_name_mapping = await self._get_raw_adapted_tools_for_llm(
            provider
        )

        # Apply validation and filtering
        valid_tools, invalid_tools = self.tool_filter.filter_tools(raw_tools, provider)

        # Update name mapping to only include valid tools
        valid_tool_names = {
            self.tool_filter._extract_tool_name(tool) for tool in valid_tools
        }
        filtered_name_mapping = {
            k: v
            for k, v in raw_name_mapping.items()
            if any(
                v.endswith(name.split(".")[-1]) or v == name
                for name in valid_tool_names
            )
        }

        # Store validation results
        self.validation_results = {
            "provider": provider,
            "total_tools": len(raw_tools),
            "valid_tools": len(valid_tools),
            "invalid_tools": len(invalid_tools),
            "disabled_tools": self.tool_filter.get_disabled_tools(),
            "validation_errors": [
                {
                    "tool": self.tool_filter._extract_tool_name(tool),
                    "error": tool.get("_validation_error"),
                    "reason": tool.get("_disabled_reason"),
                }
                for tool in invalid_tools
                if tool.get("_validation_error")
            ],
        }
        self.last_validation_provider = provider

        if invalid_tools:
            logger.warning(
                f"Tool validation for {provider}: {len(valid_tools)} valid, {len(invalid_tools)} invalid"
            )
            # Log specific errors for debugging
            for error in self.validation_results["validation_errors"]:
                logger.debug(
                    f"Tool '{error['tool']}' validation error: {error['error']}"
                )
        else:
            logger.info(
                f"Tool validation for {provider}: {len(valid_tools)} valid, 0 invalid"
            )

        return valid_tools, filtered_name_mapping

    async def _get_raw_adapted_tools_for_llm(
        self, provider: str = "openai"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Get raw tools adapted for the specified LLM provider (without validation)."""
        unique_tools = await self.get_unique_tools()

        llm_tools = []
        name_mapping = {}

        logger.info(f"Creating LLM tools for provider '{provider}'")
        logger.info(f"Processing {len(unique_tools)} unique tools:")

        for tool in unique_tools:
            # NO MANIPULATION - use tool name exactly as it exists in registry
            tool_name = tool.name
            logger.info(f"  Tool: namespace='{tool.namespace}', name='{tool.name}'")

            # NO SANITIZATION - pass through tool name as-is
            # The LLM provider handles any necessary name conversion
            name_mapping[tool_name] = (
                tool.name
            )  # Identity mapping - no conversion needed

            llm_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool.description or "",
                        "parameters": tool.parameters or {},
                    },
                }
            )

        logger.info(f"Final name mapping (identity): {name_mapping}")
        return llm_tools, name_mapping

    # ENHANCED: Tool management methods
    def disable_tool(self, tool_name: str, reason: str = "user") -> None:
        """Disable a specific tool."""
        self.tool_filter.disable_tool(tool_name, reason)

    def enable_tool(self, tool_name: str) -> None:
        """Re-enable a specific tool."""
        self.tool_filter.enable_tool(tool_name)

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled."""
        return self.tool_filter.is_tool_enabled(tool_name)

    def get_disabled_tools(self) -> Dict[str, str]:
        """Get all disabled tools with their reasons."""
        return self.tool_filter.get_disabled_tools()

    def set_auto_fix_enabled(self, enabled: bool) -> None:
        """Enable or disable automatic fixing of tool schemas."""
        self.tool_filter.auto_fix_enabled = enabled
        logger.info(f"Auto-fix {'enabled' if enabled else 'disabled'}")

    def is_auto_fix_enabled(self) -> bool:
        """Check if auto-fix is enabled."""
        return self.tool_filter.auto_fix_enabled

    def clear_validation_disabled_tools(self) -> None:
        """Clear all tools disabled due to validation errors."""
        self.tool_filter.clear_validation_disabled()

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of the last validation run."""
        summary = self.tool_filter.get_validation_summary()
        summary.update(self.validation_results)
        return summary

    async def revalidate_tools(self, provider: str = None) -> Dict[str, Any]:
        """
        Re-run validation on all tools.

        Args:
            provider: Provider to validate for (defaults to last used)

        Returns:
            Validation summary
        """
        target_provider = provider or self.last_validation_provider or "openai"

        # Clear validation-disabled tools
        self.clear_validation_disabled_tools()

        # Re-run validation
        _, _ = await self.get_adapted_tools_for_llm(target_provider)

        return self.get_validation_summary()

    async def validate_single_tool(
        self, tool_name: str, provider: str = "openai"
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a single tool by name.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Get the tool definition
        all_tools = await self.get_unique_tools()
        target_tool = None

        for tool in all_tools:
            if tool.name == tool_name or f"{tool.namespace}.{tool.name}" == tool_name:
                target_tool = tool
                break

        if not target_tool:
            return False, f"Tool '{tool_name}' not found"

        # Convert to LLM format
        llm_tools, _ = await self._get_raw_adapted_tools_for_llm(provider)

        # Find the corresponding LLM tool
        llm_tool = None
        for tool_def in llm_tools:
            if self.tool_filter._extract_tool_name(tool_def) == tool_name:
                llm_tool = tool_def
                break

        if not llm_tool:
            return False, f"Tool '{tool_name}' not found in LLM format"

        # Validate
        if provider == "openai":
            return ToolSchemaValidator.validate_openai_schema(llm_tool)
        else:
            return True, None  # Assume valid for other providers

    def get_tool_validation_details(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed validation information for a specific tool."""
        disabled_tools = self.get_disabled_tools()

        if tool_name in disabled_tools:
            reason = disabled_tools[tool_name]

            # Find validation error in results
            validation_error = None
            for error in self.validation_results.get("validation_errors", []):
                if error["tool"] == tool_name:
                    validation_error = error["error"]
                    break

            return {
                "tool_name": tool_name,
                "is_enabled": False,
                "disabled_reason": reason,
                "validation_error": validation_error,
                "can_auto_fix": self.tool_filter.auto_fix_enabled
                and reason == "validation",
            }
        else:
            return {
                "tool_name": tool_name,
                "is_enabled": True,
                "disabled_reason": None,
                "validation_error": None,
                "can_auto_fix": False,
            }

    # Formatting helpers
    @staticmethod
    def format_tool_response(response_content: Union[List[Dict[str, Any]], Any]) -> str:
        """Format tool response content for LLM consumption."""
        if (
            isinstance(response_content, list)
            and response_content
            and isinstance(response_content[0], dict)
        ):
            if all(
                isinstance(item, dict) and item.get("type") == "text"
                for item in response_content
            ):
                return "\n".join(item.get("text", "") for item in response_content)
            try:
                return json.dumps(response_content, indent=2)
            except Exception:
                return str(response_content)
        elif isinstance(response_content, dict):
            try:
                return json.dumps(response_content, indent=2)
            except Exception:
                return str(response_content)
        else:
            return str(response_content)

    # Configuration
    def set_tool_timeout(self, timeout: float) -> None:
        """Update the tool execution timeout."""
        self.tool_timeout = timeout
        if self._executor and hasattr(self._executor.strategy, "default_timeout"):
            self._executor.strategy.default_timeout = timeout

    def get_tool_timeout(self) -> float:
        """Get the current tool timeout value."""
        return self.tool_timeout

    # Resource access (kept for compatibility)
    async def list_prompts(self) -> List[Dict[str, Any]]:
        """Return all prompts from servers."""
        if self.stream_manager and hasattr(self.stream_manager, "list_prompts"):
            try:
                return await asyncio.wait_for(
                    self.stream_manager.list_prompts(), timeout=10.0
                )
            except Exception:
                pass
        return []

    async def list_resources(self) -> List[Dict[str, Any]]:
        """Return all resources from servers."""
        if self.stream_manager and hasattr(self.stream_manager, "list_resources"):
            try:
                return await asyncio.wait_for(
                    self.stream_manager.list_resources(), timeout=10.0
                )
            except Exception:
                pass
        return []

    def get_streams(self):
        """
        Get streams from the stream manager for backward compatibility.

        Returns:
            Generator of (read_stream, write_stream) tuples
        """
        if self.stream_manager:
            return self.stream_manager.get_streams()
        return []


# Global singleton
_tool_manager: Optional[ToolManager] = None


def get_tool_manager() -> Optional[ToolManager]:
    """Get the global tool manager instance."""
    return _tool_manager


def set_tool_manager(manager: ToolManager) -> None:
    """Set the global tool manager instance."""
    global _tool_manager
    _tool_manager = manager
