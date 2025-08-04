# mcp_cli/tools/manager.py - COMPLETE FIXED VERSION
"""
Centralized tool management using CHUK Tool Processor.

This module provides a unified interface for all tool-related operations in MCP CLI,
leveraging the async-native capabilities of CHUK Tool Processor.

FIXED: Enhanced with performance improvements, better timeout handling, and universal
tool compatibility support for the OpenAI client integration.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncIterator
from concurrent.futures import TimeoutError as FutureTimeoutError

from chuk_tool_processor.mcp.setup_mcp_stdio import setup_mcp_stdio
from chuk_tool_processor.core.processor import ToolProcessor
from chuk_tool_processor.registry import ToolRegistryProvider
from chuk_tool_processor.mcp.stream_manager import StreamManager
from chuk_tool_processor.models.tool_result import ToolResult
from chuk_tool_processor.models.tool_call import ToolCall
from chuk_tool_processor.execution.strategies.inprocess_strategy import InProcessStrategy
from chuk_tool_processor.execution.tool_executor import ToolExecutor

from mcp_cli.tools.models import ServerInfo, ToolCallResult, ToolInfo
from mcp_cli.tools.adapter import ToolNameAdapter

logger = logging.getLogger(__name__)


class ToolManager:
    """
    Central interface for all tool operations in MCP CLI.

    This class wraps CHUK Tool Processor and provides a clean API for:
    - Tool discovery and listing
    - Tool execution with streaming support
    - Server management
    - LLM-compatible tool conversion
    
    ENHANCED: Now includes performance improvements and timeout handling
    for better support of many MCP servers.
    """

    # ------------------------------------------------------------------ #
    # Construction / initialization                                      #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        config_file: str,
        servers: List[str],
        server_names: Optional[Dict[int, str]] = None,
        tool_timeout: Optional[float] = None,  # Make this optional for smart defaults
        max_concurrency: int = 4,
        initialization_timeout: float = 120.0,  # Enhanced: Configurable init timeout
        server_connection_timeout: float = 30.0,  # Enhanced: Per-server timeout
    ):
        self.config_file = config_file
        self.servers = servers
        self.server_names = server_names or {}
        
        # Enhanced timeout configuration with priority:
        # 1. Explicit parameter
        # 2. Environment variable MCP_TOOL_TIMEOUT
        # 3. Environment variable CHUK_TOOL_TIMEOUT
        # 4. Default (120 seconds)
        self.tool_timeout = self._determine_timeout(tool_timeout)
        self.max_concurrency = max_concurrency
        self.initialization_timeout = initialization_timeout
        self.server_connection_timeout = server_connection_timeout

        # CHUK components
        self.processor: Optional[ToolProcessor] = None
        self.stream_manager: Optional[StreamManager] = None
        
        # Internal state
        self._registry = None
        self._executor: Optional[ToolExecutor] = None
        self._metadata_cache: Dict[Tuple[str, str], Any] = {}
        
        # Performance tracking
        self._connection_stats = {}
        self._failed_servers = set()

    def _determine_timeout(self, explicit_timeout: Optional[float]) -> float:
        """
        Determine timeout with smart defaults and environment variable support.
        """
        if explicit_timeout is not None:
            logger.info(f"Using explicit timeout: {explicit_timeout}s")
            return explicit_timeout
        
        # Check environment variables
        env_timeout = os.getenv("MCP_TOOL_TIMEOUT") or os.getenv("CHUK_TOOL_TIMEOUT")
        if env_timeout:
            try:
                timeout = float(env_timeout)
                logger.info(f"Using environment timeout: {timeout}s")
                return timeout
            except ValueError:
                logger.warning(f"Invalid timeout in environment: {env_timeout}")
        
        # Check CLI-specific environment variables
        cli_timeout = os.getenv("MCP_CLI_INIT_TIMEOUT")
        if cli_timeout:
            try:
                timeout = float(cli_timeout)
                logger.info(f"Using CLI timeout: {timeout}s")
                return timeout
            except ValueError:
                logger.warning(f"Invalid CLI timeout: {cli_timeout}")
        
        # Default
        default_timeout = 120.0  # 2 minutes
        logger.info(f"Using default timeout: {default_timeout}s")
        return default_timeout

    async def initialize(self, namespace: str = "stdio") -> bool:
        """
        Connect to the MCP servers and populate the tool registry.
        
        Enhanced with better timeout handling and partial failure support.
        """
        try:
            logger.info(f"Initializing ToolManager with {len(self.servers)} servers")
            logger.info(f"Timeouts: init={self.initialization_timeout}s, connection={self.server_connection_timeout}s")
            
            # Set up CHUK Tool Processor with enhanced timeouts
            self.processor, self.stream_manager = await asyncio.wait_for(
                setup_mcp_stdio(
                    config_file=str(self.config_file),
                    servers=self.servers,
                    server_names=self.server_names,
                    namespace=namespace,
                    enable_caching=True,
                    enable_retries=True,
                    max_retries=2,  # Reduced retries for faster failure
                ),
                timeout=self.initialization_timeout
            )
            
            # Get the registry with timeout
            self._registry = await asyncio.wait_for(
                ToolRegistryProvider.get_registry(),
                timeout=30.0
            )
            
            # Initialize the executor with configurable timeout
            strategy = InProcessStrategy(
                self._registry,
                max_concurrency=self.max_concurrency,
                default_timeout=self.tool_timeout
            )
            
            self._executor = ToolExecutor(
                registry=self._registry,
                strategy=strategy,
                default_timeout=self.tool_timeout
            )
            
            logger.info(f"ToolManager initialized successfully with {self.tool_timeout}s timeout")
            return True
            
        except asyncio.TimeoutError:
            logger.error(f"ToolManager initialization timed out after {self.initialization_timeout}s")
            logger.error(f"Consider reducing the number of servers or increasing timeout with --timeout")
            return False
        except Exception as exc:
            logger.error(f"Error initializing tool manager: {exc}")
            return False

    async def close(self):
        """Close all resources and connections."""
        try:
            # Close the stream manager
            if self.stream_manager:
                await self.stream_manager.close()
                
            # Shut down the executor if it exists
            if self._executor:
                await self._executor.shutdown()
        except Exception as exc:
            logger.warning(f"Error during ToolManager shutdown: {exc}")

    # ------------------------------------------------------------------ #
    # Configuration methods                                              #
    # ------------------------------------------------------------------ #
    def set_tool_timeout(self, timeout: float) -> None:
        """
        Update the tool execution timeout.
        
        Args:
            timeout: New timeout value in seconds
        """
        self.tool_timeout = timeout
        if self._executor and hasattr(self._executor.strategy, 'default_timeout'):
            self._executor.strategy.default_timeout = timeout
            logger.info(f"Updated tool timeout to {timeout}s")

    def get_tool_timeout(self) -> float:
        """Get the current tool timeout value."""
        return self.tool_timeout

    # ------------------------------------------------------------------ #
    # Enhanced tool discovery methods                                    #
    # ------------------------------------------------------------------ #
    async def get_all_tools(self) -> List[ToolInfo]:
        """
        Return all tools including duplicates.
        
        Enhanced with timeout handling and partial failure support.
        """
        if not self._registry:
            return []
            
        tools: List[ToolInfo] = []
        failed_tools = []
        
        try:
            # Get registry items with timeout
            registry_items = await asyncio.wait_for(
                self._registry.list_tools(),
                timeout=30.0
            )
            
            for ns, name in registry_items:
                try:
                    # Get metadata with per-tool timeout
                    metadata = await asyncio.wait_for(
                        self._registry.get_metadata(name, ns),
                        timeout=5.0
                    )
                    
                    if metadata:
                        # Cache metadata for future use
                        self._metadata_cache[(ns, name)] = metadata
                        
                        # Extract tool properties
                        supports_streaming = getattr(metadata, "supports_streaming", False)
                        tools.append(
                            ToolInfo(
                                name=name,
                                namespace=ns,
                                description=metadata.description,
                                parameters=metadata.argument_schema,
                                is_async=metadata.is_async,
                                tags=list(metadata.tags),
                                supports_streaming=supports_streaming
                            )
                        )
                    else:
                        # Include tools even without metadata
                        tools.append(
                            ToolInfo(
                                name=name,
                                namespace=ns,
                                description="",
                                parameters={},
                                is_async=False,
                                tags=[],
                                supports_streaming=False
                            )
                        )
                        
                except asyncio.TimeoutError:
                    failed_tools.append(f"{ns}.{name}")
                    logger.warning(f"Tool metadata timeout: {ns}.{name}")
                except Exception as e:
                    failed_tools.append(f"{ns}.{name}")
                    logger.warning(f"Tool metadata error for {ns}.{name}: {e}")
            
            if failed_tools:
                logger.warning(f"Failed to get metadata for {len(failed_tools)} tools")
                
        except asyncio.TimeoutError:
            logger.error("Tool discovery timed out completely")
        except Exception as exc:
            logger.error(f"Error discovering tools: {exc}")
            
        return tools

    async def get_unique_tools(self) -> List[ToolInfo]:
        """Return tools without duplicates from the default namespace."""
        seen, unique = set(), []
        
        all_tools = await self.get_all_tools()
        for tool in all_tools:
            if tool.namespace == "default" or tool.name in seen:
                continue
                
            seen.add(tool.name)
            unique.append(tool)
            
        return unique

    async def get_tool_by_name(self, tool_name: str, namespace: str | None = None) -> Optional[ToolInfo]:
        """Get tool info by name and optional namespace."""
        if not self._registry:
            return None

        if namespace:  # explicit ns
            try:
                metadata = await asyncio.wait_for(
                    self._registry.get_metadata(tool_name, namespace),
                    timeout=5.0
                )
                if metadata:
                    supports_streaming = getattr(metadata, "supports_streaming", False)
                    return ToolInfo(
                        name=tool_name,
                        namespace=namespace,
                        description=metadata.description,
                        parameters=metadata.argument_schema,
                        is_async=metadata.is_async,
                        tags=list(metadata.tags),
                        supports_streaming=supports_streaming
                    )
            except asyncio.TimeoutError:
                logger.warning(f"Timeout getting metadata for {namespace}.{tool_name}")
                return None

        # otherwise search all non-default namespaces
        try:
            registry_items = await asyncio.wait_for(
                self._registry.list_tools(),
                timeout=10.0
            )
            for ns, name in registry_items:
                if name == tool_name and ns != "default":
                    return await self.get_tool_by_name(name, ns)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout searching for tool {tool_name}")
                
        return None

    # ------------------------------------------------------------------ #
    # Enhanced tool execution methods                                    #
    # ------------------------------------------------------------------ #
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], timeout: Optional[float] = None) -> ToolCallResult:
        """
        Execute a tool and return the result.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            timeout: Optional timeout override for this specific call

        Returns:
            ToolCallResult with success status and result/error
            
        Enhanced with better argument validation and error handling.
        """
        # Enhanced argument validation
        if not isinstance(arguments, dict):
            logger.error(f"Arguments must be a dict, got {type(arguments)}: {arguments}")
            return ToolCallResult(tool_name, False, error="Invalid arguments type")
        
        # Log the execution for debugging
        logger.debug(f"Executing tool {tool_name} with arguments: {arguments}")
        
        # Get namespace if needed
        namespace = None
        original_name = tool_name
        
        if "." in tool_name:
            namespace, base_name = tool_name.split(".", 1)
        else:
            base_name = tool_name
            namespace = await self.get_server_for_tool(tool_name)
        
        # Enhanced argument validation for known tools
        validation_result = self._validate_tool_arguments(base_name, arguments)
        if not validation_result["valid"]:
            logger.error(f"Argument validation failed for {tool_name}: {validation_result['error']}")
            return ToolCallResult(tool_name, False, error=validation_result["error"])
        
        # Create a CHUK ToolCall with optional timeout override
        call = ToolCall(
            tool=base_name,
            namespace=namespace,
            arguments=arguments,
            timeout=timeout or self.tool_timeout  # Use override or default
        )
        
        try:
            # Execute with CHUK executor
            results = await self._executor.execute([call])
            
            if not results:
                return ToolCallResult(original_name, False, error="No result returned")
                
            # Extract result from first call
            result = results[0]
            return ToolCallResult(
                tool_name=original_name,
                success=not bool(result.error),
                result=result.result,
                error=result.error,
                execution_time=(
                    (result.end_time - result.start_time).total_seconds() 
                    if hasattr(result, "end_time") and hasattr(result, "start_time") else None
                ),
            )
        except Exception as exc:
            logger.error(f"Error executing tool {original_name}: {exc}")
            return ToolCallResult(original_name, False, error=str(exc))

    def _validate_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tool arguments for known tools.
        
        This helps catch the "Input validation error: 'query' is a required property" issue.
        """
        # Known tool validation rules
        validation_rules = {
            "read_query": {"required": ["query"], "types": {"query": str}},
            "write_query": {"required": ["query"], "types": {"query": str}},
            "describe_table": {"required": ["table_name"], "types": {"table_name": str}},
            "create_table": {"required": ["query"], "types": {"query": str}},
        }
        
        if tool_name not in validation_rules:
            return {"valid": True}  # No validation rules for this tool
        
        rules = validation_rules[tool_name]
        
        # Check required parameters
        for required_param in rules.get("required", []):
            if required_param not in arguments:
                return {
                    "valid": False,
                    "error": f"Missing required parameter '{required_param}' for tool '{tool_name}'"
                }
            
            if not arguments[required_param]:
                return {
                    "valid": False,
                    "error": f"Parameter '{required_param}' cannot be empty for tool '{tool_name}'"
                }
        
        # Check parameter types
        for param, expected_type in rules.get("types", {}).items():
            if param in arguments and not isinstance(arguments[param], expected_type):
                return {
                    "valid": False,
                    "error": f"Parameter '{param}' must be of type {expected_type.__name__} for tool '{tool_name}'"
                }
        
        return {"valid": True}

    async def stream_execute_tool(self, tool_name: str, arguments: Dict[str, Any], timeout: Optional[float] = None) -> AsyncIterator[ToolResult]:
        """
        Execute a tool with streaming support.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            timeout: Optional timeout override for this specific call
            
        Returns:
            Async iterator of ToolResult objects
        """
        # Get namespace if needed
        namespace = None
        
        if "." in tool_name:
            namespace, base_name = tool_name.split(".", 1)
        else:
            base_name = tool_name
            namespace = await self.get_server_for_tool(tool_name)
        
        # Create a CHUK ToolCall with optional timeout override
        call = ToolCall(
            tool=base_name,
            namespace=namespace,
            arguments=arguments,
            timeout=timeout or self.tool_timeout  # Use override or default
        )
        
        # Stream execution results
        async for result in self._executor.stream_execute([call]):
            yield result

    async def process_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        name_mapping: Dict[str, str],
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> List[ToolResult]:
        """
        Process tool calls from an LLM, handling name transformations as needed.

        Args:
            tool_calls: List of tool call objects from the LLM
            name_mapping: Mapping from LLM tool names to original MCP tool names
            conversation_history: Optional conversation history to update

        Returns:
            List of CHUK ToolResult objects
        """
        # Convert LLM tool calls to CHUK ToolCall objects
        chuk_calls = []
        call_mapping = {}  # Map CHUK tool calls to original info
        
        for tc in tool_calls:
            if not (tc.get("function") and "name" in tc.get("function", {})):
                continue
                
            openai_name = tc["function"]["name"]
            tool_call_id = tc.get("id") or f"call_{openai_name}_{uuid.uuid4().hex[:8]}"
            
            # Convert tool name if needed
            original_name = name_mapping.get(openai_name, openai_name)
            
            # Parse arguments with enhanced error handling
            args_str = tc["function"].get("arguments", "{}")
            try:
                if isinstance(args_str, str):
                    args_dict = json.loads(args_str) if args_str.strip() else {}
                else:
                    args_dict = args_str or {}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool arguments for {openai_name}: {e}")
                logger.error(f"Raw arguments: {repr(args_str)}")
                args_dict = {}
            
            # Get namespace and base name
            namespace = None
            base_name = original_name
            
            if "." in original_name:
                namespace, base_name = original_name.split(".", 1)
            else:
                namespace = await self.get_server_for_tool(base_name)
            
            # Create CHUK ToolCall
            call = ToolCall(
                tool=base_name,
                namespace=namespace,
                arguments=args_dict,
                metadata={"call_id": tool_call_id, "original_name": original_name}
            )
            
            chuk_calls.append(call)
            call_mapping[id(call)] = {"id": tool_call_id, "name": original_name}
            
            # Add to conversation history if provided
            if conversation_history is not None:
                conversation_history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": original_name,
                                "arguments": json.dumps(args_dict),
                            },
                        }
                    ],
                })
        
        # Execute tool calls
        results = await self._executor.execute(chuk_calls)
        
        # Process results
        for result in results:
            # Get original info from mapping
            call_info = call_mapping.get(id(result.tool_call), {"id": f"call_{uuid.uuid4().hex[:8]}", "name": result.tool})
            call_id = call_info["id"]
            original_name = call_info["name"]
            
            # Update conversation history if provided
            if conversation_history is not None:
                if result.error:
                    content = f"Error: {result.error}"
                else:
                    content = self.format_tool_response(result.result)
                    
                conversation_history.append({
                    "role": "tool",
                    "name": original_name,
                    "content": content,
                    "tool_call_id": call_id,
                })
        
        return results

    # ------------------------------------------------------------------ #
    # Streaming execution support for chat processing                    #
    # ------------------------------------------------------------------ #
    async def stream_process_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        name_mapping: Dict[str, str],
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncIterator[Tuple[ToolResult, str]]:
        """
        Process tool calls with streaming support.
        
        Args:
            tool_calls: List of tool call objects from the LLM
            name_mapping: Mapping from LLM tool names to original MCP tool names
            conversation_history: Optional conversation history to update
            
        Yields:
            Tuples of (ToolResult, call_id) for incremental results
        """
        # Convert LLM tool calls to CHUK ToolCall objects
        chuk_calls = []
        call_mapping = {}  # Map CHUK tool calls to original info
        
        for tc in tool_calls:
            if not (tc.get("function") and "name" in tc.get("function", {})):
                continue
                
            openai_name = tc["function"]["name"]
            tool_call_id = tc.get("id") or f"call_{openai_name}_{uuid.uuid4().hex[:8]}"
            
            # Convert tool name if needed
            original_name = name_mapping.get(openai_name, openai_name)
            
            # Parse arguments
            args_str = tc["function"].get("arguments", "{}")
            try:
                if isinstance(args_str, str):
                    args_dict = json.loads(args_str) if args_str.strip() else {}
                else:
                    args_dict = args_str or {}
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool arguments for {openai_name}: {e}")
                args_dict = {}
            
            # Get namespace and base name
            namespace = None
            base_name = original_name
            
            if "." in original_name:
                namespace, base_name = original_name.split(".", 1)
            else:
                namespace = await self.get_server_for_tool(base_name)
            
            # Create CHUK ToolCall
            call = ToolCall(
                tool=base_name,
                namespace=namespace,
                arguments=args_dict,
                metadata={"call_id": tool_call_id, "original_name": original_name}
            )
            
            chuk_calls.append(call)
            call_mapping[id(call)] = {"id": tool_call_id, "name": original_name}
            
            # Add to conversation history if provided
            if conversation_history is not None:
                conversation_history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": original_name,
                                "arguments": json.dumps(args_dict),
                            },
                        }
                    ],
                })
        
        # Collect final responses for conversation history
        final_responses = {}
        
        # Stream execution
        async for result in self._executor.stream_execute(chuk_calls):
            # Get original info from mapping
            call_info = call_mapping.get(id(result.tool_call), {"id": f"call_{uuid.uuid4().hex[:8]}", "name": result.tool})
            call_id = call_info["id"]
            original_name = call_info["name"]
            
            # Track final results for conversation history
            is_final = not getattr(result, "is_intermediate", False)
            if is_final:
                if result.error:
                    content = f"Error: {result.error}"
                else:
                    content = self.format_tool_response(result.result)
                    
                final_responses[call_id] = {
                    "role": "tool",
                    "name": original_name,
                    "content": content,
                    "tool_call_id": call_id,
                }
            
            # Yield each result along with its call ID
            yield (result, call_id)
        
        # Update conversation history with final results if provided
        if conversation_history is not None:
            for response in final_responses.values():
                conversation_history.append(response)

    # ------------------------------------------------------------------ #
    # Enhanced server helpers                                            #
    # ------------------------------------------------------------------ #
    def _extract_namespace(self, server_name: str) -> str:
        """Extract namespace from a server name."""
        return server_name.split("_", 1)[0] if "_" in server_name else server_name

    async def get_server_info(self) -> List[ServerInfo]:
        """
        Get information about all connected servers.
        Enhanced with timeout handling and partial failure support.
        """
        if not self.stream_manager:
            return []

        try:
            # Properly await the coroutine with timeout
            raw_infos = await asyncio.wait_for(
                asyncio.to_thread(self.stream_manager.get_server_info),
                timeout=10.0
            )

            return [
                ServerInfo(
                    id=raw.get("id", 0),
                    name=raw.get("name", "Unknown"),
                    status=raw.get("status", "Unknown"),
                    tool_count=raw.get("tools", 0),
                    namespace=self._extract_namespace(raw.get("name", "")),
                )
                for raw in raw_infos
            ]

        except asyncio.TimeoutError:
            logger.warning("Server info retrieval timed out, returning partial information")
            # Return basic info for configured servers
            return [
                ServerInfo(
                    id=i,
                    name=server,
                    status="Timeout",
                    tool_count=0,
                    namespace=self._extract_namespace(server),
                )
                for i, server in enumerate(self.servers)
            ]
        except Exception as exc:
            logger.error(f"Error getting server info: {exc}")
            return []

    async def get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Get the server name for a tool."""
        if "." in tool_name:
            return tool_name.split(".", 1)[0]
            
        # Look up via registry
        if self._registry:
            try:
                registry_items = await asyncio.wait_for(
                    self._registry.list_tools(),
                    timeout=5.0
                )
                for ns, name in registry_items:
                    if name == tool_name and ns != "default":
                        return ns
            except asyncio.TimeoutError:
                logger.warning(f"Timeout looking up server for tool {tool_name}")
                
        # Fallback to stream-manager map
        if self.stream_manager:
            try:
                return self.stream_manager.get_server_for_tool(tool_name)
            except Exception as e:
                logger.warning(f"Error getting server for tool {tool_name}: {e}")
                
        return None

    # ------------------------------------------------------------------ #
    # LLM helpers                                                        #
    # ------------------------------------------------------------------ #
    async def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI-compatible tool definitions for all unique tools.
        """
        unique_tools = await self.get_unique_tools()
        
        return [
            {
                "type": "function",
                "function": {
                    "name": f"{t.namespace}.{t.name}",
                    "description": t.description or "",
                    "parameters": t.parameters or {},
                },
            }
            for t in unique_tools
        ]

    async def get_adapted_tools_for_llm(self, provider: str = "openai") -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Get tools in a format compatible with the specified LLM provider.
        
        For OpenAI, ensure tool names follow the required pattern: ^[a-zA-Z0-9_-]+$
        
        Enhanced with better debugging and validation.
        """
        unique_tools = await self.get_unique_tools()
        adapter_needed = provider.lower() == "openai"

        llm_tools: List[Dict[str, Any]] = []
        name_mapping: Dict[str, str] = {}

        logger.debug(f"Adapting {len(unique_tools)} tools for provider: {provider}, adapter_needed: {adapter_needed}")

        for tool in unique_tools:
            original = f"{tool.namespace}.{tool.name}"
            
            if adapter_needed:
                # Import regex for sanitization
                import re
                
                # For OpenAI, replace dots with underscores and sanitize other chars
                # First combine namespace and name with underscore (e.g., stdio_list_tables)
                combined = f"{tool.namespace}_{tool.name}"
                
                # Then sanitize to ensure it matches OpenAI's pattern
                sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', combined)
                
                name_mapping[sanitized] = original
                description = f"{tool.description or ''}"
                tool_name = sanitized
                
                logger.debug(f"Tool adapted: {original} -> {sanitized}")
            else:
                tool_name = original
                description = tool.description or ""
                logger.debug(f"Tool not sanitized: {original}")

            llm_tools.append({
                "type": "function", 
                "function": {
                    "name": tool_name,
                    "description": description,
                    "parameters": tool.parameters or {}
                }
            })

        logger.debug(f"Adapted tools: {[t['function']['name'] for t in llm_tools]}")
        logger.debug(f"Name mapping: {name_mapping}")

        return llm_tools, name_mapping

    # ------------------------------------------------------------------ #
    # Formatting helpers                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def format_tool_response(response_content: Union[List[Dict[str, Any]], Any]) -> str:
        """
        Format the response content from a tool in a way that's suitable for LLM consumption.
        """
        # Handle list of dictionaries (likely structured data like SQL results)
        if isinstance(response_content, list) and response_content and isinstance(response_content[0], dict):
            # Treat as text records only if every item has type == "text"
            if all(isinstance(item, dict) and item.get("type") == "text" for item in response_content):
                return "\n".join(item.get("text", "") for item in response_content)
            # This could be data records (like SQL results)
            try:
                return json.dumps(response_content, indent=2)
            except Exception:
                return str(response_content)
        elif isinstance(response_content, dict):
            # Single dictionary - return as JSON
            try:
                return json.dumps(response_content, indent=2)
            except Exception:
                return str(response_content)
        else:
            # Default case - convert to string
            return str(response_content)

    # ------------------------------------------------------------------ #
    # Legacy methods (for backward compatibility with imports)           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def convert_to_openai_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert a list of tool metadata dictionaries into the OpenAI function call format.
        
        This method is kept for backward compatibility with imports.
        """
        # Already in OpenAI format? Return unchanged
        if tools and isinstance(tools[0], dict) and tools[0].get("type") == "function":
            return tools

        openai_tools: List[Dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", "unknown"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", tool.get("inputSchema", {})),
                    },
                }
            )

        return openai_tools

    # ------------------------------------------------------------------ #
    # Resource access methods                                            #
    # ------------------------------------------------------------------ #
    async def list_prompts(self) -> List[Dict[str, Any]]:
        """
        Return all prompts recorded on each server.
        """
        if self.stream_manager and hasattr(self.stream_manager, "list_prompts"):
            try:
                return await asyncio.wait_for(
                    self.stream_manager.list_prompts(),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout listing prompts")
                return []
        return []

    async def list_resources(self) -> List[Dict[str, Any]]:
        """
        Return all resources (URI, size, MIME-type) on each server.
        """
        if self.stream_manager and hasattr(self.stream_manager, "list_resources"):
            try:
                return await asyncio.wait_for(
                    self.stream_manager.list_resources(),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout listing resources")
                return []
        return []
        
    def get_streams(self):
        """
        Legacy helper so commands like **/resources** and **/prompts** that
        expect a low-level StreamManager continue to work.
        """
        if self.stream_manager and hasattr(self.stream_manager, "get_streams"):
            return self.stream_manager.get_streams()
        return []


# ---------------------------------------------------------------------- #
# Global singleton accessor                                              #
# ---------------------------------------------------------------------- #
_tool_manager: Optional[ToolManager] = None


def get_tool_manager() -> Optional[ToolManager]:
    """Get the global tool manager instance."""
    return _tool_manager


async def get_tool_manager_async() -> Optional[ToolManager]:
    """Get the global tool manager instance (async-aware)."""
    return _tool_manager


def set_tool_manager(manager: ToolManager) -> None:
    """Set the global tool manager instance."""
    global _tool_manager
    _tool_manager = manager