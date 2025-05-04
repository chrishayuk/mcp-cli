# mcp_cli/tools/manager.py
"""
Centralized tool management using CHUK Tool Processor.

This module provides a unified interface for all tool-related operations in MCP CLI,
abstracting away the underlying CHUK Tool Processor implementation details.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union

from chuk_tool_processor.mcp import setup_mcp_stdio
from chuk_tool_processor.core.processor import ToolProcessor
from chuk_tool_processor.registry import ToolRegistryProvider
from chuk_tool_processor.mcp.stream_manager import StreamManager
from chuk_tool_processor.models.tool_result import ToolResult

from mcp_cli.tools.models import ToolInfo, ServerInfo, ToolCallResult
from mcp_cli.tools.adapter import ToolNameAdapter

logger = logging.getLogger(__name__)


class ToolManager:
    """
    Central interface for all tool operations in MCP CLI.

    This class wraps CHUK Tool Processor and provides a clean API for:
    - Tool discovery and listing
    - Tool execution with proper ToolProcessor integration
    - Server management
    - LLM-compatible tool conversion
    """

    # ------------------------------------------------------------------ #
    # Construction / initialization
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
        except Exception as exc:
            logger.error(f"Error initializing tool manager: {exc}")
            return False

    async def close(self):
        """Close all resources and connections."""
        if self.stream_manager:
            await self.stream_manager.close()

    # ------------------------------------------------------------------ #
    # Discovery helpers
    # ------------------------------------------------------------------ #
    def _all_registry_items(self) -> List[Tuple[str, str]]:
        """Get all (namespace, name) pairs from the registry."""
        return self._registry.list_tools() if self._registry else []

    def _metadata(self, name: str, ns: str) -> Optional[Any]:
        """Get metadata for a tool."""
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
        """Get tool info by name and optional namespace."""
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
    # Formatting helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def format_tool_response(response_content: Union[List[Dict[str, Any]], Any]) -> str:
        """
        Format the response content from a tool in a way that's suitable for LLM consumption.
        
        Args:
            response_content: Raw response content from tool execution
            
        Returns:
            Formatted string representation
        """
        # Handle list of dictionaries (likely structured data like SQL results)
        if isinstance(response_content, list) and response_content and isinstance(response_content[0], dict):
            # Check if this looks like text records with type field
            if all(item.get("type") == "text" for item in response_content if "type" in item):
                # Text records - extract just the text
                return "\n".join(
                    item.get("text", "No content")
                    for item in response_content
                    if item.get("type") == "text"
                )
            else:
                # This could be data records (like SQL results)
                # Return a JSON representation that preserves all data
                try:
                    return json.dumps(response_content, indent=2)
                except:
                    # Fallback if JSON serialization fails
                    return str(response_content)
        elif isinstance(response_content, dict):
            # Single dictionary - return as JSON
            try:
                return json.dumps(response_content, indent=2)
            except:
                return str(response_content)
        else:
            # Default case - convert to string
            return str(response_content)

    @staticmethod
    def convert_to_openai_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert a list of tool metadata dictionaries into the OpenAI function call format.
        
        Args:
            tools: List of tool metadata dictionaries
            
        Returns:
            List of OpenAI-compatible function definitions
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
    # Execution helpers
    # ------------------------------------------------------------------ #
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCallResult:
        """
        Execute a tool using the ToolProcessor.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            ToolCallResult with success status and result/error
        """
        if not self.processor:
            return ToolCallResult(tool_name, False, error="Tool manager not initialized")

        # Resolve the fully-qualified call name expected by CHUK
        if self._registry and self._metadata(tool_name, "stdio"):
            call_name = f"stdio.{tool_name}"
        else:
            call_name = tool_name

        try:
            # Format the tool call as XML for the processor
            llm_text = f'<tool name="{call_name}" args=\'{json.dumps(arguments)}\'/>'
            results = await self.processor.process_text(llm_text)
        except Exception as exc:
            logger.error(f"Error executing tool {tool_name}: {exc}")
            return ToolCallResult(tool_name, False, error=str(exc))

        if not results:
            return ToolCallResult(tool_name, False, error="No result returned")

        # Extract the result from the first tool result
        r = results[0]
        return ToolCallResult(
            tool_name=tool_name,
            success=not bool(r.error),
            result=r.result,
            error=r.error,
            execution_time=(r.end_time - r.start_time).total_seconds(),
        )

    async def process_llm_tool_calls(
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
            List of tool results
        """
        if not self.processor:
            logger.error("Tool processor not initialized")
            return []
            
        results = []
        
        for tc in tool_calls:
            if not (tc.get("function") and "name" in tc.get("function", {})):
                continue
                
            openai_name = tc["function"]["name"]
            tool_call_id = tc.get("id") or f"call_{openai_name}_{uuid.uuid4().hex[:8]}"
            
            # Convert tool name if needed
            original_name = name_mapping.get(openai_name, openai_name)
            
            # Parse arguments
            args_str = tc["function"].get("arguments", "{}")
            args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
            
            logger.debug(f"Processing tool call: {original_name} with args: {args_dict}")
            
            # If conversation history is provided, add the tool call first
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
            
            # Create tool call in MCP format and execute
            tool_call_text = f'<tool name="{original_name}" args=\'{json.dumps(args_dict)}\'/>'
            execution_results = await self.processor.process_text(tool_call_text)
            
            # Add to results list
            results.extend(execution_results)
            
            # Update conversation history if provided
            if conversation_history is not None and execution_results:
                result = execution_results[0]
                
                if result.error:
                    content = f"Error: {result.error}"
                else:
                    content = self.format_tool_response(result.result)
                
                conversation_history.append({
                    "role": "tool",
                    "name": original_name,
                    "content": content,
                    "tool_call_id": tool_call_id,
                })
        
        return results

    # ------------------------------------------------------------------ #
    # Server / stream helpers
    # ------------------------------------------------------------------ #
    def _extract_namespace(self, server_name: str) -> str:
        """Extract namespace from a server name."""
        return server_name.split("_", 1)[0] if "_" in server_name else server_name

    def get_server_info(self) -> List[ServerInfo]:
        """Get information about all connected servers."""
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
        """Get the server name for a tool."""
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
        """
        Get OpenAI-compatible tool definitions for all unique tools.
        
        Note: This uses the original names (with dots) which may not work with OpenAI.
        Use get_adapted_tools_for_llm() instead for OpenAI compatibility.
        
        Returns:
            List of tool definitions in OpenAI format
        """
        if not self._registry:
            return []
        unique = self.get_unique_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": f"{t.namespace}.{t.name}",
                    "description": t.description or "",
                    "parameters": t.parameters or {},
                },
            }
            for t in unique
        ]

    def get_adapted_tools_for_llm(self, provider: str = "openai") -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Get tools in a format compatible with the specified LLM provider.
        
        This method handles the needed name transformations for different LLM providers.
        
        Args:
            provider: LLM provider name (openai, ollama, etc.)
            
        Returns:
            Tuple of (llm_compatible_tools, name_mapping)
        """
        unique_tools = self.get_unique_tools()
        
        # Different providers have different naming constraints
        adapter_needed = provider.lower() == "openai"
        
        llm_tools = []
        name_mapping = {}
        
        for tool in unique_tools:
            # For OpenAI, we need to transform names to avoid dots
            if adapter_needed:
                tool_name = ToolNameAdapter.to_openai_compatible(tool.namespace, tool.name)
                original_name = f"{tool.namespace}.{tool.name}"
                name_mapping[tool_name] = original_name
                description = f"{tool.description or ''} (Original: {original_name})"
            else:
                # For other providers, use original name with namespace
                tool_name = f"{tool.namespace}.{tool.name}"
                description = tool.description or ""
            
            llm_tools.append({
                "name": tool_name,
                "description": description,
                "parameters": tool.parameters or {}
            })
        
        # Convert to OpenAI function format
        function_tools = self.convert_to_openai_tools(llm_tools)
        
        return function_tools, name_mapping

    # ------------------------------------------------------------------ #
    # Compatibility shims
    # ------------------------------------------------------------------ #
    def get_streams(self):
        """
        Legacy helper so commands like **/resources** and **/prompts** that
        expect a low-level StreamManager continue to work when they
        receive a ToolManager instance.
        """
        if self.stream_manager and hasattr(self.stream_manager, "get_streams"):
            return self.stream_manager.get_streams()
        return []

    def list_prompts(self) -> List[Dict[str, Any]]:
        """
        Return all prompts recorded on each server.
        Delegates to the underlying StreamManager if available.
        """
        if self.stream_manager and hasattr(self.stream_manager, "list_prompts"):
            return self.stream_manager.list_prompts()
        return []

    def list_resources(self) -> List[Dict[str, Any]]:
        """
        Return all resources (URI, size, MIME-type) on each server.
        Delegates to the underlying StreamManager if available.
        """
        if self.stream_manager and hasattr(self.stream_manager, "list_resources"):
            return self.stream_manager.list_resources()
        return []


# ---------------------------------------------------------------------- #
# Global singleton accessor
# ---------------------------------------------------------------------- #
_tool_manager: Optional[ToolManager] = None


def get_tool_manager() -> Optional[ToolManager]:
    """Get the global tool manager instance."""
    return _tool_manager


def set_tool_manager(manager: ToolManager) -> None:
    """Set the global tool manager instance."""
    global _tool_manager
    _tool_manager = manager