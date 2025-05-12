# mcp_cli/chat/chat_context.py
"""
Chat context handling for the MCP CLI.

This module provides a context for managing conversation state and tool access
during a chat session. It interfaces with the ToolManager for tool execution
and discovery, and manages the conversation history.
"""

from __future__ import annotations

import asyncio
import gc
import logging
from typing import Any, Dict, List, Optional, AsyncIterator

from rich import print
from rich.console import Console

# LLM utilities
from mcp_cli.llm.llm_client import get_llm_client

# Prompt generator
from mcp_cli.chat.system_prompt import generate_system_prompt

# Tools and configuration
from mcp_cli.tools.manager import ToolManager
from mcp_cli.provider_config import ProviderConfig

# For backward compatibility with imports
convert_to_openai_tools = ToolManager.convert_to_openai_tools

# Set up logger
logger = logging.getLogger(__name__)


class ChatContext:
    """Manage the end-to-end conversation state for the CLI session."""

    # ------------------------------------------------------------------ #
    # construction                                                       #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        tool_manager: Optional[ToolManager] = None,
        stream_manager: Optional[Any] = None,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        provider_config: Optional[ProviderConfig] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Create a new chat context.

        Parameters
        ----------
        tool_manager
            The ToolManager instance for tool operations.
        stream_manager
            Legacy test-mode stream manager (mutually exclusive with tool_manager).
        provider
            LLM provider to use (e.g., "openai", "anthropic").
        model
            LLM model to use (e.g., "gpt-4o-mini", "claude-3-opus").
        provider_config
            Optional ProviderConfig instance for LLM configurations.
        api_base / api_key
            Optional API settings that override provider_config.
        """
        if (tool_manager is None) == (stream_manager is None):
            raise ValueError("Pass either tool_manager or stream_manager, not both")

        self.tool_manager: Optional[ToolManager] = tool_manager
        self.stream_manager: Optional[Any] = stream_manager

        self.provider = provider
        self.model = model
        self.exit_requested: bool = False
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Initialize provider configuration
        self.provider_config = provider_config or ProviderConfig()
        
        # Update provider config if API settings were provided
        if api_base or api_key:
            config_updates = {}
            if api_base:
                config_updates["api_base"] = api_base
            if api_key:
                config_updates["api_key"] = api_key
                
            self.provider_config.set_provider_config(provider, config_updates)

        # Initialize LLM client
        self.client = get_llm_client(
            provider=self.provider, 
            model=self.model,
            config=self.provider_config
        )

        # Attributes filled during initialization
        self.tools: List[Dict[str, Any]] = []
        self.internal_tools: List[Dict[str, Any]] = []
        self.server_info: List[Dict[str, Any]] = []
        self.tool_to_server_map: Dict[str, str] = {}
        self.openai_tools: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # initialization                                                     #
    # ------------------------------------------------------------------ #
    async def initialize(self) -> bool:
        """
        Build the runtime context by fetching tools and server information.
        
        Returns:
            True on successful initialization, False otherwise.
        """
        console = Console()
        try:
            with console.status("[bold cyan]Setting up chat environment…[/bold cyan]", spinner="dots"):
                # Get tools and server info based on available manager
                if self.tool_manager is not None:
                    # Get unique tools
                    logger.debug("Fetching tools from ToolManager")
                    tool_infos = await self.tool_manager.get_unique_tools()
                    
                    # Convert to dictionary format for system prompt
                    self.tools = [
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                            "namespace": t.namespace,
                            "supports_streaming": getattr(t, "supports_streaming", False),
                        }
                        for t in tool_infos
                    ]
                    
                    # Keep identical copy for system prompt
                    self.internal_tools = list(self.tools)
    
                    # Get server information
                    logger.debug("Fetching server info")
                    raw_infos = await self.tool_manager.get_server_info()
                    self.server_info = self._convert_server_info(raw_infos)
                    
                    # Build mapping from tool name to server/namespace
                    self.tool_to_server_map = {t["name"]: t["namespace"] for t in self.tools}
                    
                    # Get tool specifications for LLM
                    logger.debug("Fetching LLM tool specifications")
                    self.openai_tools = await self.tool_manager.get_tools_for_llm()
                    
                else:
                    # Legacy test mode with stream_manager
                    logger.debug("Using stream_manager for tool information (test mode)")
                    if hasattr(self.stream_manager, "get_internal_tools"):
                        self.tools = list(self.stream_manager.get_internal_tools())
                    else:
                        self.tools = list(self.stream_manager.get_all_tools())
    
                    self.internal_tools = list(self.tools)
                    self.server_info = list(self.stream_manager.get_server_info())
                    
                    # Build tool to server mapping
                    self.tool_to_server_map = {
                        t["name"]: self.stream_manager.get_server_for_tool(t["name"])
                        for t in self.tools
                    }
                    
                    # Convert tools to OpenAI format
                    self.openai_tools = convert_to_openai_tools(self.tools)
                    
            # Warn on empty tool-set
            if not self.tools:
                print("[yellow]No tools available. Chat functionality may be limited.[/yellow]")
                logger.warning("No tools found during initialization")

            # Build system prompt and set initial conversation state
            system_prompt = generate_system_prompt(self.internal_tools)
            self.conversation_history = [{"role": "system", "content": system_prompt}]
            
            logger.info(f"Chat context initialized with {len(self.tools)} tools")
            return True
            
        except Exception as exc:
            logger.exception("Error initializing chat context")
            print(f"[red]Error initializing chat context: {exc}[/red]")
            return False

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _convert_server_info(server_infos):
        """Convert ServerInfo objects into plain dictionaries."""
        result = []
        for server in server_infos:
            result.append({
                "id": server.id,
                "name": server.name,
                "tools": server.tool_count,
                "status": server.status,
            })
        return result

    # ------------------------------------------------------------------ #
    # tool and server helpers                                            #
    # ------------------------------------------------------------------ #
    async def get_server_for_tool(self, tool_name: str) -> str:
        """
        Get the server/namespace for a tool.
        
        Args:
            tool_name: The name of the tool
            
        Returns:
            Server/namespace name, or "Unknown" if not found
        """
        if self.tool_manager is not None:
            return await self.tool_manager.get_server_for_tool(tool_name) or "Unknown"
            
        # Fallback to stream manager for test mode
        return self.stream_manager.get_server_for_tool(tool_name) or "Unknown"

    @staticmethod
    def get_display_name_for_tool(namespaced_tool_name: str) -> str:
        """Return a user-friendly name for display in UI."""
        return namespaced_tool_name

    # ------------------------------------------------------------------ #
    # tool execution helpers                                             #
    # ------------------------------------------------------------------ #
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool using the appropriate manager.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            The result of the tool execution
            
        Raises:
            ValueError: If no tool manager is available
        """
        if self.tool_manager is not None:
            return await self.tool_manager.execute_tool(tool_name, arguments)
            
        elif self.stream_manager is not None and hasattr(self.stream_manager, "call_tool"):
            return await self.stream_manager.call_tool(tool_name, arguments)
            
        raise ValueError(f"No tool manager available to execute {tool_name}")
    
    async def stream_execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> AsyncIterator[Any]:
        """
        Execute a tool with streaming support.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            An async iterator of incremental results
            
        Raises:
            ValueError: If streaming is not supported
        """
        if not self.tool_manager:
            raise ValueError("Streaming execution requires ToolManager")
            
        async for result in self.tool_manager.stream_execute_tool(tool_name, arguments):
            yield result

    # ------------------------------------------------------------------ #
    # serialization helpers                                              #
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Dump the context to a plain dict for command handlers / tests."""
        return {
            "conversation_history": self.conversation_history,
            "tools": self.tools,
            "internal_tools": self.internal_tools,
            "client": self.client,
            "provider": self.provider,
            "model": self.model,
            "provider_config": self.provider_config,
            "server_info": self.server_info,
            "openai_tools": self.openai_tools,
            "exit_requested": self.exit_requested,
            "tool_to_server_map": self.tool_to_server_map,
            "stream_manager": self.stream_manager,
            "tool_manager": self.tool_manager,
        }

    def update_from_dict(self, context_dict: Dict[str, Any]) -> None:
        """
        Update the context after command handling.
        
        Only updates specified keys, allowing partial updates.
        """
        if "exit_requested" in context_dict:
            self.exit_requested = context_dict["exit_requested"]

        if "client" in context_dict and context_dict["client"] is not None:
            self.client = context_dict["client"]
            
        if "provider" in context_dict:
            self.provider = context_dict["provider"]
            
        if "model" in context_dict:
            self.model = context_dict["model"]
            
        if "provider_config" in context_dict and context_dict["provider_config"] is not None:
            self.provider_config = context_dict["provider_config"]

        if "stream_manager" in context_dict and context_dict["stream_manager"] is not None:
            self.stream_manager = context_dict["stream_manager"]

        if "tool_manager" in context_dict and context_dict["tool_manager"] is not None:
            self.tool_manager = context_dict["tool_manager"]

        if "tool_to_server_map" in context_dict:
            self.tool_to_server_map = context_dict["tool_to_server_map"]