# mcp_cli/chat/chat_context_v2.py
"""
Simplified chat context that leverages chuk-llm's ConversationContext.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, AsyncIterator, Optional

from chuk_llm.api.conversation import ConversationContext as ChukConversation
from chuk_llm import stream, ask
from chuk_term.ui import output

from mcp_cli.tools.manager import ToolManager
from mcp_cli.model_manager import ModelManager

logger = logging.getLogger(__name__)


class ChatContext:
    """
    Simplified chat context that delegates to chuk-llm's ConversationContext.
    
    This version removes redundant conversation management and leverages
    chuk-llm's built-in features for:
    - Conversation history management
    - Session tracking
    - System prompt handling
    - Streaming support
    """
    
    def __init__(self, tool_manager: ToolManager, model_manager: ModelManager):
        """Initialize with required managers."""
        self.tool_manager = tool_manager
        self.model_manager = model_manager
        
        # State
        self.exit_requested = False
        
        # Tool state (filled during initialization)
        self.tools: List[Dict[str, Any]] = []
        self.server_info: List[Dict[str, Any]] = []
        self.tool_to_server_map: Dict[str, str] = {}
        self.openai_tools: List[Dict[str, Any]] = []
        self.tool_name_mapping: Dict[str, str] = {}
        
        # Will be initialized in initialize()
        self.chuk_conversation: Optional[ChukConversation] = None
        
        logger.debug(f"ChatContext created with {self.provider}/{self.model}")
    
    @classmethod
    def create(
        cls,
        tool_manager: ToolManager,
        provider: str = None,
        model: str = None,
        api_base: str = None,
        api_key: str = None,
    ) -> "ChatContext":
        """Factory method for convenient creation."""
        model_manager = ModelManager()
        
        # Configure provider if API settings provided
        if provider and (api_base or api_key):
            model_manager.configure_provider(
                provider, api_key=api_key, api_base=api_base
            )
        
        # Switch model if requested
        if provider and model:
            model_manager.switch_model(provider, model)
        elif provider:
            model_manager.switch_provider(provider)
        elif model:
            model_manager.switch_to_model(model)
        
        return cls(tool_manager, model_manager)
    
    # Properties that delegate to ModelManager
    @property
    def provider(self) -> str:
        """Current provider name."""
        return self.model_manager.get_active_provider()
    
    @property
    def model(self) -> str:
        """Current model name."""
        return self.model_manager.get_active_model()
    
    @property
    def client(self) -> Any:
        """Get current LLM client."""
        return self.model_manager.get_client()
    
    # Initialization
    async def initialize(self) -> bool:
        """Initialize tools and conversation."""
        try:
            await self._initialize_tools()
            self._initialize_conversation()
            
            if not self.tools:
                output.print(
                    "[yellow]No tools available. Chat functionality may be limited.[/yellow]"
                )
            
            logger.info(
                f"ChatContext ready: {len(self.tools)} tools, {self.provider}/{self.model}"
            )
            return True
            
        except Exception as exc:
            logger.exception("Error initializing chat context")
            output.print(f"[red]Error initializing chat context: {exc}[/red]")
            return False
    
    async def _initialize_tools(self) -> None:
        """Initialize tool discovery and adaptation."""
        # Get tools from ToolManager
        tool_infos = await self.tool_manager.get_unique_tools()
        
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
        
        # Get server info
        raw_infos = await self.tool_manager.get_server_info()
        self.server_info = [
            {"id": s.id, "name": s.name, "tools": s.tool_count, "status": s.status}
            for s in raw_infos
        ]
        
        # Build tool-to-server mapping
        self.tool_to_server_map = {t["name"]: t["namespace"] for t in self.tools}
        
        # Adapt tools for current provider
        await self._adapt_tools_for_provider()
    
    async def _adapt_tools_for_provider(self) -> None:
        """Adapt tools for current provider."""
        try:
            if hasattr(self.tool_manager, "get_adapted_tools_for_llm"):
                tools_and_mapping = await self.tool_manager.get_adapted_tools_for_llm(
                    self.provider
                )
                self.openai_tools = tools_and_mapping[0]
                self.tool_name_mapping = tools_and_mapping[1]
                logger.debug(
                    f"Adapted {len(self.openai_tools)} tools for {self.provider}"
                )
            else:
                # Fallback to generic tools
                self.openai_tools = await self.tool_manager.get_tools_for_llm()
                self.tool_name_mapping = {}
        except Exception as exc:
            logger.warning(f"Error adapting tools: {exc}")
            from mcp_cli.tools.manager import ToolManager
            self.openai_tools = ToolManager.convert_to_openai_tools(self.tools)
            self.tool_name_mapping = {}
    
    def _initialize_conversation(self) -> None:
        """Initialize chuk-llm conversation context."""
        # Create chuk-llm conversation with automatic session tracking
        self.chuk_conversation = ChukConversation(
            provider=self.provider,
            model=self.model,
            # System prompt will be auto-generated by chuk-llm
            infinite_context=True,  # Enable infinite context
            token_threshold=4000,    # Reasonable threshold
        )
        
        logger.debug("Initialized chuk-llm ConversationContext")
    
    # Simplified conversation management using chuk-llm
    async def ask_with_tools(self, prompt: str) -> Dict[str, Any]:
        """
        Ask a question with tool support using chuk-llm's native capabilities.
        
        Returns:
            Dict with 'response' and optionally 'tool_calls'
        """
        if not self.chuk_conversation:
            raise RuntimeError("Conversation not initialized")
        
        # Use chuk-llm's ask with tools
        result = await self.chuk_conversation.ask(
            prompt,
            tools=self.openai_tools if self.openai_tools else None
        )
        
        return result
    
    async def stream_with_tools(self, prompt: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream a response with tool support using chuk-llm's native streaming.
        
        Yields:
            Stream chunks with content and/or tool calls
        """
        if not self.chuk_conversation:
            raise RuntimeError("Conversation not initialized")
        
        # Use chuk-llm's streaming with tools
        async for chunk in self.chuk_conversation.stream(
            prompt,
            tools=self.openai_tools if self.openai_tools else None
        ):
            yield chunk
    
    # Tool execution (delegate to ToolManager which uses chuk-tool-processor)
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool through the tool manager (using chuk-tool-processor)."""
        # ToolManager already uses chuk-tool-processor internally
        return await self.tool_manager.execute_tool(tool_name, arguments)
    
    async def stream_execute_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> AsyncIterator[Any]:
        """Execute a tool with streaming through chuk-tool-processor."""
        async for result in self.tool_manager.stream_execute_tool(tool_name, arguments):
            yield result
    
    async def execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Any]:
        """
        Execute multiple tool calls from chuk-llm response.
        
        Args:
            tool_calls: List of tool call dictionaries from chuk-llm
        
        Returns:
            List of tool execution results
        """
        results = []
        for tc in tool_calls:
            func = tc.get("function", {})
            tool_name = func.get("name")
            args_str = func.get("arguments", "{}")
            
            # Parse arguments
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except json.JSONDecodeError:
                args = {}
            
            # Map tool name if needed
            actual_tool_name = self.tool_name_mapping.get(tool_name, tool_name)
            
            # Execute through tool manager (which uses chuk-tool-processor)
            result = await self.execute_tool(actual_tool_name, args)
            results.append(result)
        
        return results
    
    async def get_server_for_tool(self, tool_name: str) -> str:
        """Get server for tool."""
        return await self.tool_manager.get_server_for_tool(tool_name) or "Unknown"
    
    # Conversation management (simplified - delegates to chuk-llm)
    def add_user_message(self, content: str) -> None:
        """Add user message - handled by chuk-llm conversation."""
        # chuk-llm handles this internally when we call ask/stream
        pass
    
    def add_assistant_message(self, content: str) -> None:
        """Add assistant message - handled by chuk-llm conversation."""
        # chuk-llm handles this internally
        pass
    
    def get_conversation_length(self) -> int:
        """Get conversation length."""
        if self.chuk_conversation:
            return len(self.chuk_conversation.messages) - 1  # Exclude system prompt
        return 0
    
    def clear_conversation_history(self, keep_system_prompt: bool = True) -> None:
        """Clear conversation history."""
        if self.chuk_conversation:
            if keep_system_prompt:
                # Keep only system message
                self.chuk_conversation.messages = self.chuk_conversation.messages[:1]
            else:
                self.chuk_conversation.messages.clear()
    
    # Model change handling
    async def refresh_after_model_change(self) -> None:
        """Refresh context after model change."""
        await self._adapt_tools_for_provider()
        
        # Re-create conversation with new model
        self._initialize_conversation()
        
        logger.debug(f"ChatContext refreshed for {self.provider}/{self.model}")
    
    # Simple getters
    def get_tool_count(self) -> int:
        """Get number of available tools."""
        return len(self.tools)
    
    def get_server_count(self) -> int:
        """Get number of connected servers."""
        return len(self.server_info)
    
    @staticmethod
    def get_display_name_for_tool(namespaced_tool_name: str) -> str:
        """Get display name for tool."""
        return namespaced_tool_name
    
    # Compatibility methods for existing code
    @property
    def conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history for compatibility."""
        if self.chuk_conversation:
            return self.chuk_conversation.messages
        return []
    
    @conversation_history.setter
    def conversation_history(self, value: List[Dict[str, Any]]) -> None:
        """Set conversation history for compatibility."""
        if self.chuk_conversation:
            self.chuk_conversation.messages = value
    
    # Status and debug
    def get_status_summary(self) -> Dict[str, Any]:
        """Get status summary for debugging."""
        return {
            "provider": self.provider,
            "model": self.model,
            "tool_count": len(self.tools),
            "server_count": len(self.server_info),
            "conversation_length": self.get_conversation_length(),
            "tools_adapted": bool(self.openai_tools),
            "exit_requested": self.exit_requested,
            "has_session": self.chuk_conversation.has_session if self.chuk_conversation else False,
        }
    
    def __repr__(self) -> str:
        return (
            f"ChatContext(provider='{self.provider}', model='{self.model}', "
            f"tools={len(self.tools)}, messages={self.get_conversation_length()})"
        )