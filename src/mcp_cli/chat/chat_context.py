# mcp_cli/chat/chat_context_v2.py
"""
Simplified chat context that leverages chuk-llm's ConversationContext.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, AsyncIterator, Optional

from chuk_llm.api.conversation import ConversationContext as ChukConversation
from chuk_term.ui import output

from mcp_cli.tools.manager import ToolManager
from mcp_cli.model_manager import ModelManager
from mcp_cli.chat.system_prompt import generate_system_prompt

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
        self.internal_tools: List[Dict[str, Any]] = []  # For compatibility
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
        # Generate system prompt
        system_prompt = generate_system_prompt(self.tools)

        # Create chuk-llm conversation with automatic session tracking
        self.chuk_conversation = ChukConversation(
            provider=self.provider,
            model=self.model,
            system_prompt=system_prompt,
            infinite_context=True,  # Enable infinite context
            token_threshold=4000,  # Reasonable threshold
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
            prompt, tools=self.openai_tools if self.openai_tools else None
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

        # Debug: log tools being passed
        tools_to_pass = self.openai_tools if self.openai_tools else None
        logger.debug(f"Streaming with {len(tools_to_pass) if tools_to_pass else 0} tools")
        if tools_to_pass:
            logger.debug(f"Tool names: {[t.get('function', {}).get('name') for t in tools_to_pass[:3]]}")

        # Use chuk-llm's streaming with tools
        async for chunk in self.chuk_conversation.stream(
            prompt, tools=tools_to_pass
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
        """Add user message to conversation history."""
        if self.chuk_conversation:
            # Add directly to the messages list
            self.chuk_conversation.messages.append({"role": "user", "content": content})
        else:
            # Fallback if conversation not initialized
            if not hasattr(self, '_fallback_history'):
                self._fallback_history = []
            self._fallback_history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add assistant message to conversation history."""
        if self.chuk_conversation:
            # Add directly to the messages list
            self.chuk_conversation.messages.append({"role": "assistant", "content": content})
        else:
            # Fallback if conversation not initialized
            if not hasattr(self, '_fallback_history'):
                self._fallback_history = []
            self._fallback_history.append({"role": "assistant", "content": content})

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
            "has_session": self.chuk_conversation.has_session
            if self.chuk_conversation
            else False,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export context for command handlers."""
        return {
            "conversation_history": self.conversation_history,
            "tools": self.tools,
            "internal_tools": self.internal_tools,
            "client": None,  # v2 doesn't expose raw client
            "provider": self.provider,
            "model": self.model,
            "model_manager": self.model_manager,
            "server_info": self.server_info,
            "openai_tools": self.openai_tools,
            "tool_name_mapping": getattr(self, "tool_name_mapping", {}),
            "tool_manager": self.tool_manager,
            "exit_requested": self.exit_requested,
        }

    def update_from_dict(self, context_dict: Dict[str, Any]) -> None:
        """Update context from dictionary (simplified)."""
        # Core state updates
        if "exit_requested" in context_dict:
            self.exit_requested = context_dict["exit_requested"]

        if "conversation_history" in context_dict:
            self.conversation_history = context_dict["conversation_history"]

        if "model_manager" in context_dict:
            self.model_manager = context_dict["model_manager"]
            # Provider and model are read-only properties that delegate to model_manager
            # No need to update them directly

    def __repr__(self) -> str:
        return (
            f"ChatContext(provider='{self.provider}', model='{self.model}', "
            f"tools={len(self.tools)}, messages={self.get_conversation_length()})"
        )


class TestChatContext(ChatContext):
    """
    Test-specific ChatContext that works with stream_manager instead of ToolManager.

    Separated from main ChatContext to keep it clean.
    """

    def __init__(self, stream_manager: Any, model_manager: ModelManager):
        """Create test context with stream_manager."""
        # Initialize base attributes without calling super().__init__
        self.tool_manager = None  # Tests don't use ToolManager
        self.stream_manager = stream_manager
        self.model_manager = model_manager

        # Conversation state
        self.exit_requested = False
        self._conversation_history = []

        # Tool state
        self.tools = []
        self.internal_tools = []
        self.openai_tools = []
        self.server_info = []

        # Model info
        self.provider = model_manager.active_provider
        self.model = model_manager.active_model

        # Initialize conversation
        self.chuk_conversation = None
        # Don't initialize conversation in test context

    @property
    def conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self._conversation_history

    @conversation_history.setter
    def conversation_history(self, value: List[Dict[str, Any]]) -> None:
        """Set conversation history."""
        self._conversation_history = value

    def _initialize_conversation(self) -> None:
        """Initialize test conversation with mocked system prompt."""
        # For testing, use a simple system prompt
        system_prompt = generate_system_prompt(self.tools)
        self._conversation_history = [{"role": "system", "content": system_prompt}]

    async def initialize(self) -> bool:
        """Initialize test context."""
        logger.debug("Initializing TestChatContext")
        self._initialize_conversation()
        return True

    @classmethod
    def create_for_testing(
        cls,
        stream_manager: Any,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> "TestChatContext":
        """Factory method for creating test context."""
        # Create ModelManager with overrides
        mm = ModelManager()

        if provider:
            mm.active_provider = provider
        if model:
            mm.active_model = model

        return cls(stream_manager=stream_manager, model_manager=mm)
