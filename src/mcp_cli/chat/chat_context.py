# mcp_cli/chat/chat_context.py
"""
Chat context using chuk-ai-session-manager as the native conversation backend.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, AsyncIterator

from chuk_term.ui import output

from mcp_cli.chat.system_prompt import generate_system_prompt
from mcp_cli.chat.models import Message, MessageRole, ToolExecutionRecord, ChatStatus
from mcp_cli.tools.manager import ToolManager
from mcp_cli.tools.models import ToolInfo, ServerInfo
from mcp_cli.model_management import ModelManager

# Native session management from chuk-ai-session-manager
from chuk_ai_session_manager import SessionManager
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.procedural_memory import (
    ToolMemoryManager,
    ToolOutcome,
    ProceduralContextFormatter,
    FormatterConfig,
)

logger = logging.getLogger(__name__)


class ChatContext:
    """
    Chat context with SessionManager as the native conversation backend.

    SessionManager is required - no fallback to local state.
    All conversation tracking flows through chuk-ai-session-manager.

    Responsibilities:
    - Conversation history management (via SessionManager)
    - Tool discovery and adaptation coordination
    - Procedural memory for tool learning
    - Session state (exit requests, etc.)
    """

    def __init__(
        self,
        tool_manager: ToolManager,
        model_manager: ModelManager,
        session_id: str | None = None,
    ):
        """
        Create chat context with required managers.

        Args:
            tool_manager: Tool management interface
            model_manager: Model configuration and LLM client manager
            session_id: Optional session ID for conversation tracking
        """
        self.tool_manager = tool_manager
        self.model_manager = model_manager
        self.session_id = session_id or self._generate_session_id()

        # Core session manager - always required
        self.session: SessionManager = SessionManager(session_id=self.session_id)
        self._system_prompt: str = ""

        # Procedural memory for tool learning
        self.tool_memory = ToolMemoryManager.create(session_id=self.session_id)
        self.procedural_formatter = ProceduralContextFormatter(
            config=FormatterConfig(
                max_recent_calls=5,
                max_errors_per_tool=3,
                max_successes_per_tool=2,
                include_fix_suggestions=True,
            )
        )

        # Session state
        self.exit_requested = False
        self.tool_history: list[ToolExecutionRecord] = []

        # ToolProcessor back-reference (set by ToolProcessor.__init__)
        self.tool_processor: Any = None

        # Tool state (filled during initialization)
        self.tools: list[ToolInfo] = []
        self.internal_tools: list[ToolInfo] = []
        self.server_info: list[ServerInfo] = []
        self.tool_to_server_map: dict[str, str] = {}
        self.openai_tools: list[dict[str, Any]] = []
        self.tool_name_mapping: dict[str, str] = {}

        logger.debug(f"ChatContext created with {self.provider}/{self.model}")

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a unique session ID."""
        return f"chat-{uuid.uuid4().hex[:12]}"

    @classmethod
    def create(
        cls,
        tool_manager: ToolManager,
        provider: str | None = None,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        model_manager: ModelManager | None = None,
        session_id: str | None = None,
    ) -> "ChatContext":
        """
        Factory method for convenient creation.

        Args:
            tool_manager: Tool management interface
            provider: Provider to switch to (optional)
            model: Model to switch to (optional)
            api_base: API base URL override (optional)
            api_key: API key override (optional)
            model_manager: Pre-configured ModelManager (optional, creates new if None)
            session_id: Session ID for conversation tracking (optional)

        Returns:
            Configured ChatContext instance
        """
        if model_manager is None:
            model_manager = ModelManager()

            if provider and (api_base or api_key):
                model_manager.add_runtime_provider(
                    name=provider, api_key=api_key, api_base=api_base or ""
                )

            if provider and model:
                model_manager.switch_model(provider, model)
            elif provider:
                model_manager.switch_provider(provider)
            elif model:
                current_provider = model_manager.get_active_provider()
                model_manager.switch_model(current_provider, model)

        return cls(tool_manager, model_manager, session_id)

    # ── Properties ────────────────────────────────────────────────────────
    @property
    def client(self) -> Any:
        """Get current LLM client (cached automatically by ModelManager)."""
        return self.model_manager.get_client()

    @property
    def provider(self) -> str:
        """Current provider name."""
        return self.model_manager.get_active_provider()

    @property
    def model(self) -> str:
        """Current model name."""
        return self.model_manager.get_active_model()

    @property
    def conversation_history(self) -> list[Message]:
        """
        Get conversation history as list of Message objects.

        Provides backwards compatibility while using SessionManager internally.
        Handles both regular messages and tool-related messages.
        """
        messages = []

        # System prompt first
        if self._system_prompt:
            messages.append(
                Message(role=MessageRole.SYSTEM, content=self._system_prompt)
            )

        # Get events from session
        if self.session._session:
            for event in self.session._session.events:
                if event.type == EventType.MESSAGE:
                    if event.source == EventSource.USER:
                        messages.append(
                            Message(role=MessageRole.USER, content=str(event.message))
                        )
                    elif event.source in (EventSource.LLM, EventSource.SYSTEM):
                        messages.append(
                            Message(
                                role=MessageRole.ASSISTANT, content=str(event.message)
                            )
                        )
                elif event.type == EventType.TOOL_CALL:
                    # Tool messages stored as dict - reconstruct Message
                    if isinstance(event.message, dict):
                        messages.append(Message.from_dict(event.message))

        return messages

    # ── Initialization ────────────────────────────────────────────────────
    async def initialize(self) -> bool:
        """Initialize tools, session, and procedural memory."""
        try:
            await self._initialize_tools()
            self._generate_system_prompt()
            await self._initialize_session()

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

    async def _initialize_session(self) -> None:
        """Initialize the session with system prompt."""
        self.session = SessionManager(
            session_id=self.session_id,
            system_prompt=self._system_prompt,
            infinite_context=False,
        )
        await self.session._ensure_initialized()
        logger.debug(f"Session initialized: {self.session_id}")

    def _generate_system_prompt(self) -> None:
        """Generate system prompt from available tools."""
        tools_for_prompt = [
            tool.to_llm_format().to_dict() for tool in self.internal_tools
        ]
        self._system_prompt = generate_system_prompt(tools_for_prompt)

    async def _initialize_tools(self) -> None:
        """Initialize tool discovery and adaptation."""
        self.tools = await self.tool_manager.get_unique_tools()
        logger.debug(f"ChatContext: Initialized with {len(self.tools)} tools")

        self.server_info = await self.tool_manager.get_server_info()
        self.tool_to_server_map = {t.name: t.namespace for t in self.tools}

        await self._adapt_tools_for_provider()
        self.internal_tools = list(self.tools)

    def find_tool_by_name(self, name: str) -> ToolInfo | None:
        """Find a tool by its name (handles both simple and namespaced names)."""
        for tool in self.tools:
            if tool.name == name or tool.fully_qualified_name == name:
                return tool

        simple_name = name.split(".")[-1] if "." in name else name
        for tool in self.tools:
            if tool.name == simple_name:
                return tool

        return None

    def find_server_by_name(self, name: str) -> ServerInfo | None:
        """Find a server by its name."""
        for server in self.server_info:
            if server.name == name or server.namespace == name:
                return server
        return None

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
                self.openai_tools = await self.tool_manager.get_tools_for_llm()
                self.tool_name_mapping = {}
        except Exception as exc:
            logger.warning(f"Error adapting tools: {exc}")
            self.openai_tools = await self.tool_manager.get_tools_for_llm()
            self.tool_name_mapping = {}

    # ── Model change handling ─────────────────────────────────────────────
    async def refresh_after_model_change(self) -> None:
        """Refresh context after ModelManager changes the model."""
        await self._adapt_tools_for_provider()
        logger.debug(f"ChatContext refreshed for {self.provider}/{self.model}")

    # ── Tool execution (delegate to ToolManager) ──────────────────────────
    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool."""
        return await self.tool_manager.execute_tool(tool_name, arguments)

    async def stream_execute_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> AsyncIterator[Any]:
        """Execute a tool with streaming."""
        async for result in self.tool_manager.stream_execute_tool(tool_name, arguments):
            yield result

    async def get_server_for_tool(self, tool_name: str) -> str:
        """Get server for tool."""
        return await self.tool_manager.get_server_for_tool(tool_name) or "Unknown"

    # ── Conversation management ───────────────────────────────────────────
    async def add_user_message(self, content: str) -> None:
        """Add user message to conversation."""
        await self.session.user_says(content)
        logger.debug(f"User message added: {content[:50]}...")

    async def add_assistant_message(self, content: str) -> None:
        """Add assistant message to conversation."""
        await self.session.ai_responds(
            content, model=self.model, provider=self.provider
        )
        logger.debug(f"Assistant message added: {content[:50]}...")

    def inject_assistant_message(self, content: str) -> None:
        """
        Inject a synthetic assistant message for conversation flow control.

        Use this for system-generated messages (budget exhaustion, state summaries,
        error recovery) that guide the model but aren't true AI responses.
        """
        event = SessionEvent(
            message=content,
            source=EventSource.SYSTEM,
            type=EventType.MESSAGE,
        )
        self.session._session.events.append(event)
        logger.debug(f"Injected assistant message: {content[:50]}...")

    def inject_tool_message(self, message: Message) -> None:
        """
        Inject a tool-related message into conversation history.

        Tool messages (assistant with tool_calls, tool results) have special structure
        that doesn't map to SessionManager events. These are stored as raw events
        for conversation flow but tracked separately in procedural memory.
        """
        # Store as TOOL_CALL event with the full message structure
        event = SessionEvent(
            message=message.to_dict(),
            source=EventSource.SYSTEM,
            type=EventType.TOOL_CALL,
        )
        self.session._session.events.append(event)
        logger.debug(f"Injected tool message: role={message.role}")

    async def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
        success: bool = True,
        error: str | None = None,
        context_goal: str | None = None,
    ) -> None:
        """
        Record a tool call in session and procedural memory.

        Args:
            tool_name: Name of the tool called
            arguments: Arguments passed to the tool
            result: Result returned by the tool
            success: Whether the call succeeded
            error: Error message if failed
            context_goal: What the user was trying to accomplish
        """
        # Record in session
        await self.session.tool_used(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            error=error,
        )

        # Record in procedural memory for learning
        outcome = ToolOutcome.SUCCESS if success else ToolOutcome.FAILURE
        error_type = (
            type(error).__name__ if error and not isinstance(error, str) else None
        )

        await self.tool_memory.record_call(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            outcome=outcome,
            context_goal=context_goal,
            error_type=error_type,
            error_message=str(error) if error else None,
        )

        logger.debug(f"Tool call recorded: {tool_name} (success={success})")

    async def get_messages_for_llm(self) -> list[dict[str, str]]:
        """Get messages formatted for LLM API calls."""
        result: list[dict[str, str]] = await self.session.get_messages_for_llm(
            include_system=True
        )
        return result

    def get_conversation_length(self) -> int:
        """Get conversation length (excluding system prompt)."""
        if self.session._session:
            return sum(
                1 for e in self.session._session.events if e.type == EventType.MESSAGE
            )
        return 0

    async def clear_conversation_history(self, keep_system_prompt: bool = True) -> None:
        """Clear conversation history by creating a new session."""
        self.session_id = self._generate_session_id()
        await self._initialize_session()
        self.tool_memory = ToolMemoryManager.create(session_id=self.session_id)
        logger.debug(f"Conversation cleared, new session: {self.session_id}")

    async def regenerate_system_prompt(self) -> None:
        """Regenerate system prompt with current tools."""
        self._generate_system_prompt()
        await self.session.update_system_prompt(self._system_prompt)

    # ── Procedural memory helpers ─────────────────────────────────────────
    def get_procedural_context_for_tools(
        self, tool_names: list[str], context_goal: str | None = None
    ) -> str:
        """
        Get procedural memory context for tools about to be called.

        Use this to inject relevant tool history before making LLM calls.
        """
        result: str = self.procedural_formatter.format_for_tools(
            self.tool_memory, tool_names, context_goal
        )
        return result

    def get_recent_tool_history(self, limit: int = 5) -> list[dict[str, Any]]:
        """Get recent tool call history from procedural memory."""
        return [
            {
                "tool": entry.tool_name,
                "arguments": entry.arguments,
                "outcome": entry.outcome.value,
                "timestamp": entry.timestamp.isoformat(),
            }
            for entry in self.tool_memory.memory.tool_log[-limit:]
        ]

    # ── Simple getters ────────────────────────────────────────────────────
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

    # ── Serialization ─────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        """Export context for command handlers."""
        return {
            "conversation_history": [
                msg.to_dict() for msg in self.conversation_history
            ],
            "tools": self.tools,
            "internal_tools": self.internal_tools,
            "client": self.client,
            "provider": self.provider,
            "model": self.model,
            "model_manager": self.model_manager,
            "server_info": self.server_info,
            "openai_tools": self.openai_tools,
            "tool_name_mapping": self.tool_name_mapping,
            "exit_requested": self.exit_requested,
            "tool_to_server_map": self.tool_to_server_map,
            "tool_manager": self.tool_manager,
            "session_id": self.session_id,
        }

    def update_from_dict(self, context_dict: dict[str, Any]) -> None:
        """Update context from dictionary."""
        if "exit_requested" in context_dict:
            self.exit_requested = context_dict["exit_requested"]

        if "model_manager" in context_dict:
            self.model_manager = context_dict["model_manager"]

        for key in [
            "tools",
            "internal_tools",
            "server_info",
            "tool_to_server_map",
            "openai_tools",
            "tool_name_mapping",
        ]:
            if key in context_dict:
                setattr(self, key, context_dict[key])

    # ── Context manager ───────────────────────────────────────────────────
    async def __aenter__(self):
        """Async context manager entry."""
        if not await self.initialize():
            raise RuntimeError("Failed to initialize ChatContext")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass

    # ── Debug info ────────────────────────────────────────────────────────
    def get_status_summary(self) -> ChatStatus:
        """Get status summary for debugging."""
        return ChatStatus(
            provider=self.provider,
            model=self.model,
            tool_count=len(self.tools),
            internal_tool_count=len(self.internal_tools),
            server_count=len(self.server_info),
            message_count=self.get_conversation_length(),
            tool_execution_count=len(self.tool_history),
        )

    async def get_session_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        result: dict[str, Any] = await self.session.get_stats()
        return result

    def __repr__(self) -> str:
        return (
            f"ChatContext(session='{self.session_id}', provider='{self.provider}', "
            f"model='{self.model}', tools={len(self.tools)}, messages={self.get_conversation_length()})"
        )

    def __str__(self) -> str:
        return f"Chat session {self.session_id} with {self.provider}/{self.model} ({len(self.tools)} tools)"
