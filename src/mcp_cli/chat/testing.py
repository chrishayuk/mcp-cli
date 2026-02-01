# mcp_cli/chat/testing.py
"""Testing utilities for chat module.

This module contains test helpers that are used by both production test-mode
code paths and unit tests. These are separated from the main chat_context.py
to keep production code clean.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp_cli.chat.chat_context import ChatContext
from mcp_cli.model_management import ModelManager

logger = logging.getLogger(__name__)


class TestChatContext(ChatContext):
    """
    Test-specific ChatContext that works with stream_manager instead of ToolManager.

    This is used for:
    1. The --test-mode flag in chat handler
    2. Unit tests that need a mock context
    """

    def __init__(self, stream_manager: Any, model_manager: ModelManager):
        """Create test context with stream_manager."""
        # Initialize base attributes without calling super().__init__
        self.tool_manager = None  # type: ignore[assignment]  # Tests don't use ToolManager
        self.stream_manager = stream_manager
        self.model_manager = model_manager

        # Conversation state
        self.exit_requested = False
        self.conversation_history: list = []
        self.tool_history: list = []

        # ToolProcessor back-reference
        self.tool_processor: Any = None

        # Tool state
        self.tools: list = []
        self.internal_tools: list = []
        self.server_info: list = []
        self.tool_to_server_map: dict = {}
        self.openai_tools: list = []
        self.tool_name_mapping: dict = {}

        logger.debug(f"TestChatContext created with {self.provider}/{self.model}")

    @classmethod
    def create_for_testing(
        cls,
        stream_manager: Any,
        provider: str | None = None,
        model: str | None = None,
    ) -> "TestChatContext":
        """Factory for test contexts."""
        model_manager = ModelManager()

        if provider and model:
            model_manager.switch_model(provider, model)
        elif provider:
            model_manager.switch_provider(provider)
        elif model:
            # Switch model in current provider
            current_provider = model_manager.get_active_provider()
            model_manager.switch_model(current_provider, model)

        return cls(stream_manager, model_manager)

    async def _initialize_tools(self) -> None:
        """Test-specific tool initialization."""
        # Get tools from stream_manager
        if hasattr(self.stream_manager, "get_internal_tools"):
            self.tools = list(self.stream_manager.get_internal_tools())
        else:
            self.tools = list(self.stream_manager.get_all_tools())

        # Get server info
        self.server_info = list(self.stream_manager.get_server_info())

        # Build mappings - tools are ToolInfo objects
        self.tool_to_server_map = {
            t.name: self.stream_manager.get_server_for_tool(t.name) for t in self.tools
        }

        # Convert tools to OpenAI format for tests
        self.openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.parameters or {},
                },
            }
            for t in self.tools
        ]
        self.tool_name_mapping = {}

        # Copy for system prompt
        self.internal_tools = list(self.tools)

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Execute tool via stream_manager."""
        if hasattr(self.stream_manager, "call_tool"):
            return await self.stream_manager.call_tool(tool_name, arguments)
        else:
            raise ValueError("Stream manager doesn't support tool execution")

    async def get_server_for_tool(self, tool_name: str) -> str:
        """Get server for tool from stream_manager."""
        return self.stream_manager.get_server_for_tool(tool_name) or "Unknown"
