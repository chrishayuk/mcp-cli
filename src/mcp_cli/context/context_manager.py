"""
Centralized context manager for MCP CLI.

This module provides a centralized way to manage application context
instead of passing dictionaries around.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
from pathlib import Path

from mcp_cli.tools.manager import ToolManager
from mcp_cli.model_manager import ModelManager
from mcp_cli.tools.models import ServerInfo, ToolInfo


@dataclass
class ApplicationContext:
    """
    Centralized application context that holds all state and managers.

    This replaces the dictionary-based context that was being passed around.
    """

    # Core managers
    tool_manager: Optional[ToolManager] = None
    model_manager: Optional[ModelManager] = None

    # Configuration
    config_path: Path = field(default_factory=lambda: Path("server_config.json"))
    provider: str = "openai"
    model: str = "gpt-4"
    api_base: Optional[str] = None
    api_key: Optional[str] = None

    # Server and tool state
    servers: List[ServerInfo] = field(default_factory=list)
    tools: List[ToolInfo] = field(default_factory=list)
    current_server: Optional[ServerInfo] = None

    # UI state
    verbose_mode: bool = True
    confirm_tools: bool = True
    theme: str = "default"

    # Token storage
    token_backend: Optional[str] = None

    # Session state
    session_id: Optional[str] = None
    is_interactive: bool = False
    exit_requested: bool = False

    # Conversation state (for chat mode)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)

    # Additional context data
    _extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize managers if not provided."""
        if self.model_manager is None:
            from mcp_cli.model_manager import ModelManager

            self.model_manager = ModelManager()

        # Set up model configuration
        if self.provider and self.model:
            self.model_manager.switch_model(self.provider, self.model)

    @classmethod
    def create(
        cls,
        tool_manager: Optional[ToolManager] = None,
        config_path: Optional[Path] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ) -> ApplicationContext:
        """
        Factory method to create a context with common defaults.
        """
        context = cls(
            tool_manager=tool_manager,
            config_path=config_path or Path("server_config.json"),
            provider=provider or "openai",
            model=model or "gpt-4",
            **kwargs,
        )
        return context

    async def initialize(self) -> None:
        """
        Initialize the context by loading servers, tools, etc.
        """
        if self.tool_manager:
            # Load servers
            self.servers = await self.tool_manager.get_server_info()

            # Load tools
            self.tools = await self.tool_manager.get_all_tools()

            # Set current server if only one
            if len(self.servers) == 1:
                self.current_server = self.servers[0]

    def get_current_server(self) -> Optional[ServerInfo]:
        """Get the currently active server."""
        return self.current_server

    def set_current_server(self, server: ServerInfo) -> None:
        """Set the currently active server."""
        self.current_server = server

    def find_server(self, name: str) -> Optional[ServerInfo]:
        """Find a server by name."""
        for server in self.servers:
            if server.name.lower() == name.lower():
                return server
        return None

    def find_tool(self, name: str) -> Optional[ToolInfo]:
        """Find a tool by name."""
        for tool in self.tools:
            if tool.name == name or tool.fully_qualified_name == name:
                return tool
        return None

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from context (for compatibility with dict-based code).

        This method provides backwards compatibility for code expecting dict access.
        """
        # Check direct attributes first
        if hasattr(self, key):
            return getattr(self, key)

        # Check extra data
        return self._extra.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a value in context (for compatibility with dict-based code).
        """
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self._extra[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for backwards compatibility.

        This is a transitional method while migrating from dict-based context.
        """
        return {
            "tool_manager": self.tool_manager,
            "model_manager": self.model_manager,
            "config_path": str(self.config_path),
            "provider": self.provider,
            "model": self.model,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "servers": self.servers,
            "tools": self.tools,
            "current_server": self.current_server,
            "verbose_mode": self.verbose_mode,
            "confirm_tools": self.confirm_tools,
            "theme": self.theme,
            "token_backend": self.token_backend,
            "session_id": self.session_id,
            "is_interactive": self.is_interactive,
            "exit_requested": self.exit_requested,
            "conversation_history": self.conversation_history,
            **self._extra,
        }

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Update context from a dictionary (for backwards compatibility).

        This is a transitional method while migrating from dict-based context.
        """
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self._extra[key] = value

    def update(self, **kwargs) -> None:
        """
        Update context with keyword arguments.

        This method provides a simple way to update multiple context attributes.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self._extra[key] = value


class ContextManager:
    """
    Manager for application contexts.

    This provides a singleton-like pattern for managing the application context.
    """

    _instance: Optional[ContextManager] = None
    _context: Optional[ApplicationContext] = None

    def __new__(cls) -> ContextManager:
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(
        self,
        tool_manager: Optional[ToolManager] = None,
        config_path: Optional[Path] = None,
        **kwargs,
    ) -> ApplicationContext:
        """
        Initialize or get the application context.
        """
        if self._context is None:
            self._context = ApplicationContext.create(
                tool_manager=tool_manager, config_path=config_path, **kwargs
            )
        return self._context

    def get_context(self) -> ApplicationContext:
        """
        Get the current application context.

        Raises:
            RuntimeError: If context hasn't been initialized
        """
        if self._context is None:
            raise RuntimeError("Context not initialized. Call initialize() first.")
        return self._context

    def reset(self) -> None:
        """Reset the context (useful for testing)."""
        self._context = None


def get_context() -> ApplicationContext:
    """
    Convenience function to get the current application context.

    Returns:
        The current ApplicationContext

    Raises:
        RuntimeError: If context hasn't been initialized
    """
    manager = ContextManager()
    return manager.get_context()


def initialize_context(**kwargs) -> ApplicationContext:
    """
    Convenience function to initialize the application context.

    Args:
        **kwargs: Arguments to pass to ApplicationContext.create()

    Returns:
        The initialized ApplicationContext
    """
    manager = ContextManager()
    return manager.initialize(**kwargs)
