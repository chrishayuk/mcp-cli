"""Additional enums for type safety - no more magic strings!"""

from enum import Enum


class ServerStatus(str, Enum):
    """Server status values."""

    CONFIGURED = "configured"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


class ConversationAction(str, Enum):
    """Actions for /conversation command."""

    SHOW = "show"
    CLEAR = "clear"
    SAVE = "save"
    LOAD = "load"


class TokenAction(str, Enum):
    """Actions for /token command."""

    LIST = "list"
    SET = "set"
    GET = "get"
    DELETE = "delete"
    CLEAR = "clear"
    BACKENDS = "backends"


class ServerAction(str, Enum):
    """Actions for /server command."""

    ENABLE = "enable"
    DISABLE = "disable"
    STATUS = "status"
    INFO = "info"


class ToolAction(str, Enum):
    """Actions for /tools command."""

    LIST = "list"
    ENABLE = "enable"
    DISABLE = "disable"
    CONFIRM = "confirm"
    INFO = "info"


class ThemeAction(str, Enum):
    """Actions for /theme command."""

    SET = "set"
    LIST = "list"
    SHOW = "show"


__all__ = [
    "ServerStatus",
    "ConversationAction",
    "TokenAction",
    "ServerAction",
    "ToolAction",
    "ThemeAction",
]
