# src/mcp_cli/commands/enums.py
"""
Enums and constants for command actions.

Centralized enums to replace hardcoded strings throughout the codebase.
"""

from __future__ import annotations

from enum import Enum


class CommandAction(str, Enum):
    """Common command actions across different command types."""

    LIST = "list"
    ADD = "add"
    REMOVE = "remove"
    SET = "set"
    GET = "get"
    DELETE = "delete"
    CREATE = "create"
    UPDATE = "update"
    SHOW = "show"
    ENABLE = "enable"
    DISABLE = "disable"
    VALIDATE = "validate"
    STATUS = "status"
    DETAILS = "details"
    REFRESH = "refresh"
    CONFIG = "config"
    DIAGNOSTIC = "diagnostic"


class TokenNamespace(str, Enum):
    """Token storage namespaces."""

    GENERIC = "generic"
    PROVIDER = "provider"
    BEARER = "bearer"
    API_KEY = "api-key"
    OAUTH = "oauth"


class TransportType(str, Enum):
    """Server transport types."""

    STDIO = "stdio"
    SSE = "sse"
    HTTP = "http"


class OutputFormat(str, Enum):
    """Output format types."""

    JSON = "json"
    TABLE = "table"
    TEXT = "text"


class ProviderCommand(str, Enum):
    """Provider-specific commands."""

    LIST = "list"
    ADD = "add"
    REMOVE = "remove"
    SET = "set"
    CONFIG = "config"
    DIAGNOSTIC = "diagnostic"
    CUSTOM = "custom"


class TokenSource(str, Enum):
    """Where a token comes from."""

    ENV = "env"
    STORAGE = "storage"
    NONE = "none"


class ServerCommand(str, Enum):
    """Server management commands."""

    LIST = "list"
    ADD = "add"
    REMOVE = "remove"
    ENABLE = "enable"
    DISABLE = "disable"
    STATUS = "status"


class ToolCommand(str, Enum):
    """Tool management commands."""

    LIST = "list"
    DETAILS = "details"
    VALIDATE = "validate"
    STATUS = "status"
    ENABLE = "enable"
    DISABLE = "disable"
    LIST_DISABLED = "list-disabled"
    AUTO_FIX = "auto-fix"
    CLEAR_VALIDATION = "clear-validation"
    VALIDATION_ERRORS = "validation-errors"


# Constants for special values
class SpecialValues:
    """Special values used across commands."""

    STDIN = "-"
    STDOUT = "-"
    OLLAMA = "ollama"
    USER = "User"


# Error messages
class ErrorMessages:
    """Common error messages."""

    NO_CONTEXT = "Context not initialized"
    NO_TOOL_MANAGER = "Tool manager not available"
    NO_MODEL_MANAGER = "Model manager not available"
    NO_SERVER_MANAGER = "Server manager not available"
    INVALID_PROVIDER = "Unknown provider"
    INVALID_SERVER = "Unknown server"
    INVALID_TOOL = "Unknown tool"
    NO_OPERATION = "No operation specified"


# Success messages
class SuccessMessages:
    """Common success messages."""

    TOKEN_STORED = "Token stored successfully"
    TOKEN_DELETED = "Token deleted successfully"
    SERVER_ADDED = "Server added successfully"
    SERVER_REMOVED = "Server removed successfully"
    PROVIDER_SWITCHED = "Provider switched successfully"
