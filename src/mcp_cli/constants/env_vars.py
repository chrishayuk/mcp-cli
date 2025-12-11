"""Environment variable names - centralized, type-safe, no magic strings!

All environment variable access should go through this module.
"""

from __future__ import annotations

import os
from enum import Enum


class EnvVar(str, Enum):
    """All environment variable names used by MCP CLI.

    Use these instead of hardcoded strings for type safety.
    """

    # ================================================================
    # Timeout Configuration
    # ================================================================
    TOOL_TIMEOUT = "MCP_TOOL_TIMEOUT"
    STREAMING_CHUNK_TIMEOUT = "MCP_STREAMING_CHUNK_TIMEOUT"
    STREAMING_GLOBAL_TIMEOUT = "MCP_STREAMING_GLOBAL_TIMEOUT"
    STREAMING_FIRST_CHUNK_TIMEOUT = "MCP_STREAMING_FIRST_CHUNK_TIMEOUT"
    TOOL_EXECUTION_TIMEOUT = "MCP_TOOL_EXECUTION_TIMEOUT"
    SERVER_INIT_TIMEOUT = "MCP_SERVER_INIT_TIMEOUT"

    # ================================================================
    # Tool Configuration
    # ================================================================
    CLI_INCLUDE_TOOLS = "MCP_CLI_INCLUDE_TOOLS"
    CLI_EXCLUDE_TOOLS = "MCP_CLI_EXCLUDE_TOOLS"
    CLI_DYNAMIC_TOOLS = "MCP_CLI_DYNAMIC_TOOLS"
    CLI_MODIFIED_CONFIG = "MCP_CLI_MODIFIED_CONFIG"

    # ================================================================
    # Token/Auth Configuration
    # ================================================================
    CLI_TOKEN_BACKEND = "MCP_CLI_TOKEN_BACKEND"

    # ================================================================
    # LLM Provider Configuration
    # ================================================================
    LLM_PROVIDER = "LLM_PROVIDER"
    LLM_MODEL = "LLM_MODEL"
    PROVIDER = "MCP_PROVIDER"
    MODEL = "MCP_MODEL"

    # ================================================================
    # Paths and Filesystem
    # ================================================================
    SOURCE_FILESYSTEMS = "SOURCE_FILESYSTEMS"

    # ================================================================
    # ChukLLM Discovery Configuration
    # ================================================================
    CHUK_LLM_DISCOVERY_ENABLED = "CHUK_LLM_DISCOVERY_ENABLED"
    CHUK_LLM_OLLAMA_DISCOVERY = "CHUK_LLM_OLLAMA_DISCOVERY"
    CHUK_LLM_AUTO_DISCOVER = "CHUK_LLM_AUTO_DISCOVER"
    CHUK_LLM_OPENAI_TOOL_COMPATIBILITY = "CHUK_LLM_OPENAI_TOOL_COMPATIBILITY"
    CHUK_LLM_UNIVERSAL_TOOLS = "CHUK_LLM_UNIVERSAL_TOOLS"
    CHUK_LLM_DISCOVERY_FORCE_REFRESH = "CHUK_LLM_DISCOVERY_FORCE_REFRESH"

    # ================================================================
    # System (typically inherited, not set by MCP CLI)
    # ================================================================
    PATH = "PATH"
    HOME = "HOME"
    USER = "USER"


# ================================================================
# Type-Safe Helper Functions
# ================================================================


def get_env(var: EnvVar, default: str | None = None) -> str | None:
    """Get environment variable value (type-safe).

    Args:
        var: EnvVar enum member
        default: Default value if not set

    Returns:
        Environment variable value or default

    Example:
        >>> timeout = get_env(EnvVar.TOOL_TIMEOUT, "120")
    """
    return os.getenv(var.value, default)


def set_env(var: EnvVar, value: str) -> None:
    """Set environment variable (type-safe).

    Args:
        var: EnvVar enum member
        value: Value to set

    Example:
        >>> set_env(EnvVar.TOOL_TIMEOUT, "600")
    """
    os.environ[var.value] = value


def unset_env(var: EnvVar) -> None:
    """Unset environment variable if it exists.

    Args:
        var: EnvVar enum member

    Example:
        >>> unset_env(EnvVar.TOOL_TIMEOUT)
    """
    os.environ.pop(var.value, None)


def is_set(var: EnvVar) -> bool:
    """Check if environment variable is set.

    Args:
        var: EnvVar enum member

    Returns:
        True if variable is set (even if empty string)

    Example:
        >>> if is_set(EnvVar.TOOL_TIMEOUT):
        ...     print("Timeout is configured")
    """
    return var.value in os.environ


def get_env_int(var: EnvVar, default: int | None = None) -> int | None:
    """Get environment variable as integer.

    Args:
        var: EnvVar enum member
        default: Default value if not set or invalid

    Returns:
        Integer value or default

    Example:
        >>> timeout = get_env_int(EnvVar.TOOL_TIMEOUT, 120)
    """
    value = get_env(var)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        return default


def get_env_float(var: EnvVar, default: float | None = None) -> float | None:
    """Get environment variable as float.

    Args:
        var: EnvVar enum member
        default: Default value if not set or invalid

    Returns:
        Float value or default

    Example:
        >>> timeout = get_env_float(EnvVar.TOOL_TIMEOUT, 120.0)
    """
    value = get_env(var)
    if value is None:
        return default

    try:
        return float(value)
    except ValueError:
        return default


def get_env_bool(var: EnvVar, default: bool = False) -> bool:
    """Get environment variable as boolean.

    Args:
        var: EnvVar enum member
        default: Default value if not set

    Returns:
        Boolean value (true for "1", "true", "yes", case-insensitive)

    Example:
        >>> dynamic = get_env_bool(EnvVar.CLI_DYNAMIC_TOOLS, False)
    """
    value = get_env(var)
    if value is None:
        return default

    return value.lower() in ("1", "true", "yes", "on")


def get_env_list(
    var: EnvVar, separator: str = ",", default: list[str] | None = None
) -> list[str]:
    """Get environment variable as list of strings.

    Args:
        var: EnvVar enum member
        separator: String separator (default: comma)
        default: Default value if not set

    Returns:
        List of strings (stripped of whitespace)

    Example:
        >>> tools = get_env_list(EnvVar.CLI_INCLUDE_TOOLS, default=[])
        # "tool1, tool2, tool3" -> ["tool1", "tool2", "tool3"]
    """
    value = get_env(var)
    if value is None:
        return default or []

    return [item.strip() for item in value.split(separator) if item.strip()]
