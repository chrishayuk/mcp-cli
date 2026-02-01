"""Constants module - DEPRECATED, use mcp_cli.config instead.

This module re-exports everything from mcp_cli.config for backwards compatibility.
All new code should import directly from mcp_cli.config.
"""

# Re-export everything from config for backwards compatibility
from mcp_cli.config import (
    # Application constants
    APP_NAME,
    APP_VERSION,
    GENERIC_NAMESPACE,
    NAMESPACE,
    OAUTH_NAMESPACE,
    PROVIDER_NAMESPACE,
    # Timeouts
    DEFAULT_HTTP_CONNECT_TIMEOUT,
    DEFAULT_HTTP_REQUEST_TIMEOUT,
    DISCOVERY_TIMEOUT,
    REFRESH_TIMEOUT,
    SHUTDOWN_TIMEOUT,
    # Platforms
    PLATFORM_DARWIN,
    PLATFORM_LINUX,
    PLATFORM_WINDOWS,
    # Providers
    PROVIDER_ANTHROPIC,
    PROVIDER_DEEPSEEK,
    PROVIDER_GROQ,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    PROVIDER_XAI,
    SUPPORTED_PROVIDERS,
    # JSON Schema
    JSON_TYPE_ARRAY,
    JSON_TYPE_BOOLEAN,
    JSON_TYPE_INTEGER,
    JSON_TYPE_NULL,
    JSON_TYPE_NUMBER,
    JSON_TYPE_OBJECT,
    JSON_TYPE_STRING,
    JSON_TYPES,
    # Enums
    ConversationAction,
    OutputFormat,
    ServerAction,
    ServerStatus,
    ThemeAction,
    TokenAction,
    TokenNamespace,
    ToolAction,
    # Environment variables
    EnvVar,
    get_env,
    get_env_bool,
    get_env_float,
    get_env_int,
    get_env_list,
    is_set,
    set_env,
    unset_env,
)

__all__ = [
    # Environment variables
    "EnvVar",
    "get_env",
    "set_env",
    "unset_env",
    "is_set",
    "get_env_int",
    "get_env_float",
    "get_env_bool",
    "get_env_list",
    # Enums
    "ServerStatus",
    "ConversationAction",
    "OutputFormat",
    "TokenAction",
    "TokenNamespace",
    "ServerAction",
    "ToolAction",
    "ThemeAction",
    # Timeouts
    "DISCOVERY_TIMEOUT",
    "REFRESH_TIMEOUT",
    "SHUTDOWN_TIMEOUT",
    "DEFAULT_HTTP_CONNECT_TIMEOUT",
    "DEFAULT_HTTP_REQUEST_TIMEOUT",
    # Providers
    "PROVIDER_OLLAMA",
    "PROVIDER_OPENAI",
    "PROVIDER_ANTHROPIC",
    "PROVIDER_GROQ",
    "PROVIDER_DEEPSEEK",
    "PROVIDER_XAI",
    "SUPPORTED_PROVIDERS",
    # Platforms
    "PLATFORM_WINDOWS",
    "PLATFORM_DARWIN",
    "PLATFORM_LINUX",
    # JSON Schema types
    "JSON_TYPE_STRING",
    "JSON_TYPE_NUMBER",
    "JSON_TYPE_INTEGER",
    "JSON_TYPE_BOOLEAN",
    "JSON_TYPE_ARRAY",
    "JSON_TYPE_OBJECT",
    "JSON_TYPE_NULL",
    "JSON_TYPES",
    # App constants
    "NAMESPACE",
    "OAUTH_NAMESPACE",
    "PROVIDER_NAMESPACE",
    "GENERIC_NAMESPACE",
    "APP_NAME",
    "APP_VERSION",
]
