"""Default configuration values - no more magic numbers!

All default values should be defined here, not hardcoded in the code.
"""

from __future__ import annotations


# ================================================================
# Timeout Defaults (in seconds)
# ================================================================

# Streaming timeouts
DEFAULT_STREAMING_CHUNK_TIMEOUT = 45.0
"""Default timeout for receiving each streaming chunk."""

DEFAULT_STREAMING_GLOBAL_TIMEOUT = 300.0
"""Default total streaming timeout."""

DEFAULT_STREAMING_FIRST_CHUNK_TIMEOUT = 60.0
"""Default timeout for first chunk (may need longer for complex queries)."""

# Tool timeouts
DEFAULT_TOOL_EXECUTION_TIMEOUT = 120.0
"""Default timeout for tool execution."""

# Server timeouts
DEFAULT_SERVER_INIT_TIMEOUT = 120.0
"""Default timeout for server initialization."""

# HTTP timeouts
DEFAULT_HTTP_REQUEST_TIMEOUT = 30.0
"""Default timeout for HTTP requests."""

DEFAULT_HTTP_CONNECT_TIMEOUT = 10.0
"""Default timeout for HTTP connections."""

# Discovery/UI timeouts (moved from constants/timeouts.py)
DISCOVERY_TIMEOUT = 10.0
"""Provider discovery HTTP timeout."""

REFRESH_TIMEOUT = 1.0
"""Display refresh timeout."""

SHUTDOWN_TIMEOUT = 0.5
"""Graceful shutdown timeout."""


# ================================================================
# Tool Configuration Defaults
# ================================================================

DEFAULT_MAX_TOOL_CONCURRENCY = 5
"""Default maximum concurrent tool executions."""

DEFAULT_CONFIRM_TOOLS = True
"""Default: require confirmation before executing tools."""

DEFAULT_DYNAMIC_TOOLS_ENABLED = False
"""Default: dynamic tool discovery disabled."""


# ================================================================
# Conversation Defaults
# ================================================================

DEFAULT_MAX_TURNS = 100
"""Default maximum conversation turns before exit."""

DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant with access to tools."
"""Default system prompt."""


# ================================================================
# Provider/Model Defaults
# ================================================================

DEFAULT_PROVIDER = "openai"
"""Default LLM provider."""

DEFAULT_MODEL = "gpt-4o-mini"
"""Default LLM model."""


# ================================================================
# UI Defaults
# ================================================================

DEFAULT_THEME = "default"
"""Default UI theme (from chuk-term)."""

DEFAULT_VERBOSE = True
"""Default verbosity level."""


# ================================================================
# Token/Auth Defaults
# ================================================================

DEFAULT_TOKEN_BACKEND = "auto"
"""Default token storage backend."""


# ================================================================
# Path Defaults
# ================================================================

DEFAULT_CONFIG_FILENAME = "server_config.json"
"""Default configuration filename."""


# ================================================================
# Application Constants
# ================================================================

NAMESPACE = "mcp-cli"
"""Application namespace."""

OAUTH_NAMESPACE = NAMESPACE
"""OAuth namespace (same as app namespace)."""

PROVIDER_NAMESPACE = "provider"
"""Provider namespace for token storage."""

GENERIC_NAMESPACE = "generic"
"""Generic namespace."""

APP_NAME = "mcp-cli"
"""Application name."""


# ================================================================
# Platform Constants
# ================================================================

PLATFORM_WINDOWS = "win32"
"""Windows platform identifier (from sys.platform)."""

PLATFORM_DARWIN = "darwin"
"""macOS platform identifier (from sys.platform)."""

PLATFORM_LINUX = "linux"
"""Linux platform identifier (from sys.platform)."""


# ================================================================
# Provider Constants
# ================================================================

PROVIDER_OLLAMA = "ollama"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GROQ = "groq"
PROVIDER_DEEPSEEK = "deepseek"
PROVIDER_XAI = "xai"

SUPPORTED_PROVIDERS = [
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    PROVIDER_ANTHROPIC,
    PROVIDER_GROQ,
    PROVIDER_DEEPSEEK,
    PROVIDER_XAI,
]
"""List of supported LLM providers."""


# ================================================================
# JSON Schema Type Constants
# ================================================================

JSON_TYPE_STRING = "string"
JSON_TYPE_NUMBER = "number"
JSON_TYPE_INTEGER = "integer"
JSON_TYPE_BOOLEAN = "boolean"
JSON_TYPE_ARRAY = "array"
JSON_TYPE_OBJECT = "object"
JSON_TYPE_NULL = "null"

JSON_TYPES = [
    JSON_TYPE_STRING,
    JSON_TYPE_NUMBER,
    JSON_TYPE_INTEGER,
    JSON_TYPE_BOOLEAN,
    JSON_TYPE_ARRAY,
    JSON_TYPE_OBJECT,
    JSON_TYPE_NULL,
]
"""All valid JSON Schema types."""


# ================================================================
# Middleware Configuration
# ================================================================
# Middleware (retry, circuit breaker, rate limiting) is provided by
# chuk-tool-processor. See chuk_tool_processor.mcp.MiddlewareConfig
# for configuration options.
DEFAULT_MIDDLEWARE_ENABLED = True
"""Enable CTP middleware by default."""
