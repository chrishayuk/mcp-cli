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

DEFAULT_STREAMING_FIRST_CHUNK_AFTER_TOOLS_TIMEOUT = 180.0
"""Timeout for first chunk after tool calls (thinking models need extended processing time)."""

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
# Context Management Defaults
# ================================================================

DEFAULT_MAX_TOOL_RESULT_CHARS = 100_000
"""Max chars for a single tool result in conversation history (~25K tokens). 0 = unlimited."""

DEFAULT_MAX_HISTORY_MESSAGES = 200
"""Max conversation history messages (sliding window). 0 = unlimited."""

DEFAULT_INFINITE_CONTEXT = False
"""Enable infinite context mode (auto-summarization via SessionManager)."""

DEFAULT_TOKEN_THRESHOLD = 4000
"""Token threshold for infinite context segmentation."""

DEFAULT_MAX_TURNS_PER_SEGMENT = 20
"""Max turns per segment before context packing triggers."""

DEFAULT_MAX_STREAMING_BUFFER_CHARS = 1_048_576
"""Max accumulated streaming content in chars (1 MB). 0 = unlimited."""

DEFAULT_MAX_STREAMING_CHUNKS = 50_000
"""Max streaming chunks before stall detection. 0 = unlimited."""


# ================================================================
# Tier 2: Efficiency & Resilience Defaults
# ================================================================

DEFAULT_BATCH_TIMEOUT_MULTIPLIER = 2.0
"""Batch timeout = max(per_tool_timeout * multiplier, floor)."""

DEFAULT_BATCH_TIMEOUT_FLOOR = 60.0
"""Minimum batch timeout in seconds."""

DEFAULT_SYSTEM_PROMPT_TOOL_SUMMARY_THRESHOLD = 20
"""When server has more than this many tools, show summary instead of full list."""

DEFAULT_RECONNECT_ON_FAILURE = True
"""Attempt to reconnect when a tool execution fails with a connection error."""

DEFAULT_MAX_RECONNECT_ATTEMPTS = 3
"""Maximum number of reconnection attempts before giving up."""

DEFAULT_CONTEXT_NOTICES_ENABLED = True
"""Enable LLM-visible context management notices (truncation, eviction, stripping)."""

DEFAULT_MAX_CONSECUTIVE_DUPLICATES = 5
"""Abort conversation loop after this many consecutive duplicate tool calls."""

DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES = 3
"""Warn user after this many consecutive transport failures."""

DEFAULT_SYSTEM_PROMPT_TOOL_PREVIEW_COUNT = 5
"""Number of tool names to show before summarizing in system prompt."""

DYNAMIC_TOOL_PROXY_NAME = "call_tool"
"""Name of the dynamic tool proxy used in discovery mode."""

TRANSPORT_ERROR_PATTERNS = (
    "transport not initialized",
    "transport",
)
"""Patterns (lowercase) indicating transport/connection failure in error messages."""


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
