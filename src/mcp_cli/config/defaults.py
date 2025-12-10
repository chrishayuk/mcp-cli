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

DEFAULT_MAX_TURNS = 30
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
