"""Configuration enums - no magic strings!"""

from __future__ import annotations

from enum import Enum


class TimeoutType(str, Enum):
    """All timeout configuration types - type-safe timeout keys."""

    STREAMING_CHUNK = "streaming_chunk"
    STREAMING_GLOBAL = "streaming_global"
    STREAMING_FIRST_CHUNK = "streaming_first_chunk"
    TOOL_EXECUTION = "tool_execution"
    SERVER_INIT = "server_init"
    HTTP_REQUEST = "http_request"
    HTTP_CONNECT = "http_connect"


class TokenBackend(str, Enum):
    """Token storage backend types."""

    AUTO = "auto"
    KEYCHAIN = "keychain"
    WINDOWS = "windows"
    SECRET_SERVICE = "secretservice"
    ENCRYPTED = "encrypted"
    VAULT = "vault"


class ConfigSource(str, Enum):
    """Configuration value source for priority resolution."""

    CLI = "cli"
    ENV = "env"
    FILE = "file"
    DEFAULT = "default"


# NOTE: Theme names come from chuk-term.ui.theme.Theme
# Valid values: "default", "dark", "light", "minimal", "terminal"
# We don't duplicate them here - use strings and let chuk-term validate
