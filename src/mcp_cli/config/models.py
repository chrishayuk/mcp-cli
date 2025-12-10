"""Clean Pydantic configuration models - async native, type safe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from mcp_cli.config.enums import TimeoutType, TokenBackend
from mcp_cli.config.defaults import (
    DEFAULT_HTTP_CONNECT_TIMEOUT,
    DEFAULT_HTTP_REQUEST_TIMEOUT,
    DEFAULT_SERVER_INIT_TIMEOUT,
    DEFAULT_STREAMING_CHUNK_TIMEOUT,
    DEFAULT_STREAMING_FIRST_CHUNK_TIMEOUT,
    DEFAULT_STREAMING_GLOBAL_TIMEOUT,
    DEFAULT_TOOL_EXECUTION_TIMEOUT,
    DEFAULT_MAX_TOOL_CONCURRENCY,
    DEFAULT_CONFIRM_TOOLS,
    DEFAULT_DYNAMIC_TOOLS_ENABLED,
    DEFAULT_PROVIDER,
    DEFAULT_MODEL,
    DEFAULT_THEME,
    DEFAULT_VERBOSE,
)


class TimeoutConfig(BaseModel):
    """Timeout configuration with proper defaults.

    All values in seconds. Immutable after creation.
    No more magic numbers!
    """

    streaming_chunk: float = Field(
        default=DEFAULT_STREAMING_CHUNK_TIMEOUT,
        gt=0,
        description="Timeout for each streaming chunk",
    )
    streaming_global: float = Field(
        default=DEFAULT_STREAMING_GLOBAL_TIMEOUT,
        gt=0,
        description="Maximum total streaming duration",
    )
    streaming_first_chunk: float = Field(
        default=DEFAULT_STREAMING_FIRST_CHUNK_TIMEOUT,
        gt=0,
        description="Timeout for first chunk (may be longer)",
    )
    tool_execution: float = Field(
        default=DEFAULT_TOOL_EXECUTION_TIMEOUT,
        gt=0,
        description="Tool execution timeout",
    )
    server_init: float = Field(
        default=DEFAULT_SERVER_INIT_TIMEOUT,
        gt=0,
        description="Server initialization timeout",
    )
    http_request: float = Field(
        default=DEFAULT_HTTP_REQUEST_TIMEOUT,
        gt=0,
        description="HTTP request timeout",
    )
    http_connect: float = Field(
        default=DEFAULT_HTTP_CONNECT_TIMEOUT,
        gt=0,
        description="HTTP connection timeout",
    )

    model_config = {"frozen": True}

    def get(self, timeout_type: TimeoutType) -> float:
        """Get timeout by enum (type-safe)."""
        return getattr(self, timeout_type.value)

    async def get_async(self, timeout_type: TimeoutType) -> float:
        """Async getter for consistency."""
        return self.get(timeout_type)


class ToolConfig(BaseModel):
    """Tool behavior configuration.

    No more magic numbers!
    """

    include_tools: list[str] | None = Field(
        default=None,
        description="Whitelist of tool names (None = all)",
    )
    exclude_tools: list[str] | None = Field(
        default=None,
        description="Blacklist of tool names",
    )
    dynamic_tools_enabled: bool = Field(
        default=DEFAULT_DYNAMIC_TOOLS_ENABLED,
        description="Enable dynamic tool discovery",
    )
    confirm_tools: bool = Field(
        default=DEFAULT_CONFIRM_TOOLS,
        description="Require confirmation before execution",
    )
    max_concurrency: int = Field(
        default=DEFAULT_MAX_TOOL_CONCURRENCY,
        gt=0,
        le=100,
        description="Max concurrent tool executions",
    )

    model_config = {"frozen": True}

    @field_validator("include_tools", "exclude_tools")
    @classmethod
    def validate_tool_lists(cls, v: list[str] | None) -> list[str] | None:
        """Ensure tool lists are non-empty if provided."""
        if v is not None and len(v) == 0:
            return None
        return v


class VaultConfig(BaseModel):
    """HashiCorp Vault configuration."""

    url: str | None = None
    token: str | None = None
    mount_point: str = "secret"
    path_prefix: str = "mcp-cli/oauth"
    namespace: str | None = None

    model_config = {"frozen": True}


class TokenStorageConfig(BaseModel):
    """Token storage configuration."""

    backend: TokenBackend = TokenBackend.AUTO
    password: str | None = Field(default=None, repr=False)  # Don't print passwords
    vault: VaultConfig = Field(default_factory=VaultConfig)

    model_config = {"frozen": True}


class MCPConfig(BaseModel):
    """Complete MCP configuration - clean, immutable, type-safe.

    This is the source of truth loaded from config files.
    RuntimeConfig wraps this with CLI/env overrides.
    """

    # Provider/Model defaults (no more magic strings!)
    default_provider: str = DEFAULT_PROVIDER
    default_model: str = DEFAULT_MODEL

    # UI (theme names validated by chuk-term)
    theme: str = DEFAULT_THEME  # default|dark|light|minimal|terminal
    verbose: bool = DEFAULT_VERBOSE

    # Configurations
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    token_storage: TokenStorageConfig = Field(default_factory=TokenStorageConfig)

    # Servers - kept as dict for flexibility but typed
    # Use alias to support both 'servers' and 'mcpServers' from config files
    servers: dict[str, Any] = Field(default_factory=dict, alias="mcpServers")

    model_config = {"frozen": True, "populate_by_name": True}

    @classmethod
    async def load_async(cls, config_path: Path) -> MCPConfig:
        """Async load from file (future-proof for async I/O)."""
        import asyncio
        import json

        if not config_path.exists():
            return cls()

        # Use asyncio for file I/O
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, config_path.read_text)
        parsed = json.loads(data)

        return cls.model_validate(parsed)

    @classmethod
    def load_sync(cls, config_path: Path) -> MCPConfig:
        """Synchronous load (for backward compat during transition)."""
        import json

        if not config_path.exists():
            return cls()

        data = json.loads(config_path.read_text())
        return cls.model_validate(data)

    @classmethod
    def load_from_file(cls, config_path: Path) -> MCPConfig:
        """Alias for load_sync for backward compatibility."""
        return cls.load_sync(config_path)

    async def save_async(self, config_path: Path) -> None:
        """Async save to file."""
        import asyncio
        import json

        data = self.model_dump(mode="json", exclude_none=True)
        json_str = json.dumps(data, indent=2)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, config_path.write_text, json_str)

    def save_sync(self, config_path: Path) -> None:
        """Synchronous save."""
        import json

        data = self.model_dump(mode="json", exclude_none=True)
        config_path.write_text(json.dumps(data, indent=2))


class ConfigOverride(BaseModel):
    """Type-safe configuration override from CLI/env.

    Use this instead of dict[str, Any] for CLI arguments.
    """

    timeouts: dict[TimeoutType, float] = Field(default_factory=dict)
    tools: dict[str, Any] = Field(default_factory=dict)
    provider: str | None = None
    model: str | None = None
    theme: str | None = None  # Theme name - validated by chuk-term

    model_config = {"frozen": False}  # Mutable for building

    def set_timeout(self, timeout_type: TimeoutType, value: float) -> None:
        """Type-safe timeout override."""
        if value <= 0:
            raise ValueError(f"Timeout must be positive: {value}")
        self.timeouts[timeout_type] = value

    def apply_tool_timeout_to_all(self, value: float) -> None:
        """Apply single timeout to all relevant types."""
        self.set_timeout(TimeoutType.STREAMING_CHUNK, value)
        self.set_timeout(TimeoutType.STREAMING_GLOBAL, value)
        self.set_timeout(TimeoutType.TOOL_EXECUTION, value)
