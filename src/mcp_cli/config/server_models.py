"""Clean Pydantic models for server configurations - no more dict goop!"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class HTTPServerConfig(BaseModel):
    """HTTP/SSE server configuration."""

    name: str
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    disabled: bool = False
    tool_timeout: float | None = Field(
        default=None, description="Per-server tool timeout"
    )
    init_timeout: float | None = Field(
        default=None, description="Per-server init timeout"
    )

    model_config = {"frozen": True}

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class STDIOServerConfig(BaseModel):
    """STDIO server configuration."""

    name: str
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    disabled: bool = False
    tool_timeout: float | None = Field(
        default=None, description="Per-server tool timeout"
    )
    init_timeout: float | None = Field(
        default=None, description="Per-server init timeout"
    )

    model_config = {"frozen": True}

    @field_validator("command")
    @classmethod
    def validate_command(cls, v: str) -> str:
        """Validate command is not empty."""
        if not v or not v.strip():
            raise ValueError("Command cannot be empty")
        return v.strip()


class OAuthConfig(BaseModel):
    """OAuth configuration for HTTP servers."""

    client_id: str
    client_secret: str | None = Field(default=None, repr=False)  # Don't print secrets
    authorization_url: str
    token_url: str
    scopes: list[str] = Field(default_factory=list)
    redirect_uri: str = "http://localhost:8080/callback"

    model_config = {"frozen": True}


class UnifiedServerConfig(BaseModel):
    """Unified server configuration supporting both HTTP and STDIO transports.

    Exactly one of (url, command) must be provided.
    """

    name: str

    # HTTP/SSE transport
    url: str | None = None
    headers: dict[str, str] | None = None
    oauth: OAuthConfig | None = None

    # STDIO transport
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    # Common
    disabled: bool = False
    tool_timeout: float | None = Field(
        default=None, description="Per-server tool timeout"
    )
    init_timeout: float | None = Field(
        default=None, description="Per-server init timeout"
    )

    model_config = {"frozen": True}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate server name."""
        if not v or not v.strip():
            raise ValueError("Server name cannot be empty")
        return v.strip()

    @field_validator("url")
    @classmethod
    def validate_url_format(cls, v: str | None) -> str | None:
        """Validate URL format if provided."""
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    @field_validator("command")
    @classmethod
    def validate_command_not_empty(cls, v: str | None) -> str | None:
        """Validate command is not empty string if provided."""
        if v is not None and not v.strip():
            raise ValueError("Command cannot be empty string")
        return v.strip() if v else None

    def model_post_init(self, __context) -> None:
        """Validate that exactly one of (url, command) is provided."""
        has_url = self.url is not None
        has_command = self.command is not None

        if not has_url and not has_command:
            raise ValueError(
                f"Server '{self.name}' must have either 'url' (HTTP/SSE) or 'command' (STDIO)"
            )
        if has_url and has_command:
            raise ValueError(
                f"Server '{self.name}' cannot have both 'url' and 'command' (choose one transport type)"
            )

    def to_http_config(self) -> HTTPServerConfig:
        """Convert to HTTP server config (raises if not HTTP)."""
        if not self.url:
            raise ValueError(f"Server '{self.name}' is not an HTTP server")
        return HTTPServerConfig(
            name=self.name,
            url=self.url,
            headers=self.headers or {},
            disabled=self.disabled,
            tool_timeout=self.tool_timeout,
            init_timeout=self.init_timeout,
        )

    def to_stdio_config(self) -> STDIOServerConfig:
        """Convert to STDIO server config (raises if not STDIO)."""
        if not self.command:
            raise ValueError(f"Server '{self.name}' is not a STDIO server")
        return STDIOServerConfig(
            name=self.name,
            command=self.command,
            args=self.args,
            env=self.env,
            disabled=self.disabled,
            tool_timeout=self.tool_timeout,
            init_timeout=self.init_timeout,
        )

    @property
    def is_http(self) -> bool:
        """Check if this is an HTTP/SSE server."""
        return self.url is not None

    @property
    def is_stdio(self) -> bool:
        """Check if this is a STDIO server."""
        return self.command is not None


class ServerConfigInput(BaseModel):
    """Input model for parsing server configs from dicts.

    This is the mutable version used during parsing.
    Convert to UnifiedServerConfig after validation.
    """

    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] | None = None
    oauth: dict[str, Any] | OAuthConfig | None = None
    disabled: bool = False
    tool_timeout: float | None = None
    init_timeout: float | None = None

    model_config = {"frozen": False, "extra": "ignore"}

    def to_unified(self, name: str) -> UnifiedServerConfig:
        """Convert to immutable UnifiedServerConfig."""
        # Parse OAuth if it's a dict
        oauth_config = None
        if self.oauth:
            if isinstance(self.oauth, dict):
                oauth_config = OAuthConfig.model_validate(self.oauth)
            else:
                oauth_config = self.oauth

        return UnifiedServerConfig(
            name=name,
            url=self.url,
            headers=self.headers,
            oauth=oauth_config,
            command=self.command,
            args=self.args,
            env=self.env,
            disabled=self.disabled,
            tool_timeout=self.tool_timeout,
            init_timeout=self.init_timeout,
        )


__all__ = [
    "HTTPServerConfig",
    "STDIOServerConfig",
    "OAuthConfig",
    "UnifiedServerConfig",
    "ServerConfigInput",
]
