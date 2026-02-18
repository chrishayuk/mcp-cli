# mcp_cli/apps/models.py
"""Pydantic models for MCP Apps protocol (SEP-1865)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AppState(str, Enum):
    """Lifecycle states of an MCP App."""

    PENDING = "pending"
    INITIALIZING = "initializing"
    READY = "ready"
    CLOSED = "closed"


class AppInfo(BaseModel):
    """Runtime information about a running MCP App."""

    tool_name: str = Field(description="Tool that triggered this app")
    resource_uri: str = Field(description="ui:// resource URI")
    server_name: str = Field(description="MCP server providing this tool")
    state: AppState = AppState.PENDING
    port: int = Field(description="Local server port for this app")
    html_content: str = Field(default="", description="Fetched HTML content")
    csp: dict[str, Any] | None = Field(
        default=None, description="CSP configuration from resource metadata"
    )
    permissions: dict[str, Any] | None = Field(
        default=None, description="Requested iframe permissions"
    )

    model_config = {"frozen": False}

    @property
    def url(self) -> str:
        """URL to open in the browser."""
        return f"http://localhost:{self.port}"


class HostContext(BaseModel):
    """Context passed to the app during initialization."""

    theme: str = "dark"
    locale: str = "en"
    platform: str = "desktop"
    display_mode: str = "inline"
    available_display_modes: list[str] = Field(
        default_factory=lambda: ["inline", "fullscreen"]
    )

    model_config = {"frozen": False, "extra": "allow"}
