"""Pydantic models for discovery configuration - no more dict goop!"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mcp_cli.constants import EnvVar, get_env_bool


class DiscoveryConfig(BaseModel):
    """ChukLLM discovery configuration loaded from environment.

    All discovery settings in one type-safe, immutable model.
    """

    discovery_enabled: bool = Field(
        default=False,
        description="Whether discovery is enabled",
    )
    ollama_discovery: bool = Field(
        default=False,
        description="Whether Ollama discovery is enabled",
    )
    auto_discover: bool = Field(
        default=False,
        description="Whether auto-discovery is enabled",
    )
    tool_compatibility: bool = Field(
        default=False,
        description="Whether OpenAI tool compatibility is enabled",
    )
    universal_tools: bool = Field(
        default=False,
        description="Whether universal tools are enabled",
    )

    model_config = {"frozen": True}

    @classmethod
    def from_env(cls) -> "DiscoveryConfig":
        """Load discovery configuration from environment variables.

        Uses EnvVar enum for type-safe environment access.
        """
        return cls(
            discovery_enabled=get_env_bool(
                EnvVar.CHUK_LLM_DISCOVERY_ENABLED, default=False
            ),
            ollama_discovery=get_env_bool(
                EnvVar.CHUK_LLM_OLLAMA_DISCOVERY, default=False
            ),
            auto_discover=get_env_bool(EnvVar.CHUK_LLM_AUTO_DISCOVER, default=False),
            tool_compatibility=get_env_bool(
                EnvVar.CHUK_LLM_OPENAI_TOOL_COMPATIBILITY, default=False
            ),
            universal_tools=get_env_bool(
                EnvVar.CHUK_LLM_UNIVERSAL_TOOLS, default=False
            ),
        )

    def to_dict(self) -> dict[str, bool]:
        """Convert to dictionary for compatibility."""
        return self.model_dump()  # type: ignore[no-any-return]


class DiscoveryStatus(BaseModel):
    """Discovery status for debugging.

    Includes both runtime state and configuration.
    """

    env_setup_complete: bool
    discovery_triggered: bool
    config: DiscoveryConfig

    model_config = {"frozen": True}

    def to_dict(self) -> dict[str, bool | dict[str, bool]]:
        """Convert to dictionary with flattened config for backward compatibility."""
        return {
            "env_setup_complete": self.env_setup_complete,
            "discovery_triggered": self.discovery_triggered,
            **{f"{k}": v for k, v in self.config.to_dict().items()},
        }


__all__ = [
    "DiscoveryConfig",
    "DiscoveryStatus",
]
