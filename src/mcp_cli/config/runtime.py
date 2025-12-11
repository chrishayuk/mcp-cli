"""Runtime configuration resolver - async native, type safe, no magic strings."""

from __future__ import annotations

import logging
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from mcp_cli.config.enums import ConfigSource, TimeoutType
from mcp_cli.config.models import ConfigOverride, MCPConfig, TimeoutConfig, ToolConfig
from mcp_cli.constants import EnvVar, get_env, get_env_bool, get_env_float, get_env_list

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ResolvedValue(BaseModel, Generic[T]):
    """A configuration value with its source for debugging."""

    value: T
    source: ConfigSource

    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class RuntimeConfig:
    """Runtime configuration resolver with 4-tier priority.

    Priority (highest to lowest):
    1. CLI overrides (ConfigOverride)
    2. Environment variables
    3. File config (MCPConfig)
    4. Defaults

    Pure async, type-safe, immutable after creation.
    """

    def __init__(
        self,
        file_config: MCPConfig,
        cli_overrides: ConfigOverride | None = None,
    ):
        """Initialize runtime config.

        Args:
            file_config: Loaded configuration from file
            cli_overrides: CLI argument overrides (type-safe)
        """
        self._file_config = file_config
        self._cli_overrides = cli_overrides or ConfigOverride()

        # Cache resolved values
        self._timeout_cache: dict[TimeoutType, ResolvedValue[float]] = {}

    # ================================================================
    # Timeout Resolution
    # ================================================================

    def get_timeout(self, timeout_type: TimeoutType) -> float:
        """Get timeout with priority resolution (sync)."""
        if timeout_type in self._timeout_cache:
            return self._timeout_cache[timeout_type].value

        resolved = self._resolve_timeout(timeout_type)
        self._timeout_cache[timeout_type] = resolved
        return resolved.value

    async def get_timeout_async(self, timeout_type: TimeoutType) -> float:
        """Get timeout asynchronously."""
        return self.get_timeout(timeout_type)

    def _resolve_timeout(self, timeout_type: TimeoutType) -> ResolvedValue[float]:
        """Resolve timeout from all sources with priority (type-safe!)."""
        # 1. Check CLI overrides
        if timeout_type in self._cli_overrides.timeouts:
            value = self._cli_overrides.timeouts[timeout_type]
            logger.debug(f"Timeout {timeout_type.value} from CLI: {value}s")
            return ResolvedValue(value=value, source=ConfigSource.CLI)

        # 2. Check environment variables (type-safe with EnvVar!)
        # Map timeout types to env vars
        env_var_map = {
            TimeoutType.STREAMING_CHUNK: EnvVar.STREAMING_CHUNK_TIMEOUT,
            TimeoutType.STREAMING_GLOBAL: EnvVar.STREAMING_GLOBAL_TIMEOUT,
            TimeoutType.STREAMING_FIRST_CHUNK: EnvVar.STREAMING_FIRST_CHUNK_TIMEOUT,
            TimeoutType.TOOL_EXECUTION: EnvVar.TOOL_EXECUTION_TIMEOUT,
            TimeoutType.SERVER_INIT: EnvVar.SERVER_INIT_TIMEOUT,
        }

        if timeout_type in env_var_map:
            env_value = get_env_float(env_var_map[timeout_type])
            if env_value is not None and env_value > 0:
                logger.debug(f"Timeout {timeout_type.value} from ENV: {env_value}s")
                return ResolvedValue(value=env_value, source=ConfigSource.ENV)

        # Special case: MCP_TOOL_TIMEOUT applies to multiple (type-safe!)
        if timeout_type in [
            TimeoutType.STREAMING_CHUNK,
            TimeoutType.STREAMING_GLOBAL,
            TimeoutType.TOOL_EXECUTION,
        ]:
            tool_timeout = get_env_float(EnvVar.TOOL_TIMEOUT)
            if tool_timeout is not None and tool_timeout > 0:
                logger.debug(
                    f"Timeout {timeout_type.value} from MCP_TOOL_TIMEOUT: {tool_timeout}s"
                )
                return ResolvedValue(value=tool_timeout, source=ConfigSource.ENV)

        # 3. Check file config
        value = self._file_config.timeouts.get(timeout_type)
        logger.debug(f"Timeout {timeout_type.value} from config file: {value}s")
        return ResolvedValue(value=value, source=ConfigSource.FILE)

    def get_all_timeouts(self) -> TimeoutConfig:
        """Get all resolved timeouts as immutable config."""
        return TimeoutConfig(
            streaming_chunk=self.get_timeout(TimeoutType.STREAMING_CHUNK),
            streaming_global=self.get_timeout(TimeoutType.STREAMING_GLOBAL),
            streaming_first_chunk=self.get_timeout(TimeoutType.STREAMING_FIRST_CHUNK),
            tool_execution=self.get_timeout(TimeoutType.TOOL_EXECUTION),
            server_init=self.get_timeout(TimeoutType.SERVER_INIT),
            http_request=self.get_timeout(TimeoutType.HTTP_REQUEST),
            http_connect=self.get_timeout(TimeoutType.HTTP_CONNECT),
        )

    async def get_all_timeouts_async(self) -> TimeoutConfig:
        """Get all timeouts asynchronously."""
        return self.get_all_timeouts()

    # ================================================================
    # Tool Configuration
    # ================================================================

    def get_tool_config(self) -> ToolConfig:
        """Get resolved tool configuration."""
        # CLI overrides take precedence
        include_tools = self._get_tool_list("include_tools")
        exclude_tools = self._get_tool_list("exclude_tools")
        dynamic_enabled = self._get_tool_bool("dynamic_tools_enabled")
        confirm_tools = self._get_tool_bool("confirm_tools")
        max_concurrency = self._get_tool_int("max_concurrency")

        return ToolConfig(
            include_tools=include_tools or self._file_config.tools.include_tools,
            exclude_tools=exclude_tools or self._file_config.tools.exclude_tools,
            dynamic_tools_enabled=dynamic_enabled
            if dynamic_enabled is not None
            else self._file_config.tools.dynamic_tools_enabled,
            confirm_tools=confirm_tools
            if confirm_tools is not None
            else self._file_config.tools.confirm_tools,
            max_concurrency=max_concurrency or self._file_config.tools.max_concurrency,
        )

    async def get_tool_config_async(self) -> ToolConfig:
        """Get tool config asynchronously."""
        return self.get_tool_config()

    def _get_tool_list(self, key: str) -> list[str] | None:
        """Get tool list from CLI/env (type-safe!)."""
        # Check CLI
        if key in self._cli_overrides.tools:
            return self._cli_overrides.tools[key]  # type: ignore[no-any-return]

        # Check environment (type-safe with EnvVar!)
        if key == "include_tools":
            return get_env_list(EnvVar.CLI_INCLUDE_TOOLS)
        elif key == "exclude_tools":
            return get_env_list(EnvVar.CLI_EXCLUDE_TOOLS)

        return None

    def _get_tool_bool(self, key: str) -> bool | None:
        """Get boolean tool config from CLI/env (type-safe!)."""
        # Check CLI
        if key in self._cli_overrides.tools:
            return bool(self._cli_overrides.tools[key])

        # Check environment (type-safe with EnvVar!)
        if key == "dynamic_tools_enabled":
            env_val = get_env(EnvVar.CLI_DYNAMIC_TOOLS)
            if env_val is not None:
                return get_env_bool(EnvVar.CLI_DYNAMIC_TOOLS)

        return None

    def _get_tool_int(self, key: str) -> int | None:
        """Get integer tool config from CLI/env (type-safe!)."""
        # Check CLI
        if key in self._cli_overrides.tools:
            try:
                return int(self._cli_overrides.tools[key])
            except (ValueError, TypeError):
                pass

        # No env vars for max_concurrency currently
        return None

    # ================================================================
    # Provider/Model
    # ================================================================

    @property
    def provider(self) -> str:
        """Get resolved provider (type-safe!)."""
        return (
            self._cli_overrides.provider
            or get_env(EnvVar.PROVIDER)
            or self._file_config.default_provider
        )

    @property
    def model(self) -> str:
        """Get resolved model (type-safe!)."""
        return (
            self._cli_overrides.model
            or get_env(EnvVar.MODEL)
            or self._file_config.default_model
        )

    # ================================================================
    # Debug
    # ================================================================

    def debug_report(self) -> dict[str, Any]:
        """Generate debug report showing all resolved values and sources."""
        return {
            "timeouts": {
                tt.value: {
                    "value": self._timeout_cache.get(
                        tt, self._resolve_timeout(tt)
                    ).value,
                    "source": self._timeout_cache.get(
                        tt, self._resolve_timeout(tt)
                    ).source.value,
                }
                for tt in TimeoutType
            },
            "provider": self.provider,
            "model": self.model,
            "tools": self.get_tool_config().model_dump(),
        }
