"""
Configuration management for MCP CLI.

Clean, async-native, Pydantic-based configuration system.
"""

# New clean config system (primary)
from mcp_cli.config.enums import ConfigSource, TimeoutType, TokenBackend
from mcp_cli.config.models import (
    ConfigOverride,
    MCPConfig,
    TimeoutConfig,
    ToolConfig,
    TokenStorageConfig,
    VaultConfig,
)
from mcp_cli.config.runtime import ResolvedValue, RuntimeConfig

# Legacy compatibility (during transition)
from mcp_cli.config.config_manager import (
    ConfigManager,
    ServerConfig,
    detect_server_types,
    get_config,
    initialize_config,
    validate_server_config,
)
from mcp_cli.config.discovery import (
    force_discovery_refresh,
    get_available_models_quick,
    get_discovery_status,
    setup_chuk_llm_environment,
    trigger_discovery_after_setup,
    validate_provider_exists,
)
from mcp_cli.config.cli_options import (
    extract_server_names,
    get_config_summary,
    inject_logging_env_vars,
    load_config,
    process_options,
)

__all__ = [
    # Clean config system (USE THESE)
    "MCPConfig",
    "TimeoutConfig",
    "ToolConfig",
    "TokenStorageConfig",
    "VaultConfig",
    "RuntimeConfig",
    "ConfigOverride",
    "ResolvedValue",
    # Enums
    "TimeoutType",
    "TokenBackend",
    "ConfigSource",
    # Legacy (will be removed)
    "ServerConfig",
    "ConfigManager",
    "get_config",
    "initialize_config",
    "detect_server_types",
    "validate_server_config",
    # Discovery
    "setup_chuk_llm_environment",
    "trigger_discovery_after_setup",
    "get_available_models_quick",
    "validate_provider_exists",
    "get_discovery_status",
    "force_discovery_refresh",
    # CLI Options
    "load_config",
    "extract_server_names",
    "inject_logging_env_vars",
    "process_options",
    "get_config_summary",
]


# Convenience function for loading config
def load_runtime_config(
    config_path: str | None = None,
    cli_overrides: ConfigOverride | None = None,
) -> RuntimeConfig:
    """Load runtime configuration with clean API.

    Args:
        config_path: Path to config file (default: server_config.json)
        cli_overrides: CLI argument overrides

    Returns:
        RuntimeConfig instance ready to use
    """
    from pathlib import Path

    path = Path(config_path or "server_config.json")
    file_config = MCPConfig.load_sync(path)
    return RuntimeConfig(file_config, cli_overrides)


async def load_runtime_config_async(
    config_path: str | None = None,
    cli_overrides: ConfigOverride | None = None,
) -> RuntimeConfig:
    """Async load runtime configuration.

    Args:
        config_path: Path to config file (default: server_config.json)
        cli_overrides: CLI argument overrides

    Returns:
        RuntimeConfig instance ready to use
    """
    from pathlib import Path

    path = Path(config_path or "server_config.json")
    file_config = await MCPConfig.load_async(path)
    return RuntimeConfig(file_config, cli_overrides)


__all__.extend(["load_runtime_config", "load_runtime_config_async"])
