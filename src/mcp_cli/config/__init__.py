"""
Configuration management for MCP CLI.
"""

from mcp_cli.config.config_manager import (
    ServerConfig,
    MCPConfig,
    ConfigManager,
    get_config,
    initialize_config,
    detect_server_types,
    validate_server_config,
)
from mcp_cli.config.discovery import (
    setup_chuk_llm_environment,
    trigger_discovery_after_setup,
    get_available_models_quick,
    validate_provider_exists,
    get_discovery_status,
    force_discovery_refresh,
)
from mcp_cli.config.cli_options import (
    load_config,
    extract_server_names,
    inject_logging_env_vars,
    process_options,
    get_config_summary,
)

__all__ = [
    # Config Manager
    "ServerConfig",
    "MCPConfig",
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
