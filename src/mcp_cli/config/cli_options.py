# mcp_cli/config/cli_options.py
"""
CLI options processing for MCP CLI.

This module handles CLI-specific configuration processing including:
- Environment variable setup
- Server filtering based on preferences
- Logging environment variable injection
- Config file modifications for CLI use
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from chuk_term.ui import output

from mcp_cli.config import (
    MCPConfig,
    setup_chuk_llm_environment,
    trigger_discovery_after_setup,
    detect_server_types,
    validate_server_config,
)

logger = logging.getLogger(__name__)


def load_config(config_file: str) -> Optional[MCPConfig]:
    """Load MCP server config file with fallback to bundled package config."""
    try:
        config_path = Path(config_file)

        # Try explicit path or current directory first
        if config_path.is_file():
            config = MCPConfig.load_from_file(config_path)
            # If config loaded but has no servers and file exists, it might be invalid JSON
            # Check if the file has content but failed to parse
            if not config.servers and config_path.stat().st_size > 0:
                try:
                    # Try to parse as JSON to verify it's valid
                    import json

                    config_path.read_text()
                    json.loads(config_path.read_text())
                except json.JSONDecodeError:
                    # Invalid JSON - return None
                    return None
            return config

        # If not found and using default name, try package bundle
        if config_file == "server_config.json":
            try:
                import importlib.resources as resources

                # Try Python 3.9+ API
                if hasattr(resources, "files"):
                    package_files = resources.files("mcp_cli")
                    bundled_config = package_files / "server_config.json"
                    if bundled_config.is_file():
                        logger.info("Loading bundled server configuration")
                        # Create a temporary Path object from the resource
                        return MCPConfig.load_from_file(Path(str(bundled_config)))
            except (ImportError, FileNotFoundError, AttributeError, TypeError) as e:
                logger.debug(f"Could not load bundled config: {e}")

    except Exception as exc:
        logger.error("Error loading config file '%s': %s", config_file, exc)
    return None


def extract_server_names(
    cfg: Optional[MCPConfig], specified: List[str] | None = None
) -> Dict[int, str]:
    """Extract server names from config with HTTP server support, respecting disabled status from preferences."""
    if not cfg or not cfg.servers:
        return {}

    # Get preference manager to check disabled servers
    from mcp_cli.utils.preferences import get_preference_manager

    pref_manager = get_preference_manager()

    if specified:
        # Filter to only specified servers that exist in config
        valid_servers = []
        for server in specified:
            if server in cfg.servers:
                valid_servers.append(server)
            else:
                logger.warning(f"Server '{server}' not found in configuration")
        return {i: name for i, name in enumerate(valid_servers)}
    else:
        # Only include enabled servers based on preferences
        enabled_servers = []
        for server_name in cfg.servers.keys():
            if not pref_manager.is_server_disabled(server_name):
                enabled_servers.append(server_name)
        return {i: name for i, name in enumerate(enabled_servers)}


def inject_logging_env_vars(cfg: MCPConfig, quiet: bool = False) -> None:
    """Inject logging environment variables into MCP server configs (modifies in place)."""
    if not cfg or not cfg.servers:
        return

    log_level = "ERROR" if quiet else "WARNING"
    logging_env_vars = {
        "PYTHONWARNINGS": "ignore",
        "LOG_LEVEL": log_level,
        "CHUK_LOG_LEVEL": log_level,
        "MCP_LOG_LEVEL": log_level,
    }

    for server_name, server_config in cfg.servers.items():
        # Only inject env vars for STDIO servers (those with 'command')
        if server_config.command:
            # Inject logging env vars if not already set
            for env_key, env_value in logging_env_vars.items():
                if env_key not in server_config.env:
                    server_config.env[env_key] = env_value


def process_options(
    server: Optional[str],
    disable_filesystem: bool,
    provider: str,
    model: Optional[str],
    config_file: str = "server_config.json",
    quiet: bool = False,
) -> Tuple[List[str], List[str], Dict[int, str]]:
    """
    Process CLI options. Sets up environment, triggers discovery, and parses config.
    ENHANCED: Now validates server configuration and provides better error messages.
    """

    # STEP 1: Set up ChukLLM environment first
    setup_chuk_llm_environment()

    # STEP 2: Trigger discovery immediately after setup
    discovery_count = trigger_discovery_after_setup()

    if discovery_count > 0:
        logger.debug(f"Discovery found {discovery_count} new functions")

    # STEP 3: Set model environment for downstream use
    os.environ["LLM_PROVIDER"] = provider
    if model:
        os.environ["LLM_MODEL"] = model

    # STEP 4: Set filesystem environment if needed
    if not disable_filesystem:
        os.environ["SOURCE_FILESYSTEMS"] = json.dumps([os.getcwd()])

    # STEP 5: Parse server configuration
    user_specified = []
    if server:
        user_specified = [s.strip() for s in server.split(",")]

    cfg = load_config(config_file)

    if not cfg:
        logger.warning(f"Could not load config file: {config_file}")
        # Return empty configuration
        return [], user_specified, {}

    # STEP 6: Validate configuration and filter disabled servers
    # Get preference manager to check disabled servers
    from mcp_cli.utils.preferences import get_preference_manager

    pref_manager = get_preference_manager()

    # Filter out disabled servers
    if user_specified:
        # If user explicitly requested servers, check if they're disabled
        enabled_from_requested = []
        for server in user_specified:
            if pref_manager.is_server_disabled(server):
                output.warning(f"Server '{server}' is disabled and cannot be used")
                output.hint(
                    f"To enable it, use: mcp-cli chat then /servers {server} enable"
                )
            else:
                enabled_from_requested.append(server)
        servers_list = enabled_from_requested

        if not servers_list and user_specified:
            output.warning("All requested servers are disabled")
            output.hint("Use 'mcp-cli servers' to see server status")
    else:
        # No specific servers requested - filter out disabled ones from preferences
        enabled_servers = []
        for server_name in cfg.servers.keys():
            if not pref_manager.is_server_disabled(server_name):
                enabled_servers.append(server_name)
            else:
                logger.debug(f"Skipping disabled server: {server_name}")
        servers_list = enabled_servers

        if not servers_list:
            logger.warning("No enabled servers found")

    if servers_list:
        is_valid, errors = validate_server_config(cfg, servers_list)
        if not is_valid:
            logger.error("Server configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            # Continue anyway but warn user

    # STEP 7: Handle MCP server logging
    if cfg:
        inject_logging_env_vars(cfg, quiet=quiet)

        # Save modified config for MCP tool manager
        temp_config_path = (
            Path(config_file).parent / f"_modified_{Path(config_file).name}"
        )
        try:
            cfg.save_to_file(temp_config_path)
            os.environ["MCP_CLI_MODIFIED_CONFIG"] = str(temp_config_path)
        except Exception as e:
            logger.warning(f"Failed to create modified config: {e}")

    # STEP 8: Build server list and extract server names
    server_names = extract_server_names(cfg, user_specified)

    # STEP 9: Log server type detection for debugging
    if cfg:
        http_servers, stdio_servers = detect_server_types(cfg, servers_list)
        logger.debug(
            f"Detected {len(http_servers)} HTTP servers, {len(stdio_servers)} STDIO servers"
        )
        if http_servers:
            logger.debug(f"HTTP servers: {[s['name'] for s in http_servers]}")
        if stdio_servers:
            logger.debug(f"STDIO servers: {stdio_servers}")

    logger.debug(
        f"Options processed: provider={provider}, model={model}, servers={len(servers_list)}"
    )

    return servers_list, user_specified, server_names


def get_config_summary(config_file: str) -> Dict[str, Any]:
    """Get a summary of the configuration for debugging."""
    cfg = load_config(config_file)

    if not cfg:
        return {"error": "Could not load config file"}

    server_names = list(cfg.servers.keys())
    http_servers, stdio_servers = detect_server_types(cfg, server_names)

    return {
        "config_file": config_file,
        "total_servers": len(cfg.servers),
        "http_servers": len(http_servers),
        "stdio_servers": len(stdio_servers),
        "server_names": server_names,
        "http_server_details": [
            {"name": s["name"], "url": s["url"]} for s in http_servers
        ],
        "stdio_server_details": stdio_servers,
    }
