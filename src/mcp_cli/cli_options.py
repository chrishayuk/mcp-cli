# mcp_cli/cli_options.py
"""
Clean MCP CLI integration with ChukLLM.
Sets environment variables, triggers discovery, and gets out of the way.
ENHANCED: Now supports HTTP server detection and configuration.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from chuk_term.ui import output

logger = logging.getLogger(__name__)

# Global flag to ensure we only set up once
_ENV_SETUP_COMPLETE = False
_DISCOVERY_TRIGGERED = False


def setup_chuk_llm_environment():
    """
    Set up environment variables for ChukLLM discovery.
    MUST be called before any chuk_llm imports.
    """
    global _ENV_SETUP_COMPLETE

    if _ENV_SETUP_COMPLETE:
        return

    # Set environment variables (only if not already set by user)
    env_vars = {
        "CHUK_LLM_DISCOVERY_ENABLED": "true",
        "CHUK_LLM_AUTO_DISCOVER": "true",
        "CHUK_LLM_DISCOVERY_ON_STARTUP": "true",
        "CHUK_LLM_DISCOVERY_TIMEOUT": "10",
        "CHUK_LLM_OLLAMA_DISCOVERY": "true",
        "CHUK_LLM_OPENAI_DISCOVERY": "true",
        "CHUK_LLM_OPENAI_TOOL_COMPATIBILITY": "true",
        "CHUK_LLM_UNIVERSAL_TOOLS": "true",
    }

    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value

    _ENV_SETUP_COMPLETE = True
    logger.debug("ChukLLM environment variables set")


def trigger_discovery_after_setup():
    """
    Trigger discovery after environment setup.
    Call this after setup_chuk_llm_environment() and before using models.
    """
    global _DISCOVERY_TRIGGERED

    if _DISCOVERY_TRIGGERED:
        return 0

    try:
        # Import discovery functions
        from chuk_llm.api.providers import trigger_ollama_discovery_and_refresh

        logger.debug("Triggering Ollama discovery from cli_options...")

        # Trigger Ollama discovery to get all available models
        new_functions = trigger_ollama_discovery_and_refresh()

        _DISCOVERY_TRIGGERED = True

        if new_functions:
            logger.debug(f"CLI discovery: {len(new_functions)} new Ollama functions")
        else:
            logger.debug("CLI discovery: no new functions (may already be cached)")

        return len(new_functions)

    except Exception as e:
        logger.debug(f"CLI discovery failed: {e}")
        return 0


def get_available_models_quick(provider: str = "ollama") -> List[str]:
    """
    Quick function to get available models after discovery.
    """
    try:
        from chuk_llm.llm.client import list_available_providers

        providers = list_available_providers()
        return providers.get(provider, {}).get("models", [])
    except Exception as e:
        logger.debug(f"Could not get models for {provider}: {e}")
        return []


def validate_provider_exists(provider: str) -> bool:
    """Validate provider exists, potentially after discovery"""
    try:
        from chuk_llm.configuration import get_config

        config = get_config()
        config.get_provider(provider)  # This will raise if not found
        return True
    except Exception:
        return False


def load_config(config_file: str) -> Optional[dict]:
    """Load MCP server config file."""
    try:
        if Path(config_file).is_file():
            with open(config_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Error loading config file '%s': %s", config_file, exc)
    return None


def extract_server_names(
    cfg: Optional[dict], specified: List[str] = None
) -> Dict[int, str]:
    """Extract server names from config with HTTP server support, respecting disabled status from preferences."""
    if not cfg or "mcpServers" not in cfg:
        return {}

    servers = cfg["mcpServers"]

    # Get preference manager to check disabled servers
    from mcp_cli.utils.preferences import get_preference_manager

    pref_manager = get_preference_manager()

    if specified:
        # Filter to only specified servers that exist in config
        valid_servers = []
        for server in specified:
            if server in servers:
                valid_servers.append(server)
            else:
                logger.warning(f"Server '{server}' not found in configuration")
        return {i: name for i, name in enumerate(valid_servers)}
    else:
        # Only include enabled servers based on preferences
        enabled_servers = []
        for server_name in servers.keys():
            if not pref_manager.is_server_disabled(server_name):
                enabled_servers.append(server_name)
        return {i: name for i, name in enumerate(enabled_servers)}


def detect_server_types(cfg: dict, servers: List[str]) -> Tuple[List[dict], List[str]]:
    """
    Detect which servers are HTTP vs STDIO based on configuration.

    Returns:
        Tuple of (http_servers_list, stdio_servers_list)
    """
    http_servers = []
    stdio_servers = []

    if not cfg or "mcpServers" not in cfg:
        # No config, assume all are STDIO
        return [], servers

    mcp_servers = cfg["mcpServers"]

    for server in servers:
        server_config = mcp_servers.get(server, {})

        if "url" in server_config:
            # HTTP server
            http_servers.append({"name": server, "url": server_config["url"]})
            logger.debug(f"Detected HTTP server: {server} -> {server_config['url']}")
        elif "command" in server_config:
            # STDIO server
            stdio_servers.append(server)
            logger.debug(f"Detected STDIO server: {server}")
        else:
            logger.warning(
                f"Server '{server}' has unclear configuration, assuming STDIO"
            )
            stdio_servers.append(server)

    return http_servers, stdio_servers


def validate_server_config(cfg: dict, servers: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate server configuration and return status and errors.

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    if not cfg or "mcpServers" not in cfg:
        errors.append("No mcpServers section found in configuration")
        return False, errors

    mcp_servers = cfg["mcpServers"]

    for server in servers:
        if server not in mcp_servers:
            errors.append(f"Server '{server}' not found in configuration")
            continue

        server_config = mcp_servers[server]

        # Check for valid configuration
        has_url = "url" in server_config
        has_command = "command" in server_config

        if not has_url and not has_command:
            errors.append(f"Server '{server}' missing both 'url' and 'command' fields")
        elif has_url and has_command:
            errors.append(
                f"Server '{server}' has both 'url' and 'command' fields (should have only one)"
            )
        elif has_url:
            # Validate URL format
            url = server_config["url"]
            if not url.startswith(("http://", "https://")):
                errors.append(
                    f"Server '{server}' URL must start with http:// or https://"
                )
        elif has_command:
            # Validate command format
            command = server_config["command"]
            if not isinstance(command, str) or not command.strip():
                errors.append(f"Server '{server}' command must be a non-empty string")

    return len(errors) == 0, errors


def inject_logging_env_vars(cfg: dict, quiet: bool = False) -> dict:
    """Inject logging environment variables into MCP server configs."""
    if not cfg or "mcpServers" not in cfg:
        return cfg

    log_level = "ERROR" if quiet else "WARNING"
    logging_env_vars = {
        "PYTHONWARNINGS": "ignore",
        "LOG_LEVEL": log_level,
        "CHUK_LOG_LEVEL": log_level,
        "MCP_LOG_LEVEL": log_level,
    }

    modified_cfg = json.loads(json.dumps(cfg))  # Deep copy

    for server_name, server_config in modified_cfg["mcpServers"].items():
        # Only inject env vars for STDIO servers (those with 'command')
        if "command" in server_config:
            if "env" not in server_config:
                server_config["env"] = {}

            for env_key, env_value in logging_env_vars.items():
                if env_key not in server_config["env"]:
                    server_config["env"][env_key] = env_value

    return modified_cfg


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
    all_servers = cfg.get("mcpServers", {})

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
        for server_name in all_servers.keys():
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
        cfg = inject_logging_env_vars(cfg, quiet=quiet)

        # Save modified config for MCP tool manager
        temp_config_path = (
            Path(config_file).parent / f"_modified_{Path(config_file).name}"
        )
        try:
            with open(temp_config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
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


def get_discovery_status() -> Dict[str, any]:
    """Get discovery status for debugging"""
    return {
        "env_setup_complete": _ENV_SETUP_COMPLETE,
        "discovery_triggered": _DISCOVERY_TRIGGERED,
        "discovery_enabled": os.getenv("CHUK_LLM_DISCOVERY_ENABLED", "false"),
        "ollama_discovery": os.getenv("CHUK_LLM_OLLAMA_DISCOVERY", "false"),
        "auto_discover": os.getenv("CHUK_LLM_AUTO_DISCOVER", "false"),
        "tool_compatibility": os.getenv("CHUK_LLM_OPENAI_TOOL_COMPATIBILITY", "false"),
        "universal_tools": os.getenv("CHUK_LLM_UNIVERSAL_TOOLS", "false"),
    }


def force_discovery_refresh():
    """Force a fresh discovery (useful for debugging)"""
    global _DISCOVERY_TRIGGERED
    _DISCOVERY_TRIGGERED = False

    # Set force refresh environment variable
    os.environ["CHUK_LLM_DISCOVERY_FORCE_REFRESH"] = "true"

    # Trigger discovery again
    return trigger_discovery_after_setup()


def get_config_summary(config_file: str) -> Dict[str, any]:
    """Get a summary of the configuration for debugging."""
    cfg = load_config(config_file)

    if not cfg:
        return {"error": "Could not load config file"}

    servers = cfg.get("mcpServers", {})
    http_servers, stdio_servers = detect_server_types(cfg, list(servers.keys()))

    return {
        "config_file": config_file,
        "total_servers": len(servers),
        "http_servers": len(http_servers),
        "stdio_servers": len(stdio_servers),
        "server_names": list(servers.keys()),
        "http_server_details": [
            {"name": s["name"], "url": s["url"]} for s in http_servers
        ],
        "stdio_server_details": stdio_servers,
    }
