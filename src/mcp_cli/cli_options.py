# mcp_cli/cli_options.py
"""
Shared option-processing helpers for MCP-CLI commands.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def load_config(config_file: str) -> Optional[dict]:
    """Read config file and return dict or None."""
    try:
        if Path(config_file).is_file():
            with open(config_file, "r", encoding="utf-8") as fh:
                return json.load(fh)
        logger.warning("Config file '%s' not found.", config_file)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Error loading config file '%s': %s", config_file, exc)
    return None


def extract_server_names(cfg: Optional[dict], specified: List[str] = None) -> Dict[int, str]:
    """Extract server names from config, optionally filtered by specified list."""
    if not cfg or "mcpServers" not in cfg:
        return {}
    
    servers = cfg["mcpServers"]
    
    if specified:
        return {i: name for i, name in enumerate(specified) if name in servers}
    else:
        return {i: name for i, name in enumerate(servers.keys())}


def inject_logging_env_vars(cfg: dict, quiet: bool = False) -> dict:
    """
    Inject environment variables to suppress MCP server logging noise.
    
    This modifies the server configuration to pass logging environment variables
    to each MCP server subprocess, which should suppress their verbose output.
    """
    if not cfg or "mcpServers" not in cfg:
        return cfg
    
    # Environment variables to suppress logging in MCP servers
    logging_env_vars = {
        "PYTHONWARNINGS": "ignore",  # Suppress Python warnings
        "LOG_LEVEL": "ERROR" if quiet else "WARNING",
        "LOGGING_LEVEL": "ERROR" if quiet else "WARNING", 
        "CHUK_LOG_LEVEL": "ERROR" if quiet else "WARNING",
        "CHUK_MCP_LOG_LEVEL": "ERROR" if quiet else "WARNING",
        "MCP_LOG_LEVEL": "ERROR" if quiet else "WARNING",
        
        # Suppress specific chuk loggers
        "CHUK_MCP_RUNTIME_LOG_LEVEL": "ERROR",
        "CHUK_SESSIONS_LOG_LEVEL": "ERROR",
        "CHUK_ARTIFACTS_LOG_LEVEL": "ERROR",
        
        # Python specific logging configuration
        "PYTHONPATH_LOGGING_LEVEL": "ERROR" if quiet else "WARNING",
    }
    
    # Create a modified copy of the config
    modified_cfg = json.loads(json.dumps(cfg))  # Deep copy
    
    for server_name, server_config in modified_cfg["mcpServers"].items():
        # Ensure the server config has an 'env' section
        if "env" not in server_config:
            server_config["env"] = {}
        
        # Add our logging environment variables
        for env_key, env_value in logging_env_vars.items():
            # Only add if not already specified (don't override user settings)
            if env_key not in server_config["env"]:
                server_config["env"][env_key] = env_value
        
        logger.debug(f"Injected logging env vars for server '{server_name}': {logging_env_vars}")
    
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
    Process CLI options and return (servers_list, user_specified, server_names).
    
    Sets environment variables for downstream components and injects logging
    configuration into MCP server configurations.
    """
    # Parse servers
    user_specified = []
    if server:
        user_specified = [s.strip() for s in server.split(",")]
    
    # Set environment variables (components will use ModelManager for actual values)
    os.environ["LLM_PROVIDER"] = provider
    if model:
        os.environ["LLM_MODEL"] = model
    else:
        # Set default model for openai provider if none specified
        if provider == "openai":
            os.environ["LLM_MODEL"] = "gpt-4o-mini"
    
    if not disable_filesystem:
        os.environ["SOURCE_FILESYSTEMS"] = json.dumps([os.getcwd()])
    
    # Load server config
    cfg = load_config(config_file)
    
    # ENHANCED: Inject logging environment variables into server configurations
    if cfg:
        cfg = inject_logging_env_vars(cfg, quiet=quiet)
        
        # Write the modified config to a temporary file for use by the tool manager
        temp_config_path = Path(config_file).parent / f"_modified_{Path(config_file).name}"
        try:
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2)
            logger.debug(f"Created modified config with logging env vars: {temp_config_path}")
            
            # Update the config file path to use the modified version
            # Note: This is a bit of a hack, but it allows us to inject env vars
            # without modifying the tool manager extensively
            os.environ["MCP_CLI_MODIFIED_CONFIG"] = str(temp_config_path)
            
        except Exception as e:
            logger.warning(f"Failed to create modified config: {e}")
    
    servers_list = user_specified or (list(cfg["mcpServers"].keys()) if cfg and "mcpServers" in cfg else [])
    server_names = extract_server_names(cfg, user_specified)
    
    logger.debug("Processed options: provider=%s model=%s servers=%s quiet=%s", provider, model, servers_list, quiet)
    return servers_list, user_specified, server_names