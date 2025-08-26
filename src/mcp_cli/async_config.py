# mcp_cli/async_config.py - FIXED VERSION
"""
Async configuration loading for MCP servers using chuk-tool-processor APIs.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any


async def load_server_config(
    config_path: str, server_name: str = None
) -> Dict[str, Any]:
    """
    Load the server configuration from a JSON file.

    FIXED: Updated to work with chuk-tool-processor instead of old chuk_mcp APIs.
    """
    try:
        # Debug logging
        logging.debug(f"Loading config from {config_path}")

        # Read the configuration file
        config_file_path = Path(config_path)
        if not config_file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file_path, "r") as config_file:
            config = json.load(config_file)

        # If specific server requested, return just that server's config
        if server_name:
            server_config = config.get("mcpServers", {}).get(server_name)
            if not server_config:
                error_msg = f"Server '{server_name}' not found in configuration file."
                logging.error(error_msg)
                raise ValueError(error_msg)

            # Return in format expected by chuk-tool-processor
            return {
                "command": server_config["command"],
                "args": server_config.get("args", []),
                "env": server_config.get("env", {}),
            }

        # Return entire config for processing multiple servers
        return config

    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in configuration file: {e.msg}"
        logging.error(error_msg)
        raise json.JSONDecodeError(error_msg, e.doc, e.pos)
    except ValueError as e:
        logging.error(str(e))
        raise


async def load_all_server_configs(config_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all server configurations from a JSON file.

    Returns:
        Dictionary mapping server names to their configurations
    """
    config = await load_server_config(config_path)
    mcp_servers = config.get("mcpServers", {})

    # Transform to expected format
    result = {}
    for server_name, server_config in mcp_servers.items():
        result[server_name] = {
            "command": server_config["command"],
            "args": server_config.get("args", []),
            "env": server_config.get("env", {}),
        }

    return result
