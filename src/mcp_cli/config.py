# mcp_cli/config.py
import json
import logging
from pathlib import Path
from typing import Dict, Any


async def load_config(config_path: str, server_name: str = None) -> Dict[str, Any]:
    """
    Load the server configuration from a JSON file.

    FIXED: Updated to work with current chuk-tool-processor APIs instead of old chuk_mcp.
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

        # Return entire config
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
