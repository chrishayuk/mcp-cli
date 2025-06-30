# mcp_cli/async_config.py
"""
Async configuration loading for MCP servers using new chuk-mcp APIs.
"""
import json
import logging

# Updated imports for new chuk-mcp APIs
from chuk_mcp.transports.stdio.parameters import StdioParameters

async def load_server_config(config_path: str, server_name: str) -> StdioParameters:
    """Load the server configuration from a JSON file using new chuk-mcp APIs."""
    try:
        # debug
        logging.debug(f"Loading config from {config_path}")

        # Read the configuration file
        with open(config_path, "r") as config_file:
            config = json.load(config_file)

        # Retrieve the server configuration
        server_config = config.get("mcpServers", {}).get(server_name)
        if not server_config:
            error_msg = f"Server '{server_name}' not found in configuration file."
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Construct the server parameters using new StdioParameters class
        result = StdioParameters(
            command=server_config["command"],
            args=server_config.get("args", []),
            env=server_config.get("env"),
        )

        # debug
        logging.debug(
            f"Loaded config: command='{result.command}', args={result.args}, env={result.env}"
        )

        # return result
        return result

    except FileNotFoundError:
        # error
        error_msg = f"Configuration file not found: {config_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    except json.JSONDecodeError as e:
        # json error
        error_msg = f"Invalid JSON in configuration file: {e.msg}"
        logging.error(error_msg)
        raise json.JSONDecodeError(error_msg, e.doc, e.pos)
    except ValueError as e:
        # error
        logging.error(str(e))
        raise