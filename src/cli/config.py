import json
import logging

from mcp.transport.stdio.stdio_server_parameters import StdioServerParameters


async def load_config(config_path: str, server_name: str) -> StdioServerParameters:
    """Load the server configuration from a JSON file."""
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

        # Construct the server parameters
        result = StdioServerParameters(
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

async def get_all_server_names(config_path: str) -> list[str]:
    """Get a list of all server names from the configuration file."""
    try:
        # Read the configuration file
        with open(config_path, "r") as config_file:
            config = json.load(config_file)

        # Retrieve all server names
        servers = config.get("mcpServers", {})
        if not servers:
            error_msg = "No servers found in configuration file."
            logging.error(error_msg)
            raise ValueError(error_msg)
            
        return list(servers.keys())

    except FileNotFoundError:
        error_msg = f"Configuration file not found: {config_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in configuration file: {e.msg}"
        logging.error(error_msg)
        raise json.JSONDecodeError(error_msg, e.doc, e.pos)

async def get_server_names(config_path: str) -> list[str]:
    """Get a list of all server names from the configuration file."""
    try:
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
        
        servers = config.get("mcpServers", {})
        if not servers:
            error_msg = "No servers found in configuration file."
            logging.error(error_msg)
            raise ValueError(error_msg)
            
        return list(servers.keys())
        
    except FileNotFoundError:
        error_msg = f"Configuration file not found: {config_path}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)
    except json.JSONDecodeError as e:
        error_msg = f"Invalid JSON in configuration file: {e.msg}"
        logging.error(error_msg)
        raise json.JSONDecodeError(error_msg, e.doc, e.pos)