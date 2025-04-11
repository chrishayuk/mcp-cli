# mcp_cli/cli_options.py
import os
import json
import logging
from pathlib import Path

def load_config(config_file):
    """Load the configuration file."""
    config = None
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            logging.warning(f"Config file '{config_file}' not found.")
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in config file '{config_file}'")
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
    
    return config

def extract_server_names(config, specified_servers=None):
    """
    Extract server names from the config.
    
    Args:
        config: Configuration dictionary
        specified_servers: Optional list of specific servers to use
        
    Returns:
        Dictionary mapping server indices to their names
    """
    server_names = {}
    
    # Return empty dict if no config
    if not config or "mcpServers" not in config:
        return server_names
    
    # Get the list of servers from config
    mcp_servers = config["mcpServers"]
    
    # If specific servers were requested, map them in order
    if specified_servers:
        for i, name in enumerate(specified_servers):
            if name in mcp_servers:
                server_names[i] = name
    else:
        # Map all servers to their indices
        for i, name in enumerate(mcp_servers.keys()):
            server_names[i] = name
    
    return server_names

def process_options(server, disable_filesystem, provider, model, config_file="server_config.json"):
    """
    Process CLI options to produce a list of server names and set environment variables.
    """
    servers_list = []
    user_specified = []
    server_names = {}
    
    # Add debug logging
    logging.debug(f"Processing options: server={server}, disable_filesystem={disable_filesystem}")
    
    if server:
        # Allow comma-separated servers.
        user_specified = [s.strip() for s in server.split(",")]
        logging.debug(f"Parsed server parameter into: {user_specified}")
        servers_list.extend(user_specified)
    
    # Add more logging
    logging.debug(f"Final servers list: {servers_list}")
        
    # Use a default model if none is provided.
    if not model:
        model = "gpt-4o-mini" if provider.lower() == "openai" else "qwen2.5-coder"
    
    # Set environment variables used by the MCP code.
    os.environ["LLM_PROVIDER"] = provider
    os.environ["LLM_MODEL"] = model
    if not disable_filesystem:
        os.environ["SOURCE_FILESYSTEMS"] = json.dumps([os.getcwd()])
    
    # Load configuration to get server names
    config = load_config(config_file)
    
    # Extract server names from the configuration
    server_names = extract_server_names(config, user_specified)
    
    return servers_list, user_specified, server_names