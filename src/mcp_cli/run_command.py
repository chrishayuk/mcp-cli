# mcp_cli/run_command.py
"""
Utilities for running commands with proper setup and cleanup.
"""
import asyncio
import logging
import time
from typing import Callable, List, Dict, Any, Optional

# Import our StreamManager
from mcp_cli.stream_manager import StreamManager

async def run_command_async(command_func, config_file, servers, user_specified, extra_params=None):
    """
    Run a command with proper setup and cleanup.
    
    Args:
        command_func: The command function to run.
        config_file: Path to the configuration file.
        servers: List of server names to connect to.
        user_specified: List of servers specified by the user.
        extra_params: Optional dictionary of additional parameters to pass to the command function.
        
    Returns:
        The result of the command function.
    """
    logging.info(f"Running command: {command_func.__name__}")
    logging.info(f"Servers: {servers}")
    
    if not servers:
        logging.warning("No servers specified!")
        return False
        
    logging.info(f"Initializing servers: {servers}")
    
    # Create a stream manager to handle server connections
    stream_manager = await StreamManager.create(
        config_file=config_file,
        servers=servers,
        server_names={i: name for i, name in enumerate(servers)} if servers else None
    )
    
    try:
        # Initialize extra_params if None
        if extra_params is None:
            extra_params = {}
            
        # Add stream_manager to extra_params
        extra_params["stream_manager"] = stream_manager
        
        # Run the command with the stream manager
        return await command_func(**extra_params)
    finally:
        # Ensure streams are properly closed
        await stream_manager.close()

def run_command(command_func, config_file, servers, user_specified, extra_params=None):
    """
    Synchronous wrapper for run_command_async.
    
    Args:
        command_func: The command function to run.
        config_file: Path to the configuration file.
        servers: List of server names to connect to.
        user_specified: List of servers specified by the user.
        extra_params: Optional dictionary of additional parameters to pass to the command function.
        
    Returns:
        The result of the command function.
    """
    try:
        # Get the event loop
        loop = asyncio.get_event_loop()
        
        # Log the command being run
        logging.info(f"Starting command: {command_func.__name__}")
        
        # Run the command
        return loop.run_until_complete(
            run_command_async(command_func, config_file, servers, user_specified, extra_params)
        )
    except KeyboardInterrupt:
        logging.debug("KeyboardInterrupt received")
        return False
    except Exception as e:
        logging.error(f"Error running command: {e}")
        return False