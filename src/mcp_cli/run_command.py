# mcp_cli/run_command.py
"""
Utilities for running commands with proper setup and cleanup.
"""
import asyncio
import logging
import time
from contextlib import asynccontextmanager
import gc

# mcp imports
from chuk_mcp.mcp_client.transport.stdio.stdio_client import stdio_client
from chuk_mcp.mcp_client.messages.initialize.send_messages import send_initialize

# Use mcp_cli.config instead of chuk_mcp.mcp_client.config
from mcp_cli.config import load_config

@asynccontextmanager
async def get_server_streams(config_file, servers):
    """
    Context manager for setting up and cleaning up server streams.
    
    Args:
        config_file: Path to the configuration file.
        servers: List of server names to connect to.
        
    Yields:
        List of (read_stream, write_stream) tuples for each server.
    """
    streams = []
    client_contexts = []
    
    try:
        # Initialize each server
        for server_name in servers:
            try:
                logging.info(f"Initializing server: {server_name}")
                # Load the server configuration
                server_params = await load_config(config_file, server_name)
                
                # Create the stdio client context manager and add it to our tracking list
                client_ctx = stdio_client(server_params)
                client_contexts.append(client_ctx)
                
                # Enter the context to get read_stream and write_stream
                read_stream, write_stream = await client_ctx.__aenter__()
                
                # Send the initialize message
                init_success = await send_initialize(read_stream, write_stream)
                if not init_success:
                    logging.error(f"Failed to initialize server {server_name}")
                    # Close this context since we failed
                    await client_ctx.__aexit__(None, None, None)
                    continue
                
                # Store the streams
                streams.append((read_stream, write_stream))
                logging.info(f"Successfully initialized server: {server_name}")
            except Exception as e:
                logging.error(f"Error initializing server {server_name}: {e}")
        
        if not streams:
            logging.warning("No server streams were successfully initialized!")
            
        logging.info(f"Yielding {len(streams)} server streams")
        # Yield the streams for command use
        yield streams
        
    finally:
        # Exit all client contexts
        for ctx in client_contexts:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception as e:
                logging.debug(f"Error closing client context: {e}")
        
        # Clear references
        streams.clear()
        client_contexts.clear()
        
        # Force garbage collection
        gc.collect()

async def run_command_async(command_func, config_file, servers, user_specified):
    """
    Run a command with proper setup and cleanup.
    
    Args:
        command_func: The command function to run.
        config_file: Path to the configuration file.
        servers: List of server names to connect to.
        user_specified: List of servers specified by the user.
        
    Returns:
        The result of the command function.
    """
    logging.info(f"Running command: {command_func.__name__}")
    logging.info(f"Servers: {servers}")
    
    if not servers:
        logging.warning("No servers specified!")
        return False
        
    logging.info(f"Initializing servers: {servers}")
    
    # Use the context manager to handle server streams
    async with get_server_streams(config_file, servers) as server_streams:
        if not server_streams:
            logging.warning("No server streams available! Command may not work properly.")
            
        # Run the command with the server streams
        return await command_func(server_streams)

def run_command(command_func, config_file, servers, user_specified):
    """
    Synchronous wrapper for run_command_async.
    
    Args:
        command_func: The command function to run.
        config_file: Path to the configuration file.
        servers: List of server names to connect to.
        user_specified: List of servers specified by the user.
        
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
            run_command_async(command_func, config_file, servers, user_specified)
        )
    except KeyboardInterrupt:
        logging.debug("KeyboardInterrupt received")
        return False
    except Exception as e:
        logging.error(f"Error running command: {e}")
        return False