# mcp_cli/stream_manager.py
"""
StreamManager module for centralized management of server streams.

This module provides a centralized way to:
1. Initialize and maintain server connections
2. Fetch tools and resources
3. Ensure proper cleanup of streams and resources
4. Handle connection errors gracefully
"""
import asyncio
import logging
import gc
import json
from contextlib import asynccontextmanager
from typing import Dict, List, Tuple, Any, Optional, Set

# mcp imports
from chuk_mcp.mcp_client.transport.stdio.stdio_client import stdio_client
from chuk_mcp.mcp_client.messages.initialize.send_messages import send_initialize
from chuk_mcp.mcp_client.messages.tools.send_messages import send_tools_list, send_tools_call

# Use our own config loader
from mcp_cli.config import load_config

class StreamManager:
    """
    Centralized manager for server streams and connections.
    
    This class handles initialization, maintenance, and cleanup of server
    connections, ensuring proper resource management across the application.
    """
    
    def __init__(self):
        """Initialize the StreamManager."""
        self.streams = []
        self.client_contexts = []
        self.server_info = []
        self.tools = []
        self.tool_to_server_map = {}
        self.server_names = {}
        self.server_streams_map = {}  # Maps server names to stream indices
        self.active_subprocesses = set()

    @classmethod
    async def create(cls, config_file: str, servers: List[str], 
                    server_names: Optional[Dict[int, str]] = None) -> 'StreamManager':
        """
        Create and initialize a StreamManager instance.
        
        This is a convenience method that creates a new instance and initializes it.
        
        Args:
            config_file: Path to the configuration file
            servers: List of server names to connect to
            server_names: Optional dictionary mapping server indices to friendly names
            
        Returns:
            An initialized StreamManager instance
        """
        manager = cls()
        await manager.initialize_servers(config_file, servers, server_names)
        return manager
        
    async def initialize_servers(self, config_file: str, servers: List[str], 
                                server_names: Optional[Dict[int, str]] = None) -> bool:
        """
        Initialize connections to the specified servers.
        
        Args:
            config_file: Path to the configuration file
            servers: List of server names to connect to
            server_names: Optional dictionary mapping server indices to friendly names
            
        Returns:
            bool: True if at least one server was successfully initialized
        """
        self.server_names = server_names or {}
        tool_index = 0
        
        for i, server_name in enumerate(servers):
            try:
                # Get the display name for this server
                server_display_name = self._get_server_display_name(i, server_name)
                
                logging.info(f"Initializing server: {server_display_name}")
                
                # Load the server configuration
                server_params = await load_config(config_file, server_name)
                
                # Create the stdio client context manager and add it to our tracking list
                client_ctx = stdio_client(server_params)
                self.client_contexts.append(client_ctx)
                
                # Enter the context to get read_stream and write_stream
                read_stream, write_stream = await client_ctx.__aenter__()
                
                # Send the initialize message
                init_success = await send_initialize(read_stream, write_stream)
                if not init_success:
                    logging.error(f"Failed to initialize server {server_display_name}")
                    await client_ctx.__aexit__(None, None, None)
                    self.client_contexts.remove(client_ctx)
                    
                    # Add failed server to server_info
                    self.server_info.append({
                        "id": i+1,
                        "name": server_display_name,
                        "tools": 0,
                        "status": "Failed to initialize",
                        "tool_start_index": tool_index
                    })
                    continue
                
                # Fetch tools from this server
                fetched_tools = await send_tools_list(read_stream, write_stream)
                tools = fetched_tools.get("tools", [])
                
                # Map each tool to its server
                for tool in tools:
                    self.tool_to_server_map[tool["name"]] = server_display_name
                
                # Store the stream index in the map
                self.server_streams_map[server_display_name] = len(self.streams)
                
                # Store the streams and update server info
                self.streams.append((read_stream, write_stream))
                self.tools.extend(tools)
                
                # Track the connection info
                self.server_info.append({
                    "id": i+1,
                    "name": server_display_name,
                    "tools": len(tools),
                    "status": "Connected",
                    "tool_start_index": tool_index
                })
                
                tool_index += len(tools)
                logging.info(f"Successfully initialized server: {server_display_name}")
                
            except Exception as e:
                # Use the proper server name in error message
                server_display_name = self._get_server_display_name(i, server_name)
                
                # Log the error
                logging.error(f"Error initializing server {server_display_name}: {e}")
                
                # Add to server info with error status
                self.server_info.append({
                    "id": i+1,
                    "name": server_display_name,
                    "tools": 0,
                    "status": f"Error: {str(e)}",
                    "tool_start_index": tool_index
                })
            
            # Collect any subprocesses created during initialization
            self._collect_subprocesses()
        
        # Return success if we have at least one stream
        return len(self.streams) > 0
    
    def _get_server_display_name(self, index: int, server_name: str) -> str:
        """Get the display name for a server based on index or custom mapping."""
        if isinstance(self.server_names, dict) and index in self.server_names:
            return self.server_names[index]
        elif isinstance(self.server_names, list) and index < len(self.server_names):
            return self.server_names[index]
        else:
            # If no mapping, use the server name or a generic name
            return server_name or f"Server {index+1}"
    
    def _collect_subprocesses(self) -> None:
        """Collect all active subprocess.Popen instances for tracking."""
        from subprocess import Popen
        for obj in gc.get_objects():
            if isinstance(obj, Popen) and obj.poll() is None:  # Still running
                if obj not in self.active_subprocesses:
                    self.active_subprocesses.add(obj)
    
    async def call_tool(self, tool_name: str, arguments: Any, server_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Call a tool with the given name and arguments.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            server_name: Optional server name to target (if None, uses the tool_to_server_map)
            
        Returns:
            The tool response
        """
        # If no server_name provided, look it up in the tool_to_server_map
        if server_name is None or server_name == "Unknown":
            server_name = self.tool_to_server_map.get(tool_name, None)
            
        if not server_name:
            # If we still don't have a server name, try all servers
            return await self._call_tool_on_all_servers(tool_name, arguments)
        
        # Look up the server index
        server_index = self.server_streams_map.get(server_name)
        if server_index is None:
            logging.warning(f"Server '{server_name}' not found in stream map. Trying all servers.")
            return await self._call_tool_on_all_servers(tool_name, arguments)
        
        # Get the stream for this server
        if server_index >= len(self.streams):
            logging.error(f"Invalid server index: {server_index}")
            raise ValueError(f"Invalid server index: {server_index}")
        
        # Get the streams
        read_stream, write_stream = self.streams[server_index]
        
        # Call the tool
        try:
            # Ensure arguments are properly formatted
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logging.warning(f"Could not parse arguments as JSON: {arguments}")
                    # Keep as string if it's not valid JSON
            
            # Call the tool
            result = await send_tools_call(
                read_stream=read_stream,
                write_stream=write_stream,
                name=tool_name,
                arguments=arguments
            )
            
            # Check for errors
            if result.get("isError"):
                logging.error(f"Error calling tool {tool_name}: {result.get('error')}")
                return {
                    "isError": True,
                    "error": result.get("error", "Unknown error"),
                    "content": f"Error: {result.get('error', 'Unknown error')}"
                }
            
            return result
        except Exception as e:
            logging.error(f"Exception calling tool {tool_name}: {e}")
            return {
                "isError": True,
                "error": str(e),
                "content": f"Error: {str(e)}"
            }
    
    async def _call_tool_on_all_servers(self, tool_name: str, arguments: Any) -> Dict[str, Any]:
        """Try calling a tool on all available servers until one succeeds."""
        for i, (read_stream, write_stream) in enumerate(self.streams):
            try:
                # Ensure arguments are properly formatted
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        # Keep as string if it's not valid JSON
                        pass
                
                # Call the tool
                result = await send_tools_call(
                    read_stream=read_stream,
                    write_stream=write_stream,
                    name=tool_name,
                    arguments=arguments
                )
                
                # Return on first successful call
                if not result.get("isError"):
                    return result
            except Exception as e:
                logging.debug(f"Error calling tool {tool_name} on server {i}: {e}")
                # Continue trying other servers
        
        # If we get here, all servers failed
        return {
            "isError": True,
            "error": f"Tool '{tool_name}' not found on any server or all calls failed",
            "content": f"Error: Tool '{tool_name}' not found on any server or all calls failed"
        }
    
    async def close(self) -> None:
        """
        Close all streams and clean up resources.
        
        This should be called when the application is shutting down to ensure
        proper resource cleanup and prevent leaks.
        """
        logging.debug("Closing StreamManager resources")
        
        # 1. Close all client contexts
        for ctx in self.client_contexts:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception as e:
                logging.debug(f"Error closing client context: {e}")
        
        # 2. Terminate all tracked subprocesses
        for proc in self.active_subprocesses:
            try:
                if proc.poll() is None:  # Process still running
                    proc.terminate()
                    try:
                        proc.wait(timeout=0.5)  # Short timeout
                    except:
                        proc.kill()  # Force kill if terminate doesn't work
            except Exception as e:
                logging.debug(f"Error cleaning up subprocess: {e}")
        
        # 3. Clean up transports in the event loop
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                # Clean up transports
                for transport in getattr(loop, '_transports', set()):
                    if hasattr(transport, 'close'):
                        try:
                            transport.close()
                        except Exception:
                            pass
                        
                # Find and clean up subprocess transports specifically
                for obj in gc.get_objects():
                    if hasattr(obj, '__class__') and 'SubprocessTransport' in obj.__class__.__name__:
                        # Disable the pipe to prevent EOF writing
                        if hasattr(obj, '_protocol') and obj._protocol is not None:
                            if hasattr(obj._protocol, 'pipe'):
                                obj._protocol.pipe = None
        except Exception as e:
            logging.debug(f"Error cleaning up transports: {e}")
        
        # 4. Clear references
        self.streams.clear()
        self.client_contexts.clear()
        self.active_subprocesses.clear()
        self.server_streams_map.clear()
        
        # 5. Force garbage collection
        gc.collect()
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all tools from all servers."""
        return self.tools

    def get_server_info(self) -> List[Dict[str, Any]]:
        """Get information about all servers."""
        return self.server_info

    def get_server_for_tool(self, tool_name: str) -> str:
        """Get the server name that a tool belongs to."""
        return self.tool_to_server_map.get(tool_name, "Unknown")

    def has_tools(self) -> bool:
        """Check if any tools are available from any server."""
        return len(self.tools) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert StreamManager state to a dictionary (for context sharing)."""
        return {
            "server_info": self.server_info,
            "tools": self.tools,
            "tool_to_server_map": self.tool_to_server_map,
            "server_names": self.server_names
        }

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update manager state from a dictionary (for context sharing)."""
        # Only update fields that shouldn't change the core stream management
        if "tool_to_server_map" in data:
            self.tool_to_server_map = data["tool_to_server_map"]
        if "server_names" in data:
            self.server_names = data["server_names"]