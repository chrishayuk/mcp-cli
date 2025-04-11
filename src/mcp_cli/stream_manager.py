# mcp_cli/stream_manager.py
"""
StreamManager module for centralized management of server streams.

This module provides a centralized way to:
1. Initialize and maintain server connections
2. Fetch tools and resources
3. Ensure proper cleanup of streams and resources
4. Handle connection errors gracefully
5. Handle duplicate tool names across servers automatically
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
        self.tools = []  # Display tools (with original names)
        self.internal_tools = []  # Internal tools (with namespaced names)
        self.tool_to_server_map = {}  # Maps display tool names to server names
        self.namespaced_tool_map = {}  # Maps namespaced tool names to original names
        self.original_to_namespaced = {}  # Maps original tool names to namespaced names (possibly multiple)
        self.original_to_default = {}    # Maps original tool names to default namespaced name
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
                
                # Process tools to handle duplicates
                display_tools = []  # For UI display (original names)
                namespaced_tools = []  # For internal use (namespaced names)
                
                for tool in tools:
                    # Create display tool (original names for UI)
                    display_tool = tool.copy()
                    original_name = tool["name"]
                    
                    # Create namespaced tool (for internal use)
                    namespaced_tool = tool.copy()
                    namespaced_name = f"{server_display_name}_{original_name}"
                    namespaced_tool["name"] = namespaced_name
                    
                    # Store mappings
                    self.tool_to_server_map[original_name] = server_display_name
                    self.namespaced_tool_map[namespaced_name] = original_name
                    
                    # Handle the case where one original name maps to multiple namespaced names
                    if original_name in self.original_to_namespaced:
                        # Append this namespaced name to the list
                        self.original_to_namespaced[original_name].append(namespaced_name)
                    else:
                        # First server with this tool, create new list
                        self.original_to_namespaced[original_name] = [namespaced_name]
                        # Also set this as the default namespaced name for this tool
                        self.original_to_default[original_name] = namespaced_name
                    
                    display_tools.append(display_tool)
                    namespaced_tools.append(namespaced_tool)
                
                # Store the stream index in the map
                self.server_streams_map[server_display_name] = len(self.streams)
                
                # Store the streams and update server info
                self.streams.append((read_stream, write_stream))
                self.tools.extend(display_tools)
                self.internal_tools.extend(namespaced_tools)
                
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
    
    def _resolve_tool_name(self, tool_name: str) -> Tuple[str, str]:
        """
        Resolve a tool name to its proper namespaced version and server.
        
        This method handles the automatic namespacing of tool names:
        1. If the tool name is already namespaced, use it as is
        2. If the tool name exists on multiple servers, use the default namespaced version
        3. If the tool name exists on only one server, use that server's namespaced version
        
        Args:
            tool_name: The original or namespaced tool name
            
        Returns:
            Tuple of (resolved_tool_name, server_name)
        """
        # Check if this is already a namespaced tool name
        if tool_name in self.namespaced_tool_map:
            # Extract server name from the namespaced tool
            parts = tool_name.split('_', 1)
            server_name = parts[0] if len(parts) > 1 else "Unknown"
            return tool_name, server_name
            
        # Check if this is an original tool name
        if tool_name in self.original_to_namespaced:
            # Get all namespaced versions
            namespaced_versions = self.original_to_namespaced[tool_name]
            
            if len(namespaced_versions) > 1:
                # Use the default (first server we saw with this tool)
                default_namespaced = self.original_to_default[tool_name]
                parts = default_namespaced.split('_', 1)
                server_name = parts[0] if len(parts) > 1 else "Unknown"
                logging.debug(f"Tool '{tool_name}' exists on multiple servers. Using default: {default_namespaced}")
                return default_namespaced, server_name
            else:
                # Only one server has this tool
                namespaced_name = namespaced_versions[0]
                parts = namespaced_name.split('_', 1)
                server_name = parts[0] if len(parts) > 1 else "Unknown"
                return namespaced_name, server_name
                
        # Tool name not found
        return tool_name, "Unknown"
    
    async def call_tool(self, tool_name: str, arguments: Any, server_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Call a tool with the given name and arguments.
        
        Args:
            tool_name: Name of the tool to call (can be original or namespaced)
            arguments: Arguments to pass to the tool
            server_name: Optional server name to target (if None, uses mapping)
            
        Returns:
            The tool response
        """
        # Automatically resolve the tool name to its proper namespaced version
        original_tool_name = tool_name  # Keep original for error messages
        
        # If server_name is provided, try to construct and verify a namespaced name
        if server_name and server_name != "Unknown":
            # Try to construct a namespaced tool name from server and tool
            constructed_name = f"{server_name}_{tool_name}"
            if constructed_name in self.namespaced_tool_map:
                # This is a valid namespaced tool name
                tool_name = constructed_name
            else:
                # Check if tool_name is already a namespaced name
                if tool_name not in self.namespaced_tool_map:
                    # Not a valid namespaced name, use _resolve_tool_name
                    tool_name, server_name = self._resolve_tool_name(tool_name)
        else:
            # No server name provided, resolve automatically
            tool_name, server_name = self._resolve_tool_name(tool_name)
        
        logging.debug(f"Resolved tool name '{original_tool_name}' to '{tool_name}' on server '{server_name}'")
            
        # Check if we have a valid server name
        if server_name == "Unknown":
            logging.warning(f"Unknown server for tool '{original_tool_name}'")
            return {
                "isError": True,
                "error": f"Tool '{original_tool_name}' not found on any server",
                "content": f"Error: Tool '{original_tool_name}' not found on any server"
            }
        
        # Look up the server index
        server_index = self.server_streams_map.get(server_name)
        if server_index is None:
            logging.error(f"Server '{server_name}' not found in stream map")
            return {
                "isError": True,
                "error": f"Server '{server_name}' not found",
                "content": f"Error: Server '{server_name}' not found"
            }
        
        # Get the stream for this server
        if server_index >= len(self.streams):
            logging.error(f"Invalid server index: {server_index}")
            return {
                "isError": True,
                "error": f"Invalid server index: {server_index}",
                "content": f"Error: Invalid server index: {server_index}"
            }
        
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
            
            # When making the actual call, use the original tool name if this is a namespaced tool
            # as the server expects the original name
            tool_to_call = original_tool_name
            if tool_name in self.namespaced_tool_map:
                tool_to_call = self.namespaced_tool_map[tool_name]
            
            logging.debug(f"Calling tool '{tool_to_call}' on server '{server_name}'")
            
            # Call the tool
            result = await send_tools_call(
                read_stream=read_stream,
                write_stream=write_stream,
                name=tool_to_call,
                arguments=arguments
            )
            
            # Check for errors
            if result.get("isError"):
                logging.error(f"Error calling tool {tool_to_call}: {result.get('error')}")
                return {
                    "isError": True,
                    "error": result.get("error", "Unknown error"),
                    "content": f"Error: {result.get('error', 'Unknown error')}"
                }
            
            return result
        except Exception as e:
            logging.error(f"Exception calling tool {original_tool_name}: {e}")
            return {
                "isError": True,
                "error": str(e),
                "content": f"Error: {str(e)}"
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
            except asyncio.CancelledError:
                logging.debug("Cancellation encountered while closing client context; continuing cleanup.")
            except Exception as e:
                logging.debug(f"Error closing client context: {e}")
        
        # 2. Terminate all tracked subprocesses
        for proc in self.active_subprocesses:
            try:
                if proc.poll() is None:  # Process still running
                    proc.terminate()
                    try:
                        proc.wait(timeout=0.5)  # Short timeout
                    except Exception:
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
        """Get all tools from all servers for display purposes."""
        return self.tools
    
    def get_internal_tools(self) -> List[Dict[str, Any]]:
        """Get all tools with namespaced names for internal use."""
        return self.internal_tools

    def get_server_info(self) -> List[Dict[str, Any]]:
        """Get information about all servers."""
        return self.server_info

    def get_server_for_tool(self, tool_name: str) -> str:
        """Get the server name that a tool belongs to."""
        # Use the resolution method which handles all cases
        _, server_name = self._resolve_tool_name(tool_name)
        return server_name

    def has_tools(self) -> bool:
        """Check if any tools are available from any server."""
        return len(self.tools) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert StreamManager state to a dictionary (for context sharing)."""
        return {
            "server_info": self.server_info,
            "tools": self.tools,
            "internal_tools": self.internal_tools,
            "tool_to_server_map": self.tool_to_server_map,
            "namespaced_tool_map": self.namespaced_tool_map,
            "original_to_namespaced": self.original_to_namespaced,
            "original_to_default": self.original_to_default,
            "server_names": self.server_names
        }

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update manager state from a dictionary (for context sharing)."""
        # Only update fields that shouldn't change the core stream management
        if "tool_to_server_map" in data:
            self.tool_to_server_map = data["tool_to_server_map"]
        if "server_names" in data:
            self.server_names = data["server_names"]
        if "namespaced_tool_map" in data:
            self.namespaced_tool_map = data["namespaced_tool_map"]
        if "original_to_namespaced" in data:
            self.original_to_namespaced = data["original_to_namespaced"]
        if "original_to_default" in data:
            self.original_to_default = data["original_to_default"]
        if "internal_tools" in data:
            self.internal_tools = data["internal_tools"]