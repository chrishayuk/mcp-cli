# src/mcp_cli/commands/impl/ping.py
"""
Unified ping command implementation.
"""

from __future__ import annotations

import time
from typing import List

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandParameter,
    CommandResult,
)
from mcp_cli.context import get_context
from chuk_term.ui import output


class PingCommand(UnifiedCommand):
    """Test connectivity to MCP servers."""
    
    @property
    def name(self) -> str:
        return "ping"
    
    @property
    def aliases(self) -> List[str]:
        return []
    
    @property
    def description(self) -> str:
        return "Test connectivity to MCP servers"
    
    @property
    def help_text(self) -> str:
        return """
Test connectivity to MCP servers.

Usage:
  /ping [server_index]    - Test server connectivity (chat mode)
  ping [server_index]     - Test server connectivity (interactive mode)
  mcp-cli ping            - Test all servers (CLI mode)
  
Options:
  --all     - Test all servers
  --timeout - Timeout in seconds (default: 5)

Examples:
  /ping           - Test all servers
  /ping 0         - Test first server
  ping --all      - Test all servers
"""
    
    @property
    def parameters(self) -> List[CommandParameter]:
        return [
            CommandParameter(
                name="server_index",
                type=int,
                required=False,
                help="Index of server to ping (omit for all)",
            ),
            CommandParameter(
                name="all",
                type=bool,
                default=False,
                help="Test all servers",
                is_flag=True,
            ),
            CommandParameter(
                name="timeout",
                type=float,
                default=5.0,
                help="Timeout in seconds",
            ),
        ]
    
    async def execute(self, **kwargs) -> CommandResult:
        """Execute the ping command."""
        # Get tool manager
        tool_manager = kwargs.get("tool_manager")
        if not tool_manager:
            try:
                context = get_context()
                if context:
                    tool_manager = context.tool_manager
            except:
                pass
        
        if not tool_manager:
            return CommandResult(
                success=False,
                error="No active tool manager. Please connect to a server first.",
            )
        
        # Get parameters
        server_index = kwargs.get("server_index")
        test_all = kwargs.get("all", False)
        timeout = kwargs.get("timeout", 5.0)
        
        # Handle positional argument
        if server_index is None and "args" in kwargs:
            args_val = kwargs["args"]
            if isinstance(args_val, list) and args_val:
                try:
                    server_index = int(args_val[0])
                except (ValueError, TypeError):
                    pass
            elif isinstance(args_val, str):
                try:
                    server_index = int(args_val)
                except (ValueError, TypeError):
                    pass
        
        # If no specific server and not --all, test all
        if server_index is None and not test_all:
            test_all = True
        
        try:
            results = []
            
            if test_all:
                # Get all servers
                server_info = await tool_manager.get_server_info()
                for i, server in enumerate(server_info):
                    server_name = getattr(server, "name", f"server-{i}")
                    
                    # Test ping
                    start_time = time.perf_counter()
                    try:
                        success = await tool_manager.ping_server(i, timeout=timeout)
                        elapsed = (time.perf_counter() - start_time) * 1000
                        
                        if success:
                            results.append(f"✓ {server_name}: {elapsed:.1f}ms")
                        else:
                            results.append(f"✗ {server_name}: Failed")
                    except Exception as e:
                        results.append(f"✗ {server_name}: Error - {str(e)}")
            else:
                # Test specific server
                start_time = time.perf_counter()
                try:
                    success = await tool_manager.ping_server(server_index, timeout=timeout)
                    elapsed = (time.perf_counter() - start_time) * 1000
                    
                    if success:
                        results.append(f"✓ Server {server_index}: {elapsed:.1f}ms")
                    else:
                        results.append(f"✗ Server {server_index}: Failed")
                except Exception as e:
                    results.append(f"✗ Server {server_index}: Error - {str(e)}")
            
            # Format output
            output_text = "Ping Results:\n" + "\n".join(results)
            
            # Determine overall success
            success = any("✓" in r for r in results)
            
            return CommandResult(
                success=success,
                output=output_text,
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to ping servers: {str(e)}",
            )