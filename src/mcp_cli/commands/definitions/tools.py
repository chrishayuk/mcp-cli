# src/mcp_cli/commands/definitions/tools.py
"""
Unified tools command implementation with subcommands.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandGroup,
    CommandMode,
    CommandParameter,
    CommandResult,
)
from mcp_cli.context import get_context
from mcp_cli.ui.formatting import create_tools_table
from chuk_term.ui import output, format_table


class ToolsCommand(CommandGroup):
    """Tools command group."""
    
    def __init__(self):
        super().__init__()
        # Add subcommands
        self.add_subcommand(ToolsListCommand())
        self.add_subcommand(ToolsCallCommand())
        self.add_subcommand(ToolsConfirmCommand())
    
    @property
    def name(self) -> str:
        return "tools"
    
    @property
    def aliases(self) -> List[str]:
        return []
    
    @property
    def description(self) -> str:
        return "Manage and interact with MCP tools"
    
    @property
    def help_text(self) -> str:
        return """
Manage and interact with MCP tools.

Subcommands:
  list     - List all available tools
  call     - Call a specific tool
  confirm  - Configure tool confirmation settings

Usage:
  /tools list              - List all tools (chat mode)
  tools list               - List all tools (interactive mode)
  mcp-cli tools list       - List all tools (CLI mode)
  
  /tools call <name>       - Call a tool
  /tools confirm [mode]    - Set confirmation mode

Examples:
  /tools list --raw        - Show tools as JSON
  /tools list --details    - Show detailed parameter info
  /tools call read_file --args '{"path": "README.md"}'
  /tools confirm always   - Always confirm tool execution
"""


class ToolsListCommand(UnifiedCommand):
    """List available tools."""
    
    @property
    def name(self) -> str:
        return "list"
    
    @property
    def aliases(self) -> List[str]:
        return ["ls", "show"]
    
    @property
    def description(self) -> str:
        return "List all available MCP tools"
    
    @property
    def help_text(self) -> str:
        return """
List all available MCP tools from connected servers.

Usage:
  /tools list [options]
  
Options:
  --details    - Show parameter schemas
  --raw        - Output as JSON
  --validation - Show validation status
  --all        - Show all details

Examples:
  /tools list
  /tools list --raw
  /tools list --details --validation
"""
    
    @property
    def parameters(self) -> List[CommandParameter]:
        return [
            CommandParameter(
                name="details",
                type=bool,
                default=False,
                help="Show parameter schemas",
                is_flag=True,
            ),
            CommandParameter(
                name="raw",
                type=bool,
                default=False,
                help="Output as JSON",
                is_flag=True,
            ),
            CommandParameter(
                name="validation",
                type=bool,
                default=False,
                help="Show validation status",
                is_flag=True,
            ),
            CommandParameter(
                name="all",
                type=bool,
                default=False,
                help="Show all details",
                is_flag=True,
            ),
        ]
    
    async def execute(self, **kwargs) -> CommandResult:
        """Execute the tools list command."""
        # Import the tools action from the actions module
        from mcp_cli.commands.actions.tools import tools_action_async
        
        # Extract parameters
        show_details = kwargs.get("details", False) or kwargs.get("all", False)
        show_raw = kwargs.get("raw", False)
        show_validation = kwargs.get("validation", False) or kwargs.get("all", False)
        
        try:
            # Use the existing enhanced implementation
            # It handles all the display internally
            tools = await tools_action_async(
                show_details=show_details,
                show_raw=show_raw,
                show_validation=show_validation,
            )
            
            # The existing implementation handles all output directly
            # Just return success
            return CommandResult(
                success=True,
                data=tools
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to list tools: {str(e)}",
            )


class ToolsCallCommand(UnifiedCommand):
    """Call a specific tool."""
    
    @property
    def name(self) -> str:
        return "call"
    
    @property
    def aliases(self) -> List[str]:
        return ["run", "execute"]
    
    @property
    def description(self) -> str:
        return "Call a specific MCP tool"
    
    @property
    def help_text(self) -> str:
        return """
Call a specific MCP tool with arguments.

Usage:
  /tools call <tool_name> [options]
  
Options:
  --args <json>  - Tool arguments as JSON
  --confirm      - Confirm before execution

Examples:
  /tools call read_file --args '{"path": "README.md"}'
  /tools call list_directory --args '{"path": "."}'
  /tools call search --args '{"query": "test"}' --confirm
"""
    
    @property
    def parameters(self) -> List[CommandParameter]:
        return [
            CommandParameter(
                name="tool_name",
                type=str,
                required=True,
                help="Name of the tool to call",
            ),
            CommandParameter(
                name="args",
                type=str,
                default="{}",
                help="Tool arguments as JSON string",
            ),
            CommandParameter(
                name="confirm",
                type=bool,
                default=False,
                help="Confirm before execution",
                is_flag=True,
            ),
        ]
    
    async def execute(self, **kwargs) -> CommandResult:
        """Execute the tool call command."""
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
        
        # Get tool name from positional args or parameter
        tool_name = kwargs.get("tool_name")
        if not tool_name and "args" in kwargs and isinstance(kwargs["args"], (list, str)):
            # Handle positional argument
            args_val = kwargs["args"]
            if isinstance(args_val, list):
                tool_name = args_val[0] if args_val else None
            else:
                tool_name = args_val
        
        if not tool_name:
            return CommandResult(
                success=False,
                error="Tool name is required. Usage: /tools call <tool_name>",
            )
        
        # Parse tool arguments
        args_json = kwargs.get("args", "{}")
        if args_json == tool_name:  # If args wasn't provided separately
            args_json = "{}"
        
        try:
            tool_args = json.loads(args_json) if isinstance(args_json, str) else args_json
        except json.JSONDecodeError as e:
            return CommandResult(
                success=False,
                error=f"Invalid JSON arguments: {e}",
            )
        
        # Confirm if requested
        if kwargs.get("confirm", False):
            from chuk_term.ui.prompts import confirm
            if not confirm(f"Execute tool '{tool_name}' with args: {tool_args}?"):
                return CommandResult(
                    success=True,
                    output="Tool execution cancelled.",
                )
        
        try:
            # Execute the tool
            result = await tool_manager.execute_tool(
                tool_name=tool_name,
                arguments=tool_args,
            )
            
            # Format result
            if isinstance(result, dict):
                output_text = json.dumps(result, indent=2, default=str)
            else:
                output_text = str(result)
            
            return CommandResult(
                success=True,
                output=f"Tool '{tool_name}' executed successfully:\n{output_text}",
            )
            
        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to execute tool '{tool_name}': {str(e)}",
            )


class ToolsConfirmCommand(UnifiedCommand):
    """Configure tool confirmation settings."""
    
    @property
    def name(self) -> str:
        return "confirm"
    
    @property
    def aliases(self) -> List[str]:
        return ["confirmation"]
    
    @property
    def description(self) -> str:
        return "Configure tool execution confirmation"
    
    @property
    def help_text(self) -> str:
        return """
Configure when to confirm tool execution.

Usage:
  /tools confirm [mode]
  
Modes:
  always  - Always ask for confirmation
  never   - Never ask for confirmation
  smart   - Ask based on risk assessment (default)

Examples:
  /tools confirm          - Show current mode
  /tools confirm always   - Always confirm
  /tools confirm never    - Never confirm
  /tools confirm smart    - Smart confirmation
"""
    
    @property
    def parameters(self) -> List[CommandParameter]:
        return [
            CommandParameter(
                name="mode",
                type=str,
                required=False,
                help="Confirmation mode",
                choices=["always", "never", "smart"],
            ),
        ]
    
    @property
    def modes(self) -> CommandMode:
        """This is primarily for chat and interactive modes."""
        return CommandMode.CHAT | CommandMode.INTERACTIVE
    
    async def execute(self, **kwargs) -> CommandResult:
        """Execute the confirm command."""
        from mcp_cli.utils.preferences import get_preference_manager
        
        pref_manager = get_preference_manager()
        mode = kwargs.get("mode")
        
        # Handle positional argument
        if not mode and "args" in kwargs:
            args_val = kwargs["args"]
            if isinstance(args_val, list):
                mode = args_val[0] if args_val else None
            elif isinstance(args_val, str):
                mode = args_val
        
        if mode:
            # Set new mode
            if mode not in ["always", "never", "smart"]:
                return CommandResult(
                    success=False,
                    error=f"Invalid mode: {mode}. Must be 'always', 'never', or 'smart'.",
                )
            
            pref_manager.set_tool_confirmation_mode(mode)
            
            return CommandResult(
                success=True,
                output=f"Tool confirmation mode set to: {mode}",
            )
        else:
            # Show current mode
            current_mode = pref_manager.get_tool_confirmation_mode()
            return CommandResult(
                success=True,
                output=f"Current tool confirmation mode: {current_mode}",
            )