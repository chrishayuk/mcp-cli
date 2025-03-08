# src/cli/chat/commands/tool_history.py
"""
Tool history command module for displaying executed tool calls in the current session.
"""
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
import json
import time

# Import the registration function
from cli.chat.commands import register_command


async def tool_history_command(args, context):
    """
    Display history of executed tool calls in the current chat session.
    
    Usage:
      /toolhistory       - Show all tool calls in the current session
      /toolhistory -n 5  - Show only the last 5 tool calls
      /toolhistory --json - Show tool calls in JSON format
    
    This command is particularly useful when running in compact mode
    to review all the tools that have been called during the conversation.
    """
    console = Console()
    
    # Get the UI manager from context
    ui_manager = context.get("ui_manager")
    
    # Get the conversation history from context
    conversation_history = context.get("conversation_history", [])
    
    # Get tool calls - they may be in different places depending on API
    all_tool_calls = []
    
    # If the UI manager is available, try to get current tool calls from it
    current_tool_calls = []
    current_tool_times = []
    
    if ui_manager:
        current_tool_calls = getattr(ui_manager, "tool_calls", [])
        current_tool_times = getattr(ui_manager, "tool_times", [])
    
    # Also scan conversation history for tool calls from previous turns
    for msg in conversation_history:
        # Skip non-assistant messages
        if msg.get("role") != "assistant":
            continue
            
        # Look for tool_calls in the message
        tool_calls = msg.get("tool_calls", [])
        for tool_call in tool_calls:
            # Format may vary depending on the specific API
            if hasattr(tool_call, "function"):
                tool_name = getattr(tool_call.function, "name", "unknown tool")
                raw_arguments = getattr(tool_call.function, "arguments", {})
            elif isinstance(tool_call, dict) and "function" in tool_call:
                fn_info = tool_call["function"]
                tool_name = fn_info.get("name", "unknown tool")
                raw_arguments = fn_info.get("arguments", {})
            else:
                tool_name = "unknown tool"
                raw_arguments = {}
                
            # Try to parse arguments if they're a string
            if isinstance(raw_arguments, str):
                try:
                    raw_arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    pass
                    
            # Add to our collection, but avoid duplicates with current tools
            tool_entry = {
                "name": tool_name,
                "args": raw_arguments
            }
            
            # Use a simple heuristic to avoid duplicates - check name and args
            # This isn't perfect but should catch most duplicates
            duplicate = False
            for current_tool in current_tool_calls:
                if (current_tool.get("name") == tool_name and 
                    json.dumps(current_tool.get("args")) == json.dumps(raw_arguments)):
                    duplicate = True
                    break
                    
            if not duplicate:
                all_tool_calls.append(tool_entry)
    
    # Add current tool calls at the end (they're the most recent)
    all_tool_calls.extend(current_tool_calls)
    
    # Check if there are any tool calls to display
    if not all_tool_calls:
        console.print("[italic yellow]No tool calls have been recorded in this session.[/italic yellow]")
        return True
    
    # Parse arguments
    show_json = "--json" in args
    limit = None
    
    # Check for limit argument (-n X)
    if "-n" in args:
        try:
            n_index = args.index("-n")
            if n_index + 1 < len(args):
                limit = int(args[n_index + 1])
        except (ValueError, IndexError):
            console.print("[bold red]Invalid -n argument. Using all tool calls.[/bold red]")
    
    # Apply limit if specified
    if limit is not None and limit > 0:
        all_tool_calls = all_tool_calls[-limit:]
        # Also limit tool times if they exist
        if current_tool_times:
            current_tool_times = current_tool_times[-limit:]
    
    if show_json:
        # Show raw JSON for tool calls
        raw_json = json.dumps(all_tool_calls, indent=2)
        console.print(Syntax(raw_json, "json", theme="monokai", line_numbers=True))
        return True
    
    # Create a rich table for standard view
    table = Table(title=f"Tool Call History ({len(all_tool_calls)} calls)")
    
    # Add columns
    table.add_column("#", style="dim")
    table.add_column("Tool", style="green")
    table.add_column("Arguments", style="yellow")
    
    # Add time column if we have timing info
    has_timing = len(current_tool_times) > 0
    if has_timing:
        table.add_column("Time", style="cyan")
    
    # Add rows for each tool call
    for i, tool in enumerate(all_tool_calls):
        tool_name = tool.get("name", "unknown")
        
        # Format arguments as JSON
        args = tool.get("args", {})
        args_str = json.dumps(args, indent=2)
        
        # Truncate long argument strings
        if len(args_str) > 80:
            args_str = args_str[:77] + "..."
        
        # Check if we have timing information for this tool    
        if i < len(current_tool_times) and has_timing:
            time_str = f"{current_tool_times[i]:.2f}s"
            table.add_row(str(i+1), tool_name, args_str, time_str)
        else:
            if has_timing:
                table.add_row(str(i+1), tool_name, args_str, "-")
            else:
                table.add_row(str(i+1), tool_name, args_str)
    
    console.print(table)
    
    # Add a note about viewing full arguments
    if any(len(json.dumps(tool.get("args", {}))) > 80 for tool in all_tool_calls):
        console.print("[dim]Note: Some arguments were truncated. Use --json to see full details.[/dim]")
    
    return True


# Register the command with completions
register_command("/toolhistory", tool_history_command, ["-n", "--json"])

# Add a shorter alias for convenience
register_command("/th", tool_history_command, ["-n", "--json"])