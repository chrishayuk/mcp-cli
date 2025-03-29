# src/cli/chat/commands/tool_history.py
"""
Tool history command module for displaying executed tool calls in the current session.
"""
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
import json
import traceback

# Import the registration function
from mcp_cli.chat.commands import register_command

async def tool_history_command(args, context):
    """
    Display history of executed tool calls in the current chat session.
    
    Usage:
      /toolhistory         - Show all tool calls in the current session.
      /toolhistory -n 5    - Show only the last 5 tool calls.
      /toolhistory --json  - Show tool calls in JSON format.
      /toolhistory <row>   - Show full details for a specific tool call (e.g., /toolhistory 1).
    """
    console = Console()
    
    try:
        # Gather all tool calls from conversation history.
        conversation_history = context.get("conversation_history", [])
        all_tool_calls = []
        
        # Scan conversation history for tool calls (assumed to be in messages with role "assistant")
        for msg in conversation_history:
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls", [])
            for tool_call in tool_calls:
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
                if isinstance(raw_arguments, str):
                    try:
                        raw_arguments = json.loads(raw_arguments)
                    except json.JSONDecodeError:
                        pass
                all_tool_calls.append({
                    "name": tool_name,
                    "args": raw_arguments
                })
        
        if not all_tool_calls:
            console.print("[italic yellow]No tool calls have been recorded in this session.[/italic yellow]")
            return True
        
        # Parse arguments - skip the command name itself
        clean_args = args[1:] if args else []
        
        # If first argument is a number, display that specific tool call in full.
        if clean_args and clean_args[0].isdigit():
            row_number = int(clean_args[0])
            if 1 <= row_number <= len(all_tool_calls):
                tool_entry = all_tool_calls[row_number - 1]
                console.print(
                    Panel(
                        Syntax(json.dumps(tool_entry, indent=2), "json", theme="monokai", line_numbers=True),
                        title=f"Tool Call #{row_number} Details",
                        style="cyan"
                    )
                )
            else:
                console.print(f"[red]Invalid row number. Please enter a number between 1 and {len(all_tool_calls)}.[/red]")
            return True

        # Check for --json flag.
        if "--json" in clean_args:
            raw_json = json.dumps(all_tool_calls, indent=2)
            console.print(Syntax(raw_json, "json", theme="monokai", line_numbers=True))
            return True
        
        # Optionally support limiting results with -n flag.
        limit = None
        start_index = 0
        if "-n" in clean_args:
            try:
                n_index = clean_args.index("-n")
                if n_index + 1 < len(clean_args):
                    limit = int(clean_args[n_index + 1])
            except (ValueError, IndexError):
                console.print("[bold red]Invalid -n argument. Showing all tool calls.[/bold red]")
        
        # Apply filtering based on limit
        filtered_tool_calls = all_tool_calls
        if limit is not None and limit > 0:
            filtered_tool_calls = all_tool_calls[-limit:]
        
        # Otherwise, display a summary table.
        table = Table(title=f"Tool Call History ({len(filtered_tool_calls)} calls)")
        table.add_column("#", style="dim")
        table.add_column("Tool", style="green")
        table.add_column("Arguments", style="yellow")

        for i, tool in enumerate(filtered_tool_calls, start=len(all_tool_calls) - len(filtered_tool_calls) + 1):
            tool_name = tool.get("name", "unknown")
            args_str = json.dumps(tool.get("args", {}))
            if len(args_str) > 80:
                args_str = args_str[:77] + "..."
            table.add_row(str(i), tool_name, args_str)
        
        console.print(table)
    
    except Exception as e:
        # Print exception for debugging
        console.print(f"[bold red]ERROR: An exception occurred:[/bold red]")
        console.print(f"[red]{traceback.format_exc()}[/red]")
    
    return True

# Register the command with aliases.
register_command("/toolhistory", tool_history_command, ["-n", "--json"])
register_command("/th", tool_history_command, ["-n", "--json"])