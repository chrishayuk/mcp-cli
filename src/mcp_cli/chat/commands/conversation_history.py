# src/cli/chat/commands/conversation_history.py
import json
import traceback
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
from rich import box
from rich.text import Text

# Import the registration function from your commands package
from mcp_cli.chat.commands import register_command

async def conversation_history_command(args, context):
    """
    Display the conversation history of the current chat session.

    Usage:
      /conversation         - Show the conversation history in a tabular view.
      /conversation -n 5    - Show only the last 5 messages.
      /conversation --json  - Show the conversation history in JSON format.
      /conversation <row>   - Show details for the specified message row (e.g., /conversation 3).
      /conversation <row> --json - Show message #<row> in JSON format.
    """
    console = Console()
    
    try:
        conversation_history = context.get("conversation_history", [])
        
        if not conversation_history:
            console.print("[italic yellow]No conversation history available.[/italic yellow]")
            return True

        # Parse arguments - skip the first arg which is the command name itself
        clean_args = args[1:] if args else []
        row_specified = False
        row_number = None
        show_json = "--json" in clean_args
        limit = None
        
        # Check for row number specification - first real argument
        if clean_args and clean_args[0].isdigit():
            row_specified = True
            row_number = int(clean_args[0])
            
            if not (1 <= row_number <= len(conversation_history)):
                console.print(f"[red]Invalid row number. Please enter a number between 1 and {len(conversation_history)}.[/red]")
                return True

        # Check for limit flag
        if "-n" in clean_args:
            try:
                n_index = clean_args.index("-n")
                if n_index + 1 < len(clean_args):
                    limit = int(clean_args[n_index + 1])
            except (ValueError, IndexError):
                console.print("[bold red]Invalid -n argument. Showing all messages.[/bold red]")
        
        # Filter the history based on our arguments
        if row_specified:
            # Just show the one requested message
            filtered_history = [conversation_history[row_number - 1]]
        elif limit is not None and limit > 0:
            # Show the last N messages
            filtered_history = conversation_history[-limit:]
        else:
            # Show all messages
            filtered_history = conversation_history
        
        # Display the appropriate format
        if show_json:
            if row_specified:
                # Show just the one message as JSON with full content, not truncated
                message_json = json.dumps(filtered_history[0], indent=2, ensure_ascii=False)
                
                # Use a Panel with a larger width constraint to prevent truncation
                console.print(
                    Panel(
                        Syntax(message_json, "json", theme="monokai", word_wrap=True),
                        title=f"Message #{row_number} (JSON)",
                        border_style="cyan",
                        box=box.ROUNDED,
                        expand=True,  # Allow panel to expand to full width
                        padding=(1, 2)  # Add some padding
                    )
                )
            else:
                # Show all filtered messages as JSON
                all_json = json.dumps(filtered_history, indent=2, ensure_ascii=False)
                console.print(
                    Panel(
                        Syntax(all_json, "json", theme="monokai", word_wrap=True),
                        title="Conversation History (JSON)",
                        border_style="cyan",
                        box=box.ROUNDED,
                        expand=True,  # Allow panel to expand to full width
                        padding=(1, 2)  # Add some padding
                    )
                )
        else:
            # Display in table format
            if row_specified:
                # For a single row, display with full content in a panel
                message = filtered_history[0]
                original_index = row_number
                
                # Format role
                role = message.get("role", "unknown")
                name = message.get("name", "")
                if name:
                    role = f"{role} ({name})"
                
                # Get full content without truncation
                content = message.get("content", "")
                if content is None:
                    if "tool_calls" in message and message["tool_calls"]:
                        tool_calls = message["tool_calls"]
                        tool_names = []
                        for tc in tool_calls:
                            if "function" in tc:
                                tool_names.append(tc["function"].get("name", "unknown"))
                        content = f"[Tool call: {', '.join(tool_names)}]"
                    else:
                        content = "[None]"
                
                # Format tool call info if present
                tool_calls_info = ""
                if "tool_calls" in message and message["tool_calls"]:
                    tool_calls = message["tool_calls"]
                    tool_calls_info = "\n\nTool Calls:\n"
                    for i, tool_call in enumerate(tool_calls):
                        tool_id = tool_call.get("id", "unknown_id")
                        tool_type = tool_call.get("type", "unknown_type")
                        
                        if "function" in tool_call:
                            fn_info = tool_call["function"]
                            tool_name = fn_info.get("name", "unknown")
                            args_str = fn_info.get("arguments", "{}")
                            
                            tool_calls_info += f"  {i+1}. ID: {tool_id}, Type: {tool_type}, Name: {tool_name}\n"
                            tool_calls_info += f"     Arguments: {args_str}\n"
                
                # Add tool call ID for tool messages
                tool_call_id_info = ""
                if role == "tool" and "tool_call_id" in message:
                    tool_call_id_info = f"\n\nTool Call ID: {message.get('tool_call_id')}"
                
                # Display panel with message details and full content
                panel_title = f"Message #{original_index} (Role: {role})"
                panel_content = Text.from_markup(f"{content}{tool_calls_info}{tool_call_id_info}")
                
                console.print(
                    Panel(
                        panel_content,
                        title=panel_title,
                        border_style="cyan",
                        box=box.ROUNDED,
                        expand=True,
                        padding=(1, 2)
                    )
                )
            else:
                # For multiple rows, use a regular table with truncated content
                table = Table(title=f"Conversation History ({len(filtered_history)} messages)")
                table.add_column("#", style="dim")
                table.add_column("Role", style="cyan")
                table.add_column("Content", style="white")
                
                for message in filtered_history:
                    # Get original index
                    original_index = conversation_history.index(message) + 1
                    
                    # Format role
                    role = message.get("role", "unknown")
                    name = message.get("name", "")
                    if name:
                        role = f"{role} ({name})"
                    
                    # Format content with truncation for table view
                    content = message.get("content", "")
                    if content is None:
                        if "tool_calls" in message and message["tool_calls"]:
                            tool_calls = message["tool_calls"]
                            tool_names = []
                            for tc in tool_calls:
                                if "function" in tc:
                                    tool_names.append(tc["function"].get("name", "unknown"))
                            content = f"[Tool call: {', '.join(tool_names)}]"
                        else:
                            content = "[None]"
                    elif isinstance(content, str):
                        if len(content) > 100:
                            content = content[:97] + "..."
                    else:
                        try:
                            content = json.dumps(content)
                            if len(content) > 100:
                                content = content[:97] + "..."
                        except Exception:
                            content = str(content)
                            if len(content) > 100:
                                content = content[:97] + "..."
                    
                    # Add row to table
                    table.add_row(str(original_index), role, content)
                
                # Display table
                console.print(table)
        
    except Exception as e:
        # Print exception for debugging
        console.print(f"[bold red]ERROR: An exception occurred:[/bold red]")
        console.print(f"[red]{traceback.format_exc()}[/red]")
    
    return True

# Register commands
register_command("/conversation", conversation_history_command, ["-n", "--json"])
register_command("/ch", conversation_history_command, ["-n", "--json"])