# src/cli/chat/commands/conversation_history.py
import json
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax

# Import the registration function from your commands package
from cli.chat.commands import register_command

async def conversation_history_command(args, context):
    """
    Display the conversation history of the current chat session.

    Usage:
      /conversation         - Show the conversation history in a tabular view.
      /conversation --json  - Show the conversation history in JSON format.

    The conversation history should be stored in the context under the "conversation_history" key.
    Each message is expected to be a dictionary with at least "role" and "content" keys.
    """
    console = Console()
    conversation_history = context.get("conversation_history", [])
    
    if not conversation_history:
        console.print("[italic yellow]No conversation history available.[/italic yellow]")
        return True

    # Check if the user requested JSON output.
    show_json = "--json" in args

    if show_json:
        raw_json = json.dumps(conversation_history, indent=2)
        console.print(Syntax(raw_json, "json", theme="monokai", line_numbers=True))
        return True

    # Otherwise, show a table of conversation messages.
    table = Table(title=f"Conversation History ({len(conversation_history)} messages)")
    table.add_column("#", style="dim")
    table.add_column("Role", style="cyan")
    table.add_column("Content", style="white")

    for i, message in enumerate(conversation_history):
        role = message.get("role", "unknown")
        content = message.get("content", "")
        if isinstance(content, str):
            # Truncate long messages for display purposes.
            if len(content) > 100:
                content = content[:97] + "..."
        else:
            try:
                content = json.dumps(content, indent=2)
            except Exception:
                content = str(content)
        table.add_row(str(i + 1), role, content)

    console.print(table)
    return True

# Register the command with aliases.
register_command("/conversation", conversation_history_command, ["--json"])
register_command("/ch", conversation_history_command, ["--json"])
