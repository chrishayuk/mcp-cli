# src/cli/chat/commands/conversation.py
"""
Commands for managing conversation history and context.
"""
import json
from typing import List, Dict, Any
from rich import print
from rich.panel import Panel
from rich.markdown import Markdown
from rich.console import Console

# imports
from mcp_cli.chat.commands import register_command
from mcp_cli.ui.ui_helpers import display_welcome_banner, clear_screen


async def cmd_cls(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Clear the terminal screen only, preserving conversation history.
    
    Usage: /cls
    
    This command clears the screen but keeps your conversation history intact.
    It's useful for decluttering your terminal without losing context.
    """
    # Clear the screen
    clear_screen()
    
    # Redisplay welcome banner
    display_welcome_banner(context)
    
    print("[green]Screen cleared. Conversation history preserved.[/green]")
    return True


async def cmd_clear(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Clear screen and reset conversation history.
    
    Usage: /clear
    
    This command clears both the screen and resets conversation history to start fresh.
    Only the system prompt is preserved.
    """
    # Clear the screen
    clear_screen()
    
    # Reset conversation history
    history = context['conversation_history']
    if history and history[0]["role"] == "system":
        system_prompt = history[0]["content"]
        history.clear()
        history.append({"role": "system", "content": system_prompt})
    
    # Redisplay welcome banner
    display_welcome_banner(context)
    
    print("[green]Screen cleared and conversation history reset.[/green]")
    return True


async def cmd_compact(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Clear history but keep a summary in context.
    This preserves the conversation's essence while freeing up token space.
    
    Usage: /compact
    """
    history = context['conversation_history']
    client = context['client']
    
    # Check if there's any history to compact
    if len(history) <= 1:
        print("[yellow]No conversation history to compact.[/yellow]")
        return True
    
    system_prompt = history[0]["content"]
    
    # Add a summary request to the conversation
    summary_request = {
        "role": "user", 
        "content": "Please provide a brief summary of our conversation so far. Keep it concise."
    }
    summary_history = history.copy()
    summary_history.append(summary_request)
    
    # Use the client to generate a summary
    console = Console()
    with console.status("[cyan]Generating conversation summary...[/cyan]", spinner="dots"):
        try:
            completion = client.create_completion(messages=summary_history)
            summary = completion.get("response", "No summary available")
        except Exception as e:
            print(f"[red]Error generating summary: {e}[/red]")
            summary = "Failed to generate summary."
    
    # Reset history with system prompt and summary
    history.clear()
    history.append({"role": "system", "content": system_prompt})
    history.append({
        "role": "assistant", 
        "content": f"**Conversation Summary**\n\n{summary}\n\n*The conversation history has been compacted.*"
    })
    
    print("[green]Conversation history compacted with summary.[/green]")
    print(Panel(Markdown(f"**Summary:**\n\n{summary}"), style="cyan", title="Conversation Summary"))
    return True


async def cmd_save(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Save conversation history to a file.
    
    Usage: /save <filename>
    """
    if len(cmd_parts) < 2:
        print("[yellow]Please provide a filename: /save <filename>[/yellow]")
        return True
        
    filename = cmd_parts[1]
    if not filename.endswith('.json'):
        filename += '.json'
    
    history = context['conversation_history']    
    try:
        # Save conversation excluding system prompt
        with open(filename, 'w') as f:
            json.dump(history[1:], f, indent=2)
        print(f"[green]Conversation saved to {filename}[/green]")
    except Exception as e:
        print(f"[red]Failed to save conversation: {e}[/red]")
    return True


# Register all commands in this module
register_command("/cls", cmd_cls)
register_command("/clear", cmd_clear)
register_command("/compact", cmd_compact)
register_command("/save", cmd_save, ["<filename>"])