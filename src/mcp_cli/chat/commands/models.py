# mcp_cli/chat/commands/models.py
"""
System-related commands for changing LLM settings in the chat interface.
"""

from typing import List, Dict, Any
from rich import print

# Chat registration helper
from mcp_cli.chat.commands import register_command


async def cmd_model(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Display or change the current LLM model.

    Usage:
      /model            — Show the active model
      /model <name>     — Switch to a different model
    """
    client = context.get("client")
    current_model = context.get("model")

    if not client or current_model is None:
        print("[red]Error: chat client or model context not available[/red]")
        return True

    # No argument → just display
    if len(cmd_parts) < 2:
        print(f"[yellow]Current model:[/] {current_model}")
        print("[yellow]To change model:[/] /model <model_name>")
        return True

    # Change model
    new_model = cmd_parts[1]
    try:
        client.model = new_model
        context["model"] = new_model
        print(f"[green]Switched to model:[/] {new_model}")
    except Exception as e:
        print(f"[red]Failed to switch model:[/] {e}")
    return True


async def cmd_provider(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Display or change the current LLM provider.

    Usage:
      /provider         — Show the active provider
      /provider <name>  — Switch to a different provider
    """
    client = context.get("client")
    current_provider = context.get("provider")

    if not client or current_provider is None:
        print("[red]Error: chat client or provider context not available[/red]")
        return True

    if len(cmd_parts) < 2:
        print(f"[yellow]Current provider:[/] {current_provider}")
        print("[yellow]To change provider:[/] /provider <provider_name>")
        return True

    new_provider = cmd_parts[1].lower()
    try:
        if hasattr(client, "provider"):
            client.provider = new_provider
            context["provider"] = new_provider
            print(f"[green]Switched to provider:[/] {new_provider}")
        else:
            print("[yellow]This client does not support changing provider at runtime.[/yellow]")
    except Exception as e:
        print(f"[red]Failed to switch provider:[/] {e}")
    return True


# Register commands
register_command("/model",    cmd_model)
register_command("/provider", cmd_provider)
