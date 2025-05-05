# mcp_cli/chat/commands/models.py
"""
System-related commands for changing LLM settings in the chat interface.
"""

import os
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
    # Check for client
    client = context.get("client")
    if not client:
        print("[red]Error: LLM client not available[/red]")
        return True
    
    # Get current model - first try the client's model attribute
    current_model = None
    try:
        if hasattr(client, "model"):
            current_model = client.model
    except Exception:
        pass
    
    # If that failed, try the context
    if current_model is None:
        current_model = context.get("model")
    
    # If still failed, try the environment variable as last resort
    if current_model is None:
        current_model = os.environ.get("LLM_MODEL")
    
    # If we still don't have a model, report error
    if current_model is None:
        print("[red]Error: Could not determine current model[/red]")
        return True

    # No argument → just display
    if len(cmd_parts) < 2:
        print(f"[yellow]Current model:[/] {current_model}")
        print("[yellow]To change model:[/] /model <model_name>")
        return True

    # Change model
    new_model = cmd_parts[1]
    try:
        # Update client
        if hasattr(client, "model"):
            client.model = new_model
        
        # Update context
        context["model"] = new_model
        
        # Update environment variable for good measure
        os.environ["LLM_MODEL"] = new_model
        
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
