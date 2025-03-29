# mcp_cli/chat/commands/models.py
"""
System-related commands for changing settings and configurations.
"""

from typing import List, Dict, Any
from rich import print

# imports
from mcp_cli.chat.commands import register_command


async def cmd_model(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Display or change the current LLM model.
    
    Usage: /model [model_name]
    """
    client = context['client']
    current_model = context['model']
    
    if len(cmd_parts) < 2:
        print(f"[yellow]Current model: {current_model}[/yellow]")
        print("[yellow]To change model: /model <model_name>[/yellow]")
        return True
        
    new_model = cmd_parts[1]
    try:
        # Update the model in the client
        client.model = new_model
        # Also update in the context
        context['model'] = new_model
        print(f"[green]Switched to model: {new_model}[/green]")
    except Exception as e:
        print(f"[red]Failed to switch model: {e}[/red]")
    return True


async def cmd_provider(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Display or change the current LLM provider.
    
    Usage: /provider [provider_name]
    """
    client = context['client']
    current_provider = context['provider']
    
    if len(cmd_parts) < 2:
        print(f"[yellow]Current provider: {current_provider}[/yellow]")
        print("[yellow]To change provider: /provider <provider_name>[/yellow]")
        return True
        
    new_provider = cmd_parts[1].lower()
    try:
        # Update the provider in the client (if supported)
        if hasattr(client, 'provider'):
            client.provider = new_provider
            # Also update in the context
            context['provider'] = new_provider
            print(f"[green]Switched to provider: {new_provider}[/green]")
        else:
            print("[yellow]Changing provider not supported with current client.[/yellow]")
    except Exception as e:
        print(f"[red]Failed to switch provider: {e}[/red]")
    return True


# Register all commands in this module
register_command("/model", cmd_model)
register_command("/provider", cmd_provider)