# mcp_cli/chat/commands/model.py
"""
Command for changing LLM model settings in the chat interface.
"""

import os
from typing import List, Dict, Any
from rich import print

# Chat registration helper
from mcp_cli.chat.commands import register_command
from mcp_cli.provider_config import ProviderConfig
from mcp_cli.llm.llm_client import get_llm_client


async def cmd_model(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Display or change the current LLM model.

    Usage:
      /model            — Show the active model
      /model <name>     — Switch to a different model
    """
    # Get provider config
    provider_config = context.get("provider_config") or ProviderConfig()
    
    # Get current provider and model from provider_config, not context
    current_provider = provider_config.get_active_provider()
    current_model = provider_config.get_active_model()
    
    # Check for client
    client = context.get("client")
    if not client:
        print("[red]Error: LLM client not available[/red]")
        return True

    # No argument → just display
    if len(cmd_parts) < 2:
        print(f"[yellow]Current model:[/] {current_model}")
        print(f"[yellow]Current provider:[/] {current_provider}")
        print("[yellow]To change model:[/] /model <model_name>")
        return True

    # Change model
    new_model = cmd_parts[1]
    try:
        # Update the active model in provider_config
        provider_config.set_active_model(new_model)
        
        # Update context with new model for this session
        context["model"] = new_model
        
        # Update client
        try:
            context["client"] = get_llm_client(
                provider=current_provider,
                model=new_model,
                config=provider_config
            )
        except Exception as client_err:
            print(f"[red]Error creating new client:[/] {client_err}")
            # Fall back to just changing the model attribute
            if hasattr(client, "model"):
                client.model = new_model
        
        print(f"[green]Switched to model:[/] {new_model}")
    except Exception as e:
        print(f"[red]Failed to switch model:[/] {e}")
    
    return True


# Register commands
register_command("/model", cmd_model)
register_command("/m", cmd_model)