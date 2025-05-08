# mcp_cli/chat/commands/provider.py
"""
Command for managing LLM providers in the chat interface.
"""

from typing import List, Dict, Any
from rich import print
from rich.table import Table
from rich.console import Console

# Chat registration helper
from mcp_cli.chat.commands import register_command
from mcp_cli.provider_config import ProviderConfig
from mcp_cli.llm.llm_client import get_llm_client


async def cmd_provider(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Display or change the current LLM provider.

    Usage:
      /provider                   — Show the active provider
      /provider <name>            — Switch to a different provider
      /provider list              — List all configured providers
      /provider set <name> <key> <value>  — Set provider configuration
      /provider config            — Show provider configurations
    """
    console = Console()
    
    # Get or create provider config
    provider_config = context.get("provider_config") or ProviderConfig()
    
    # No arguments - show current provider
    if len(cmd_parts) == 1:
        # Get current provider and model from provider_config
        current_provider = provider_config.get_active_provider()
        current_model = provider_config.get_active_model()
        
        print(f"[cyan]Current provider:[/cyan] {current_provider}")
        print(f"[cyan]Current model:[/cyan] {current_model}")
        print("[dim]To change provider: /provider <provider_name>[/dim]")
        return True
        
    # Get subcommand
    subcommand = cmd_parts[1].lower() if len(cmd_parts) > 1 else ""
    
    # List providers
    if subcommand == "list":
        table = Table(title="Available Providers")
        table.add_column("Provider", style="green")
        table.add_column("Default Model", style="cyan")
        table.add_column("API Base", style="yellow")
        
        for name, config in provider_config.providers.items():
            if name == "__global__":
                continue  # Skip the global settings entry
                
            table.add_row(
                name,
                config.get("default_model", "-"),
                config.get("api_base", "-")
            )
            
        console.print(table)
        return True
        
    # Show provider config
    elif subcommand == "config":
        table = Table(title="Provider Configurations")
        table.add_column("Provider", style="green")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="yellow")
        
        for provider_name, config in provider_config.providers.items():
            if provider_name == "__global__":
                # Add a special section for the global settings
                for key, value in config.items():
                    table.add_row(
                        "__global__" if key == list(config.keys())[0] else "",
                        key,
                        str(value) if value is not None else "-"
                    )
                continue
                
            for key, value in config.items():
                if key == "api_key" and value:
                    value = "********"  # Mask API key
                table.add_row(
                    provider_name if key == list(config.keys())[0] else "",
                    key,
                    str(value) if value is not None else "-"
                )
                
        console.print(table)
        return True
        
    # Set provider config
    elif subcommand == "set" and len(cmd_parts) >= 4:
        provider_name = cmd_parts[2]
        setting_key = cmd_parts[3]
        setting_value = cmd_parts[4] if len(cmd_parts) > 4 else None
        
        if provider_name == "__global__":
            print("[red]Cannot directly modify global settings[/red]")
            return True
            
        # Handle special case for clearing value
        if setting_value == "none" or setting_value == "null":
            setting_value = None
            
        try:
            # Update single setting
            provider_config.set_provider_config(
                provider_name, 
                {setting_key: setting_value}
            )
            print(f"[green]Updated {provider_name}.{setting_key} configuration[/green]")
            
            # Update provider_config in context
            context["provider_config"] = provider_config
        except Exception as e:
            print(f"[red]Error updating provider configuration: {e}[/red]")
            
        return True
        
    # Switch provider
    else:
        new_provider = cmd_parts[1]
        # Check if this provider exists in config
        if new_provider not in provider_config.providers or new_provider == "__global__":
            print(f"[red]Unknown provider: {new_provider}[/red]")
            print(f"[yellow]Available providers: {', '.join([p for p in provider_config.providers.keys() if p != '__global__'])}[/yellow]")
            return True
            
        # Update active provider in provider_config
        provider_config.set_active_provider(new_provider)
        
        # Update context for the current session
        context["provider"] = new_provider
        
        # Also update model to the default for this provider if available
        default_model = provider_config.get_default_model(new_provider)
        if default_model:
            # Update active model in provider_config
            provider_config.set_active_model(default_model)
            
            # Update context for the current session
            context["model"] = default_model
            
            print(f"[green]Switched to provider: {new_provider} with model: {default_model}[/green]")
        else:
            print(f"[green]Switched to provider: {new_provider}[/green]")
            print("[yellow]No default model found for this provider. Use /model to set a model.[/yellow]")
            
        # Update client
        try:
            context["client"] = get_llm_client(
                provider=new_provider,
                model=context.get("model", default_model),
                config=provider_config
            )
            print("[green]LLM client updated successfully[/green]")
        except Exception as e:
            print(f"[red]Error updating LLM client: {e}[/red]")
            
        return True


# Register commands
register_command("/provider", cmd_provider)
register_command("/p", cmd_provider)