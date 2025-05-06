# mcp_cli/chat/commands/provider.py
"""
Chat-mode commands for managing LLM providers.
"""
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich import print

from mcp_cli.provider_config import ProviderConfig
from mcp_cli.chat.commands import register_command

async def cmd_provider(cmd_parts: List[str], context: Dict[str, Any]) -> bool:
    """
    Display or configure LLM providers.
    
    Usage:
      /provider                   Show current provider and model
      /provider <name>            Switch to provider
      /provider list              List all configured providers
      /provider set <name> <key> <value>  Set provider configuration
      /provider config            Show provider configurations
    """
    console = Console()
    
    # Get or create provider config
    provider_config = context.get("provider_config") or ProviderConfig()
    
    # No arguments - show current provider
    if len(cmd_parts) == 1:
        current_provider = context.get("provider", "openai")
        current_model = context.get("model", "unknown")
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
        except Exception as e:
            print(f"[red]Error updating provider configuration: {e}[/red]")
            
        return True
        
    # Switch provider
    else:
        new_provider = cmd_parts[1]
        # Check if this provider exists in config
        if new_provider not in provider_config.providers:
            print(f"[red]Unknown provider: {new_provider}[/red]")
            print(f"[yellow]Available providers: {', '.join(provider_config.providers.keys())}[/yellow]")
            return True
            
        # Update context with new provider
        context["provider"] = new_provider
        
        # Also update model to the default for this provider if available
        default_model = provider_config.get_default_model(new_provider)
        if default_model:
            context["model"] = default_model
            print(f"[green]Switched to provider: {new_provider} with model: {default_model}[/green]")
        else:
            print(f"[green]Switched to provider: {new_provider}[/green]")
            print("[yellow]No default model found for this provider. Use /model to set a model.[/yellow]")
            
        # Update client
        try:
            from mcp_cli.llm.llm_client import get_llm_client
            
            context["client"] = get_llm_client(
                provider=new_provider,
                model=context.get("model"),
                config=provider_config
            )
            print("[green]LLM client updated successfully[/green]")
        except Exception as e:
            print(f"[red]Error updating LLM client: {e}[/red]")
            
        return True

# Register commands
register_command("/provider", cmd_provider)
register_command("/p", cmd_provider)