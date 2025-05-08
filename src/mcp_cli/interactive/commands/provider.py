# mcp_cli/interactive/commands/provider.py
from typing import Any, List
from rich.console import Console
from rich.table import Table

from .base import InteractiveCommand
from mcp_cli.provider_config import ProviderConfig
from mcp_cli.llm.llm_client import get_llm_client

class ProviderCommand(InteractiveCommand):
    """Command to manage LLM providers in interactive mode."""
    
    def __init__(self):
        super().__init__(
            name="provider",
            help_text="""Manage LLM providers.
            
Usage:
  provider                   Show current provider and model
  provider <name>            Switch to provider
  provider list              List all configured providers
  provider set <name> <key> <value>  Set provider configuration
  provider config            Show provider configurations
            """,
            aliases=["p"],
        )
    
    async def execute(
        self,
        args: List[str],
        tool_manager: Any = None,
        **kwargs: Any,
    ) -> None:
        """Execute the provider command."""
        console = Console()
        
        # Get or create provider config
        provider_config = kwargs.get("provider_config") or ProviderConfig()
        
        # No arguments - show current provider
        if not args:
            # Get active provider from provider_config, not kwargs
            current_provider = provider_config.get_active_provider()
            current_model = provider_config.get_active_model()
            
            console.print(f"[cyan]Current provider:[/cyan] {current_provider}")
            console.print(f"[cyan]Current model:[/cyan] {current_model}")
            console.print("[dim]To change provider: provider <provider_name>[/dim]")
            return
            
        # Get subcommand
        subcommand = args[0].lower() if args else ""
        
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
            return
            
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
            return
            
        # Set provider config
        elif subcommand == "set" and len(args) >= 3:
            provider_name = args[1]
            setting_key = args[2]
            setting_value = args[3] if len(args) > 3 else None
            
            # Handle special case for clearing value
            if setting_value == "none" or setting_value == "null":
                setting_value = None
                
            try:
                # Update single setting
                provider_config.set_provider_config(
                    provider_name, 
                    {setting_key: setting_value}
                )
                console.print(f"[green]Updated {provider_name}.{setting_key} configuration[/green]")
                
                # Update the context provider_config
                kwargs["provider_config"] = provider_config
            except Exception as e:
                console.print(f"[red]Error updating provider configuration: {e}[/red]")
                
            return
            
        # Switch provider
        else:
            new_provider = args[0]
            # Check if this provider exists in config
            if new_provider not in provider_config.providers or new_provider == "__global__":
                console.print(f"[red]Unknown provider: {new_provider}[/red]")
                console.print(f"[yellow]Available providers: {', '.join([p for p in provider_config.providers.keys() if p != '__global__'])}[/yellow]")
                return
                
            # Update active provider in provider_config
            provider_config.set_active_provider(new_provider)
            
            # Also update kwargs for the current session
            kwargs["provider"] = new_provider
            
            # Also update model to the default for this provider if available
            default_model = provider_config.get_default_model(new_provider)
            if default_model:
                # Update the active model in provider_config
                provider_config.set_active_model(default_model)
                
                # Update kwargs for the current session
                kwargs["model"] = default_model
                
                console.print(f"[green]Switched to provider: {new_provider} with model: {default_model}[/green]")
            else:
                console.print(f"[green]Switched to provider: {new_provider}[/green]")
                console.print("[yellow]No default model found for this provider. Use 'model' to set a model.[/yellow]")
                
            # Update client
            try:
                client = get_llm_client(
                    provider=new_provider,
                    model=kwargs.get("model", default_model),
                    config=provider_config
                )
                kwargs["client"] = client
                console.print("[green]LLM client updated successfully[/green]")
            except Exception as e:
                console.print(f"[red]Error updating LLM client: {e}[/red]")