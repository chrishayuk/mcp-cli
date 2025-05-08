# mcp_cli/cli/commands/provider.py
from __future__ import annotations
import typer
from typing import Any, Dict, Optional
import logging
import json

from rich.console import Console
from rich.table import Table

from mcp_cli.provider_config import ProviderConfig
from mcp_cli.cli.commands.base import BaseCommand

logger = logging.getLogger(__name__)

# ─── Typer sub‐app ───────────────────────────────────────────────────────────
app = typer.Typer(help="Manage LLM providers")

@app.command("list")
def provider_list() -> None:
    """
    List all configured providers.
    """
    console = Console()
    provider_config = ProviderConfig()
    
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


@app.command("config")
def provider_config() -> None:
    """
    Show detailed provider configurations.
    """
    console = Console()
    provider_config = ProviderConfig()
    
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


@app.command("set")
def provider_set(
    provider_name: str = typer.Argument(..., help="Provider name"),
    key: str = typer.Argument(..., help="Configuration key"),
    value: Optional[str] = typer.Argument(None, help="Configuration value (omit to clear)")
) -> None:
    """
    Set a provider configuration value.
    """
    console = Console()
    provider_config = ProviderConfig()
    
    # Handle special case for clearing value
    if value in ("none", "null"):
        value = None
        
    try:
        # Update single setting
        provider_config.set_provider_config(
            provider_name, 
            {key: value}
        )
        console.print(f"[green]Updated {provider_name}.{key} configuration[/green]")
    except Exception as e:
        console.print(f"[red]Error updating provider configuration: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("show")
def provider_show() -> None:
    """
    Show current provider and its configuration.
    """
    console = Console()
    provider_config = ProviderConfig()
    
    # Get current provider from environment or default
    import os
    current_provider = os.environ.get("LLM_PROVIDER", "openai")
    current_model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    
    console.print(f"[cyan]Current provider:[/cyan] {current_provider}")
    console.print(f"[cyan]Current model:[/cyan] {current_model}")
    
    # Show configuration for current provider
    if current_provider in provider_config.providers:
        config = provider_config.providers[current_provider]
        
        table = Table(title=f"{current_provider} Configuration")
        table.add_column("Setting", style="green")
        table.add_column("Value", style="yellow")
        
        for key, value in config.items():
            if key == "api_key" and value:
                value = "********"  # Mask API key
            table.add_row(key, str(value) if value is not None else "-")
            
        console.print(table)


# ─── In‐process command for CommandRegistry ─────────────────────────────────
class ProviderCommand(BaseCommand):
    """CLI 'provider' command."""

    def __init__(self):
        super().__init__(
            name="provider",
            help_text="Manage LLM provider configurations."
        )

    async def execute(self, tool_manager: Any, **params: Any) -> Any:
        """
        Execute provider configuration command based on subcommand.
        """
        logger.debug("Executing ProviderCommand")
        console = Console()
        provider_config = ProviderConfig()
        
        # Get subcommand
        subcommand = params.get("subcommand", "show")
        
        if subcommand == "list":
            # List providers
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
            
        elif subcommand == "config":
            # Show provider config
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
            
        elif subcommand == "set":
            # Set provider config
            provider_name = params.get("provider_name", "")
            key = params.get("key", "")
            value = params.get("value")
            
            if not (provider_name and key):
                console.print("[red]Error: Missing provider name or key[/red]")
                return False
                
            # Handle special case for clearing value
            if value in ("none", "null"):
                value = None
                
            try:
                # Update single setting
                provider_config.set_provider_config(
                    provider_name, 
                    {key: value}
                )
                console.print(f"[green]Updated {provider_name}.{key} configuration[/green]")
            except Exception as e:
                console.print(f"[red]Error updating provider configuration: {e}[/red]")
                return False
                
        else:  # show current provider
            # Get current provider from environment or params
            current_provider = params.get("provider", "openai")
            current_model = params.get("model", "gpt-4o-mini")
            
            console.print(f"[cyan]Current provider:[/cyan] {current_provider}")
            console.print(f"[cyan]Current model:[/cyan] {current_model}")
            
            # Show configuration for current provider
            if current_provider in provider_config.providers:
                config = provider_config.providers[current_provider]
                
                table = Table(title=f"{current_provider} Configuration")
                table.add_column("Setting", style="green")
                table.add_column("Value", style="yellow")
                
                for key, value in config.items():
                    if key == "api_key" and value:
                        value = "********"  # Mask API key
                    table.add_row(key, str(value) if value is not None else "-")
                    
                console.print(table)
                
        return True

    def register(self, app: typer.Typer, run_command_func: Callable) -> None:
        """Register provider command with Typer."""
        
        @app.command(self.name)
        def _provider(
            subcommand: str = typer.Argument("show", help="Action: show, list, config, set"),
            provider_name: Optional[str] = typer.Option(None, help="Provider name (for set)"),
            key: Optional[str] = typer.Option(None, help="Configuration key (for set)"),
            value: Optional[str] = typer.Option(None, help="Configuration value (for set)"),
            config_file: str = "server_config.json",
            server: Optional[str] = None,
            provider: str = "openai",
            model: Optional[str] = None,
            disable_filesystem: bool = False,
        ) -> None:
            from mcp_cli.cli_options import process_options
            
            # Process options
            servers, _, server_names = process_options(
                server, disable_filesystem, provider, model, config_file
            )
            
            # Prepare parameters
            extra_params = {
                "provider": provider,
                "model": model,
                "server_names": server_names,
                "subcommand": subcommand,
                "provider_name": provider_name,
                "key": key,
                "value": value,
            }
            
            # Execute the command
            run_command_func(
                self.wrapped_execute,
                config_file,
                servers,
                extra_params=extra_params,
            )