# mcp_cli/interactive/commands/model.py
from typing import Any, List
from rich.console import Console

from .base import InteractiveCommand
from mcp_cli.provider_config import ProviderConfig
from mcp_cli.llm.llm_client import get_llm_client

class ModelCommand(InteractiveCommand):
    """Command to change the LLM model in interactive mode."""
    
    def __init__(self):
        super().__init__(
            name="model",
            help_text="View or change the current LLM model.",
            aliases=["m"],
        )
    
    async def execute(
        self,
        args: List[str],
        tool_manager: Any = None,
        **kwargs: Any,
    ) -> None:
        """Execute the model command."""
        console = Console()
        
        # Get provider config and current state
        provider_config = kwargs.get("provider_config") or ProviderConfig()
        
        # Get current provider and model from provider_config, not kwargs
        current_provider = provider_config.get_active_provider()
        current_model = provider_config.get_active_model()
        
        # No arguments - show current model
        if not args:
            console.print(f"[cyan]Current model:[/cyan] {current_model}")
            console.print(f"[cyan]Provider:[/cyan] {current_provider}")
            console.print("[dim]To change model: model <model_name>[/dim]")
            return
            
        # Change model
        new_model = args[0]
        
        # Update active model in provider_config
        provider_config.set_active_model(new_model)
        
        # Update kwargs for the current session
        kwargs["model"] = new_model
        
        # Update client with new model
        try:
            client = get_llm_client(
                provider=current_provider,
                model=new_model,
                config=provider_config
            )
            kwargs["client"] = client
            console.print(f"[green]Switched to model: {new_model}[/green]")
        except Exception as e:
            console.print(f"[red]Error updating LLM client: {e}[/red]")