# src/mcp_cli/core/app.py
"""Core application setup and configuration."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

import typer

from mcp_cli.core.commands import CommandRegistry
from mcp_cli.core.resolver import ModelResolver
from mcp_cli.core.config import AppConfig
from mcp_cli.logging_config import setup_logging, get_logger
from mcp_cli.cli_options import process_options
from mcp_cli.ui.ui_helpers import restore_terminal

logger = get_logger("app")


class MCPApp:
    """Main MCP CLI application."""
    
    def __init__(self):
        self.typer_app = typer.Typer(add_completion=False)
        self.command_registry = CommandRegistry()
        self.model_resolver = ModelResolver()
        self.config = AppConfig()
        
        # Setup main callback
        self._setup_main_callback()
        
        # Register all commands
        self.command_registry.register_all(self.typer_app)
    
    def _setup_main_callback(self):
        """Setup the main callback that handles no-subcommand case."""
        
        @self.typer_app.callback(invoke_without_command=True)
        def main_callback(
            ctx: typer.Context,
            config_file: str = typer.Option("server_config.json", help="Configuration file path"),
            server: Optional[str] = typer.Option(None, help="Server to connect to"),
            provider: Optional[str] = typer.Option(None, help="LLM provider name"),
            model: Optional[str] = typer.Option(None, help="Model name"),
            api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL"),
            api_key: Optional[str] = typer.Option(None, "--api-key", help="API key"),
            disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
            quiet: bool = typer.Option(False, "-q", "--quiet", help="Suppress most log output"),
            verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging"),
            log_level: str = typer.Option("WARNING", "--log-level", help="Set log level"),
        ) -> None:
            """MCP CLI - If no subcommand is given, start chat mode."""
            
            # Configure logging
            setup_logging(level=log_level, quiet=quiet, verbose=verbose)
            
            # Store options in context for subcommands
            ctx.obj = self.config.from_options(
                config_file=config_file,
                server=server,
                provider=provider,
                model=model,
                api_base=api_base,
                api_key=api_key,
                disable_filesystem=disable_filesystem,
                quiet=quiet,
                verbose=verbose,
                log_level=log_level
            )
            
            # If a subcommand was invoked, let it handle things
            if ctx.invoked_subcommand is not None:
                return
            
            # No subcommand - start default chat mode
            self._start_default_chat(ctx.obj)
    
    def _start_default_chat(self, config: AppConfig):
        """Start the default chat mode."""
        from mcp_cli.core.handlers import ChatHandler
        
        try:
            handler = ChatHandler(config, self.model_resolver)
            asyncio.run(handler.run())
        except KeyboardInterrupt:
            print("\n[yellow]Interrupted[/yellow]")
            logger.debug("Chat mode interrupted by user")
        except Exception as e:
            print(f"[red]Error:[/red] {e}")
            logger.error(f"Chat mode failed: {e}", exc_info=True)
        finally:
            restore_terminal()
            raise typer.Exit()
    
    def __call__(self):
        """Make the app callable."""
        return self.typer_app()


def create_app() -> MCPApp:
    """
    Factory function to create the application.
    
    Returns:
        Configured MCPApp instance
    """
    app = MCPApp()
    
    # Show startup message
    print("✓ MCP CLI ready")
    print("  Use --help to see all options")
    
    return app