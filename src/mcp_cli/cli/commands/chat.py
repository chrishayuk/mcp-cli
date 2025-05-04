# mcp_cli/cli/commands/chat.py
"""Chat command implementation."""
from __future__ import annotations
import logging
import signal
from typing import Any, Callable
import typer

# mcp cli imports
from mcp_cli.cli.commands.base import BaseCommand
from mcp_cli.cli_options import process_options
from mcp_cli.tools.manager import ToolManager

# logger
logger = logging.getLogger(__name__)


class ChatCommand(BaseCommand):
    """Command implementation for chat mode."""
    
    def __init__(self):
        super().__init__("chat", "Start interactive chat mode.")
    
    async def execute(self, tool_manager: ToolManager, **params) -> Any:
        """Execute the chat mode."""
        from mcp_cli.chat.chat_handler import handle_chat_mode
        
        provider = params.get("provider", "openai")
        model = params.get("model", "gpt-4o-mini")
        
        logger.debug(f"Starting chat mode with provider={provider}, model={model}")
        
        return await handle_chat_mode(
            tool_manager,
            provider=provider,
            model=model
        )
        
    def register_as_default(self, app: typer.Typer, run_command_func: Callable) -> None:
        """Register this command as the default (no subcommand) action."""
        @app.callback(invoke_without_command=True)
        def common_options(
            ctx: typer.Context,
            config_file: str = "server_config.json",
            server: str = None,
            provider: str = "openai",
            model: str = None,
            disable_filesystem: bool = False,
            logging_level: str = typer.Option(
                "WARNING",
                help="Set the logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
            ),
        ):
            """Global CLI options. If no subcommand is provided, chat mode is launched."""
            # Set up logging
            numeric_level = getattr(logging, logging_level.upper(), None)
            if not isinstance(numeric_level, int):
                raise typer.BadParameter(f"Invalid logging level: {logging_level}")
            logging.getLogger().setLevel(numeric_level)
            
            # Process general options
            servers, _, server_names = process_options(
                server, disable_filesystem, provider, model, config_file
            )
            
            ctx.obj = {
                "config_file": config_file,
                "servers": servers,
                "server_names": server_names,
            }
            
            # If no subcommand, launch chat mode
            if ctx.invoked_subcommand is None:
                # Set up signal handlers
                def signal_handler(sig, _frame):
                    logging.debug(f"Received signal {sig}")
                    from mcp_cli.ui.ui_helpers import restore_terminal
                    restore_terminal()
                    raise typer.Exit()
                
                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)
                
                # Launch chat mode
                run_command_func(
                    self.wrapped_execute,
                    config_file,
                    servers,
                    extra_params={"provider": provider, "model": model, "server_names": server_names},
                )
                
                from mcp_cli.ui.ui_helpers import restore_terminal
                restore_terminal()
                raise typer.Exit()