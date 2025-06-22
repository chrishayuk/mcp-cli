# mcp_cli/cli/commands/chat.py
"""Clean chat command implementation using ModelManager."""

from __future__ import annotations

import logging
import signal
from typing import Any, Callable, Optional

import typer

from mcp_cli.cli.commands.base import BaseCommand
from mcp_cli.cli_options import process_options
from mcp_cli.model_manager import ModelManager
from mcp_cli.tools.manager import ToolManager

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def _set_logging(level: str) -> None:
    """Set logging level from string."""
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        raise typer.BadParameter(f"Invalid logging level: {level}")
    logging.getLogger().setLevel(numeric)


# ──────────────────────────────────────────────────────────────────────────────
# command
# ──────────────────────────────────────────────────────────────────────────────
class ChatCommand(BaseCommand):
    """Chat command that starts interactive chat mode."""

    def __init__(self) -> None:
        super().__init__("chat", "Start interactive chat mode.")

    async def execute(self, tool_manager: ToolManager, **params) -> Any:
        """Start the chat UI."""
        from mcp_cli.chat.chat_handler import handle_chat_mode

        # Get parameters - ModelManager will handle defaults
        provider = params.get("provider")
        model = params.get("model")
        api_base = params.get("api_base")
        api_key = params.get("api_key")

        log.debug("Starting chat (provider=%s model=%s)", provider, model)
        
        return await handle_chat_mode(
            tool_manager=tool_manager,
            provider=provider,
            model=model,
            api_base=api_base,
            api_key=api_key
        )

    def register(self, app: typer.Typer, run_command_func: Callable) -> None:
        """Register chat as an explicit sub-command."""

        @app.command(self.name, help=self.help)
        def _chat(
            config_file: str = typer.Option("server_config.json", help="Configuration file path"),
            server: Optional[str] = typer.Option(None, help="Server to connect to"),
            provider: Optional[str] = typer.Option(None, help="LLM provider name"),
            model: Optional[str] = typer.Option(None, help="Model name"),
            api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL"),
            api_key: Optional[str] = typer.Option(None, "--api-key", help="API key"),
            disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
            logging_level: str = typer.Option("WARNING", help="Set logging level"),
        ) -> None:
            """Start interactive chat mode."""
            _set_logging(logging_level)

            # Use ModelManager to determine provider/model if not specified
            model_manager = ModelManager()
            effective_provider = provider or model_manager.get_active_provider()
            effective_model = model or model_manager.get_active_model()

            servers, _, server_names = process_options(
                server, disable_filesystem, effective_provider, effective_model, config_file
            )

            extra = {
                "provider": provider,  # Pass None if not specified - let ModelManager decide
                "model": model,        # Pass None if not specified - let ModelManager decide
                "api_base": api_base,
                "api_key": api_key,
                "server_names": server_names,
            }
            
            run_command_func(self.wrapped_execute, config_file, servers, extra_params=extra)

    def register_as_default(self, app: typer.Typer, run_command_func: Callable) -> None:
        """
        Make chat the default action when no sub-command is given.
        """

        @app.callback(invoke_without_command=True)
        def _default(
            ctx: typer.Context,
            config_file: str = typer.Option("server_config.json", help="Configuration file path"),
            server: Optional[str] = typer.Option(None, help="Server to connect to"),
            provider: Optional[str] = typer.Option(None, help="LLM provider name"),
            model: Optional[str] = typer.Option(None, help="Model name"),
            api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL"),
            api_key: Optional[str] = typer.Option(None, "--api-key", help="API key"),
            disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
            logging_level: str = typer.Option("WARNING", help="Set logging level"),
        ) -> None:
            """Default command handler."""
            _set_logging(logging_level)

            # Use ModelManager to determine provider/model if not specified
            model_manager = ModelManager()
            effective_provider = provider or model_manager.get_active_provider()
            effective_model = model or model_manager.get_active_model()

            servers, _, server_names = process_options(
                server, disable_filesystem, effective_provider, effective_model, config_file
            )
            
            ctx.obj = {
                "config_file": config_file,
                "servers": servers,
                "server_names": server_names,
            }

            # If no sub-command was invoked, start chat mode
            if ctx.invoked_subcommand is None:
                # Set up signal handlers for graceful shutdown
                def _sig_handler(sig, _frame):
                    log.debug("Received signal %s - exiting chat", sig)
                    from mcp_cli.ui.ui_helpers import restore_terminal
                    restore_terminal()
                    raise typer.Exit()

                signal.signal(signal.SIGINT, _sig_handler)
                signal.signal(signal.SIGTERM, _sig_handler)

                extra = {
                    "provider": provider,  # Pass None if not specified
                    "model": model,        # Pass None if not specified  
                    "api_base": api_base,
                    "api_key": api_key,
                    "server_names": server_names,
                }
                
                try:
                    run_command_func(
                        self.wrapped_execute,
                        config_file,
                        servers,
                        extra_params=extra,
                    )
                finally:
                    from mcp_cli.ui.ui_helpers import restore_terminal
                    restore_terminal()
                    raise typer.Exit()


# ──────────────────────────────────────────────────────────────────────────────
# Alternative: Simplified version without default registration
# ──────────────────────────────────────────────────────────────────────────────

class SimpleChatCommand(BaseCommand):
    """Simplified chat command that only handles explicit chat invocation."""

    def __init__(self) -> None:
        super().__init__("chat", "Start interactive chat mode.")

    async def execute(self, tool_manager: ToolManager, **params) -> Any:
        """Start the chat UI."""
        from mcp_cli.chat.chat_handler import handle_chat_mode

        return await handle_chat_mode(
            tool_manager=tool_manager,
            provider=params.get("provider"),
            model=params.get("model"),
            api_base=params.get("api_base"),
            api_key=params.get("api_key")
        )

    def register(self, app: typer.Typer, run_command_func: Callable) -> None:
        """Register chat command."""

        @app.command(self.name, help=self.help)
        def _chat(
            config_file: str = typer.Option("server_config.json", help="Configuration file path"),
            server: Optional[str] = typer.Option(None, help="Server to connect to"),
            provider: Optional[str] = typer.Option(None, help="LLM provider name"),
            model: Optional[str] = typer.Option(None, help="Model name"),
            api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL"),
            api_key: Optional[str] = typer.Option(None, "--api-key", help="API key"),
            disable_filesystem: bool = typer.Option(False, help="Disable filesystem access"),
            logging_level: str = typer.Option("WARNING", help="Set logging level"),
        ) -> None:
            """Start interactive chat mode."""
            _set_logging(logging_level)

            # Let ModelManager handle provider/model defaults
            servers, _, server_names = process_options(
                server, disable_filesystem, provider or "openai", model or "gpt-4o-mini", config_file
            )

            extra = {
                "provider": provider,
                "model": model,
                "api_base": api_base,
                "api_key": api_key,
                "server_names": server_names,
            }
            
            run_command_func(self.wrapped_execute, config_file, servers, extra_params=extra)
            