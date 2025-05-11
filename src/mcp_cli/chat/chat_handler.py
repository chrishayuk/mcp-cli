# mcp_cli/chat/chat_handler.py
"""
mcp_cli.chat.chat_handler
=========================

High-level entry point for the MCP-CLI "chat" mode.
Initialises a ChatContext, sets up the TUI, then enters the
main read–eval–print loop.

The module contains *no* model-specific code; all LLM access is delegated
to the llm/ sub-package through ChatContext.
"""

from __future__ import annotations

import asyncio
import gc
import logging
from typing import Any, Optional

from rich import print
from rich.panel import Panel
from rich.console import Console

# ── local imports ──────────────────────────────────────────────────────
from mcp_cli.chat.chat_context import ChatContext
from mcp_cli.chat.ui_manager import ChatUIManager
from mcp_cli.chat.conversation import ConversationProcessor
from mcp_cli.ui.ui_helpers import clear_screen, display_welcome_banner
from mcp_cli.provider_config import ProviderConfig
from mcp_cli.tools.manager import ToolManager

# Set up logger
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------- #
# public helper                                                         #
# --------------------------------------------------------------------- #
async def handle_chat_mode(
    manager: Any,                    # ToolManager *or* stream-manager stub
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    provider_config: Optional[ProviderConfig] = None,
) -> bool:
    """
    Launch the interactive chat loop.

    Parameters
    ----------
    manager
        Either a fully-initialised ``ToolManager`` **or** a lightweight
        stream-manager-like object (used by the test-suite).
    provider / model
        Passed straight through to the ChatContext / LLM client.
    api_base / api_key
        Optional API settings that override provider_config.
    provider_config
        Optional ProviderConfig instance for LLM configurations.

    Returns
    -------
    bool
        *True* if the session ended normally, *False* on hard failure.
    """
    ui: Optional[ChatUIManager] = None
    exit_ok = True
    console = Console()

    try:
        # Initialize provider configuration if not provided
        if provider_config is None:
            provider_config = ProviderConfig()
            
        # Update provider config if API settings were provided
        if api_base or api_key:
            config_updates = {}
            if api_base:
                config_updates["api_base"] = api_base
            if api_key:
                config_updates["api_key"] = api_key
                
            provider_config.set_provider_config(provider, config_updates)

        # ── build chat context ─────────────────────────────────────────
        with console.status("[cyan]Initializing chat context...[/cyan]", spinner="dots"):
            if isinstance(manager, ToolManager):
                logger.debug("Creating ChatContext with ToolManager")
                ctx = ChatContext(
                    tool_manager=manager, 
                    provider=provider, 
                    model=model,
                    provider_config=provider_config,
                    api_base=api_base,
                    api_key=api_key
                )
            else:
                # assume test stub
                logger.debug("Creating ChatContext with stream manager (test mode)")
                ctx = ChatContext(
                    stream_manager=manager, 
                    provider=provider, 
                    model=model,
                    provider_config=provider_config,
                    api_base=api_base,
                    api_key=api_key
                )
    
            if not await ctx.initialize():
                print("[red]Failed to initialise chat context.[/red]")
                return False

        # ── welcome banner (only once) ────────────────────────────────
        clear_screen()
        display_welcome_banner(ctx.to_dict())

        # ── UI + conversation helpers ────────────────────────────────
        ui = ChatUIManager(ctx)
        convo = ConversationProcessor(ctx, ui)

        # ── main REPL loop ────────────────────────────────────────────
        while True:
            try:
                user_msg = await ui.get_user_input()

                # blank line → ignore
                if not user_msg:
                    continue

                # plain "exit/quit" terminates chat
                if user_msg.lower() in ("exit", "quit"):
                    print(Panel("Exiting chat mode.", style="bold red"))
                    break

                # slash-commands
                if user_msg.startswith("/"):
                    handled = await ui.handle_command(user_msg)
                    if ctx.exit_requested or handled:
                        if ctx.exit_requested:
                            break
                        continue

                # normal chat turn
                ui.print_user_message(user_msg)
                ctx.conversation_history.append({"role": "user", "content": user_msg})
                await convo.process_conversation()

            except KeyboardInterrupt:
                print("\n[yellow]Interrupted – type 'exit' to quit.[/yellow]")
            except EOFError:
                print(Panel("EOF detected – exiting chat.", style="bold red"))
                break
            except Exception as exc:
                logger.exception("Error processing message")
                print(f"[red]Error processing message:[/red] {exc}")
                continue

    except Exception as exc:
        logger.exception("Error in chat mode")
        print(f"[red]Error in chat mode:[/red] {exc}")
        exit_ok = False

    finally:
        if ui:
            await _safe_cleanup(ui)
            
        # Close the manager if possible
        try:
            if isinstance(manager, ToolManager) and hasattr(manager, "close"):
                logger.debug("Closing ToolManager")
                await manager.close()
        except Exception as exc:
            logger.warning(f"Error closing ToolManager: {exc}")
            print(f"[yellow]Warning: Error closing ToolManager: {exc}[/yellow]")
            
        # encourage prompt cleanup of any lingering transports
        gc.collect()

    return exit_ok


# --------------------------------------------------------------------- #
# helpers                                                               #
# --------------------------------------------------------------------- #
async def _safe_cleanup(ui: ChatUIManager) -> None:
    """
    Run the UI-manager's cleanup coroutine (or plain function) defensively.

    Any exception during cleanup is caught and reported, never propagated.
    """
    try:
        maybe_coro = ui.cleanup()
        if asyncio.iscoroutine(maybe_coro):
            await maybe_coro
    except Exception as exc:
        logger.warning(f"Cleanup failed: {exc}")
        print(f"[yellow]Cleanup failed:[/yellow] {exc}")