# mcp_cli/chat/chat_handler.py
"""
mcp_cli.chat.chat_handler
=========================

High-level entry point for the MCP-CLI “chat” mode.
Initialises a ChatContext, sets up the TUI, then enters the
main read–eval–print loop.

The module contains *no* model-specific code; all LLM access is delegated
to the llm/ sub-package through ChatContext.
"""

from __future__ import annotations

import asyncio
import gc
from typing import Any, Optional

from rich import print
from rich.panel import Panel

# ── local imports ──────────────────────────────────────────────────────
from mcp_cli.chat.chat_context import ChatContext
from mcp_cli.chat.ui_manager import ChatUIManager
from mcp_cli.chat.conversation import ConversationProcessor
from mcp_cli.ui.ui_helpers import clear_screen, display_welcome_banner

from mcp_cli.tools.manager import ToolManager

# --------------------------------------------------------------------- #
# public helper                                                         #
# --------------------------------------------------------------------- #
async def handle_chat_mode(          # ← provider / model can be positional
    manager: Any,                    # ToolManager *or* stream-manager stub
    provider: str = "openai",
    model: str = "gpt-4o-mini",
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

    Returns
    -------
    bool
        *True* if the session ended normally, *False* on hard failure.
    """
    ui: Optional[ChatUIManager] = None
    exit_ok = True

    try:
        # ── build chat context ─────────────────────────────────────────
        if isinstance(manager, ToolManager):
            ctx = ChatContext(tool_manager=manager, provider=provider, model=model)
        else:
            # assume test stub
            ctx = ChatContext(stream_manager=manager, provider=provider, model=model)

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

                # plain “exit/quit” terminates chat
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
            except Exception as exc:                       # noqa: BLE001
                print(f"[red]Error processing message:[/red] {exc}")
                import traceback
                traceback.print_exc()
                continue

    except Exception as exc:                               # noqa: BLE001
        print(f"[red]Error in chat mode:[/red] {exc}")
        import traceback
        traceback.print_exc()
        exit_ok = False

    finally:
        if ui:
            await _safe_cleanup(ui)
        # encourage prompt cleanup of any lingering transports
        gc.collect()

    return exit_ok


# --------------------------------------------------------------------- #
# helpers                                                               #
# --------------------------------------------------------------------- #
async def _safe_cleanup(ui: ChatUIManager) -> None:
    """
    Run the UI-manager’s cleanup coroutine (or plain function) defensively.

    Any exception during cleanup is caught and reported, never propagated.
    """
    try:
        maybe_coro = ui.cleanup()
        if asyncio.iscoroutine(maybe_coro):
            await maybe_coro
    except Exception as exc:                              # noqa: BLE001
        print(f"[yellow]Cleanup failed:[/yellow] {exc}")
