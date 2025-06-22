# mcp_cli/chat/chat_handler.py
"""
Clean chat handler that uses ModelManager and ChatContext with streaming support.
"""

from __future__ import annotations

import asyncio
import gc
import logging
from typing import Optional

from rich import print
from rich.panel import Panel
from rich.console import Console

# Local imports
from mcp_cli.chat.chat_context import ChatContext, TestChatContext
from mcp_cli.chat.ui_manager import ChatUIManager
from mcp_cli.chat.conversation import ConversationProcessor
from mcp_cli.ui.ui_helpers import clear_screen, display_welcome_banner
from mcp_cli.model_manager import ModelManager
from mcp_cli.tools.manager import ToolManager

# Set up logger
logger = logging.getLogger(__name__)


async def handle_chat_mode(
    tool_manager: ToolManager,
    provider: str = None,
    model: str = None,
    api_base: str = None,
    api_key: str = None,
) -> bool:
    """
    Launch the interactive chat loop with streaming support.

    Args:
        tool_manager: Initialized ToolManager instance
        provider: Provider to use (optional, uses ModelManager active if None)
        model: Model to use (optional, uses ModelManager active if None)
        api_base: API base URL override (optional)
        api_key: API key override (optional)

    Returns:
        True if session ended normally, False on failure
    """
    ui: Optional[ChatUIManager] = None
    console = Console()

    try:
        # Create chat context using clean factory
        with console.status("[cyan]Initializing chat context...[/cyan]", spinner="dots"):
            ctx = ChatContext.create(
                tool_manager=tool_manager,
                provider=provider,
                model=model,
                api_base=api_base,
                api_key=api_key
            )
    
            if not await ctx.initialize():
                print("[red]Failed to initialize chat context.[/red]")
                return False

        # Welcome banner
        clear_screen()
        display_welcome_banner({
            "provider": ctx.provider,
            "model": ctx.model,
        })

        # UI and conversation processor
        ui = ChatUIManager(ctx)
        convo = ConversationProcessor(ctx, ui)

        # Main chat loop with streaming support
        await _run_enhanced_chat_loop(ui, ctx, convo)
        
        return True

    except Exception as exc:
        logger.exception("Error in chat mode")
        print(f"[red]Error in chat mode:[/red] {exc}")
        return False

    finally:
        # Cleanup
        if ui:
            await _safe_cleanup(ui)
            
        # Close tool manager
        try:
            await tool_manager.close()
        except Exception as exc:
            logger.warning(f"Error closing ToolManager: {exc}")
            
        gc.collect()


async def handle_chat_mode_for_testing(
    stream_manager,
    provider: str = None,
    model: str = None,
) -> bool:
    """
    Launch chat mode for testing with stream_manager.
    
    Separated from main function to keep it clean.

    Args:
        stream_manager: Test stream manager
        provider: Provider for testing
        model: Model for testing

    Returns:
        True if session ended normally, False on failure
    """
    ui: Optional[ChatUIManager] = None
    console = Console()

    try:
        # Create test chat context
        with console.status("[cyan]Initializing test chat context...[/cyan]", spinner="dots"):
            ctx = TestChatContext.create_for_testing(
                stream_manager=stream_manager,
                provider=provider,
                model=model
            )
    
            if not await ctx.initialize():
                print("[red]Failed to initialize test chat context.[/red]")
                return False

        # Welcome banner
        clear_screen()
        display_welcome_banner({
            "provider": ctx.provider,
            "model": ctx.model,
        })

        # UI and conversation processor
        ui = ChatUIManager(ctx)
        convo = ConversationProcessor(ctx, ui)

        # Main chat loop with streaming support
        await _run_enhanced_chat_loop(ui, ctx, convo)
        
        return True

    except Exception as exc:
        logger.exception("Error in test chat mode")
        print(f"[red]Error in test chat mode:[/red] {exc}")
        return False

    finally:
        if ui:
            await _safe_cleanup(ui)
        gc.collect()


async def _run_enhanced_chat_loop(ui: ChatUIManager, ctx: ChatContext, convo: ConversationProcessor) -> None:
    """
    Run the main chat loop with enhanced streaming support.
    
    Args:
        ui: UI manager with streaming coordination
        ctx: Chat context  
        convo: Conversation processor with streaming support
    """
    while True:
        try:
            user_msg = await ui.get_user_input()

            # Skip empty messages
            if not user_msg:
                continue

            # Handle exit commands
            if user_msg.lower() in ("exit", "quit"):
                print(Panel("Exiting chat mode.", style="bold red"))
                break

            # Handle slash commands
            if user_msg.startswith("/"):
                # Special handling for interrupt command during streaming
                if user_msg.lower() in ("/interrupt", "/stop", "/cancel"):
                    if ui.is_streaming_response:
                        ui.interrupt_streaming()
                        print("[yellow]Streaming response interrupted.[/yellow]")
                        continue
                    elif ui.tools_running:
                        ui._interrupt_now()
                        continue
                    else:
                        print("[yellow]Nothing to interrupt.[/yellow]")
                        continue
                
                handled = await ui.handle_command(user_msg)
                if ctx.exit_requested:
                    break
                if handled:
                    continue

            # Normal conversation turn with streaming support
            ui.print_user_message(user_msg)
            ctx.add_user_message(user_msg)
            
            # Use the enhanced conversation processor that handles streaming
            await convo.process_conversation()

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            if ui.is_streaming_response:
                print("\n[yellow]Streaming interrupted - type 'exit' to quit.[/yellow]")
                ui.interrupt_streaming()
            elif ui.tools_running:
                print("\n[yellow]Tool execution interrupted - type 'exit' to quit.[/yellow]")
                ui._interrupt_now()
            else:
                print("\n[yellow]Interrupted - type 'exit' to quit.[/yellow]")
        except EOFError:
            print(Panel("EOF detected - exiting chat.", style="bold red"))
            break
        except Exception as exc:
            logger.exception("Error processing message")
            print(f"[red]Error processing message:[/red] {exc}")
            continue


async def _safe_cleanup(ui: ChatUIManager) -> None:
    """
    Safely cleanup UI manager with enhanced error handling.

    Args:
        ui: UI manager to cleanup
    """
    try:
        # Stop any streaming responses
        if ui.is_streaming_response:
            ui.interrupt_streaming()
            ui.stop_streaming_response()
            
        # Stop any tool execution
        if ui.tools_running:
            ui.stop_tool_calls()
            
        # Standard cleanup
        cleanup_result = ui.cleanup()
        if asyncio.iscoroutine(cleanup_result):
            await cleanup_result
    except Exception as exc:
        logger.warning(f"Cleanup failed: {exc}")
        print(f"[yellow]Cleanup failed:[/yellow] {exc}")


# ═══════════════════════════════════════════════════════════════════════════════════
# Enhanced interrupt command for chat mode
# ═══════════════════════════════════════════════════════════════════════════════════

async def handle_interrupt_command(ui: ChatUIManager) -> bool:
    """
    Handle the /interrupt command with streaming awareness.
    
    Args:
        ui: UI manager instance
        
    Returns:
        True if command was handled
    """
    if ui.is_streaming_response:
        ui.interrupt_streaming()
        print("[yellow]Streaming response interrupted.[/yellow]")
    elif ui.tools_running:
        ui._interrupt_now()
        print("[yellow]Tool execution interrupted.[/yellow]")
    else:
        print("[yellow]Nothing currently running to interrupt.[/yellow]")
    
    return True


# ═══════════════════════════════════════════════════════════════════════════════════
# Legacy wrapper for backward compatibility (can be removed eventually)
# ═══════════════════════════════════════════════════════════════════════════════════

async def handle_chat_mode_legacy(
    manager,  # ToolManager or stream_manager
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_base: str = None,
    api_key: str = None,
    **kwargs  # Ignore other legacy parameters
) -> bool:
    """
    Legacy wrapper for backward compatibility.
    
    This can be removed once all callers are updated.
    """
    import warnings
    warnings.warn(
        "handle_chat_mode_legacy is deprecated, use handle_chat_mode or handle_chat_mode_for_testing",
        DeprecationWarning,
        stacklevel=2
    )
    
    if isinstance(manager, ToolManager):
        return await handle_chat_mode(
            tool_manager=manager,
            provider=provider,
            model=model,
            api_base=api_base,
            api_key=api_key
        )
    else:
        # Assume test mode
        return await handle_chat_mode_for_testing(
            stream_manager=manager,
            provider=provider,
            model=model
        )


# ═══════════════════════════════════════════════════════════════════════════════════
# Usage examples:
# ═══════════════════════════════════════════════════════════════════════════════════

"""
# Production usage with streaming:
success = await handle_chat_mode(
    tool_manager,
    provider="anthropic",
    model="claude-3-sonnet",
    api_key="your-key"
)

# Test usage:
success = await handle_chat_mode_for_testing(
    stream_manager,
    provider="openai",
    model="gpt-4"
)

# Simple usage (uses ModelManager defaults):
success = await handle_chat_mode(tool_manager)
"""