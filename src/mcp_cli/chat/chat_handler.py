# mcp_cli/chat/chat_handler.py
import asyncio
import sys
import gc
from rich import print
from rich.panel import Panel

# mcp cli imports
from mcp_cli.chat.chat_context import ChatContext
from mcp_cli.chat.ui_manager import ChatUIManager
from mcp_cli.chat.conversation import ConversationProcessor
from mcp_cli.ui.ui_helpers import display_welcome_banner, clear_screen

# Import StreamManager (now mandatory)
from mcp_cli.stream_manager import StreamManager

async def handle_chat_mode(stream_manager, provider="openai", model="gpt-4o-mini"):
    """
    Enter chat mode with multi-call support for autonomous tool chaining.
    
    Args:
        stream_manager: StreamManager instance (required)
        provider: LLM provider name (default: "openai")
        model: LLM model name (default: "gpt-4o-mini")
    """
    ui_manager = None
    exit_code = 0
    
    try:
        # Clear the screen to start fresh
        clear_screen()
        
        # Initialize chat context with StreamManager
        chat_context = ChatContext(stream_manager, provider, model)
        
        if not await chat_context.initialize():
            return False
            
        # Display the welcome banner (and show tools info here only)
        display_welcome_banner(chat_context.to_dict())
        
        # Initialize UI manager
        ui_manager = ChatUIManager(chat_context)
        
        # Initialize conversation processor
        conv_processor = ConversationProcessor(chat_context, ui_manager)
        
        # Main chat loop
        while True:
            try:
                # Get user input
                user_message = await ui_manager.get_user_input()
                
                # Handle empty messages
                if not user_message:
                    continue
                
                # Handle exit/quit commands - avoid direct sys.exit()
                if user_message.lower() in ["exit", "quit"]:
                    print(Panel("Exiting chat mode.", style="bold red"))
                    break  # Exit the loop cleanly instead of sys.exit()
                
                # Handle special commands
                if user_message.startswith('/'):
                    # Process command and check if an exit was requested
                    if await ui_manager.handle_command(user_message):
                        if chat_context.exit_requested:
                            break  # Exit the loop cleanly instead of sys.exit()
                        continue
                
                # Display user message in the styled panel
                ui_manager.print_user_message(user_message)

                # Add user message to history
                chat_context.conversation_history.append(
                    {"role": "user", "content": user_message}
                )
                
                # Process conversation
                await conv_processor.process_conversation()

            except KeyboardInterrupt:
                print("\n[yellow]Chat interrupted. Type 'exit' to quit.[/yellow]")
            except EOFError:
                # EOF (Ctrl+D) should exit cleanly
                print(Panel("EOF detected. Exiting chat mode.", style="bold red"))
                break  # Exit the loop cleanly instead of sys.exit()
            except Exception as e:
                print(f"[red]Error processing message:[/red] {e}")
                continue
    except asyncio.CancelledError:
        # Handle task cancellation gracefully
        print("[yellow]Chat task cancelled.[/yellow]")
    except Exception as e:
        print(f"[red]Error in chat mode:[/red] {e}")
        exit_code = 1
    finally:
        # Clean up all resources in order
        
        # 1. First clean up UI manager
        if ui_manager:
            await _safe_cleanup(ui_manager)
        
        # 2. Force garbage collection to run before exit
        gc.collect()
    
    return exit_code == 0  # Return success status

# Helper functions for safer resource cleanup
async def _safe_cleanup(ui_manager):
    """Safely cleanup UI manager resources."""
    try:
        # Check if cleanup is a coroutine function
        if hasattr(ui_manager, 'cleanup'):
            if asyncio.iscoroutinefunction(ui_manager.cleanup):
                await ui_manager.cleanup()
            else:
                ui_manager.cleanup()
    except Exception as e:
        print(f"[red]Error during UI cleanup:[/red] {e}")