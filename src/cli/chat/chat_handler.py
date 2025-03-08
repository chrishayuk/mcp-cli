# src/cli/chat/chat_handler.py
import asyncio
from rich import print
from rich.panel import Panel

from cli.chat.chat_context import ChatContext
from cli.chat.ui_manager import ChatUIManager
from cli.chat.conversation import ConversationProcessor

async def handle_chat_mode(server_streams, provider="openai", model="gpt-4o-mini"):
    """Enter chat mode with multi-call support for autonomous tool chaining."""
    try:
        # Initialize chat context
        chat_context = ChatContext(server_streams, provider, model)
        if not await chat_context.initialize():
            return False
        
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
                
                # Handle exit/quit commands
                if user_message.lower() in ["exit", "quit"]:
                    print(Panel("Exiting chat mode.", style="bold red"))
                    break
                
                # Handle special commands
                if user_message.startswith('/'):
                    # Process command and check if an exit was requested
                    if await ui_manager.handle_command(user_message):
                        if chat_context.exit_requested:
                            break
                        continue
                
                # Display user message
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
                break
            except Exception as e:
                print(f"[red]Error processing message:[/red] {e}")
                continue
    except asyncio.CancelledError:
        # Handle task cancellation gracefully
        print("[yellow]Chat task cancelled.[/yellow]")
    except Exception as e:
        print(f"[red]Error in chat mode:[/red] {e}")
    
    # Clean return to ensure proper exit
    return True