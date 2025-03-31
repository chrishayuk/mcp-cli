# src/cli/chat/chat_handler.py
import asyncio
import sys
import gc
from subprocess import Popen
from rich import print
from rich.panel import Panel

from cli.chat.chat_context import ChatContext
from cli.chat.ui_manager import ChatUIManager
from cli.chat.conversation import ConversationProcessor
from cli.ui.ui_helpers import display_welcome_banner, clear_screen

async def handle_chat_mode(server_streams, provider="ollama", model="llama3.2"):
    """Enter chat mode with multi-call support for autonomous tool chaining."""
    ui_manager = None
    exit_code = 0
    active_subprocesses = []  # Track subprocess instances
    
    try:
        # Clear the screen to start fresh
        clear_screen()
        
        # Initialize chat context
        chat_context = ChatContext(server_streams, provider, model)
        if not await chat_context.initialize():
            return False
            
        # Track subprocesses created during initialization
        _collect_subprocesses(active_subprocesses)
        
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
                
                # Track any new subprocesses created during processing
                _collect_subprocesses(active_subprocesses)

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
        
        # 2. Clean up explicit subprocess instances
        _cleanup_subprocesses(active_subprocesses)
        
        # 3. Clean up server streams
        for stream in server_streams:
            await _safe_close(stream)
            
        # 4. Clean up transports in event loop
        _cleanup_transports()
        
        # 5. Force garbage collection to run before exit
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

async def _safe_close(stream):
    """Safely close a stream resource."""
    try:
        if hasattr(stream, 'close'):
            if asyncio.iscoroutinefunction(stream.close):
                await stream.close()
            else:
                stream.close()
                
        # If it has a process attribute, terminate it
        if hasattr(stream, 'process') and stream.process is not None:
            if hasattr(stream.process, 'terminate'):
                stream.process.terminate()
                try:
                    stream.process.wait(timeout=0.5)
                except:
                    if hasattr(stream.process, 'kill'):
                        stream.process.kill()
    except Exception as e:
        print(f"[red]Error closing stream:[/red] {e}")

def _collect_subprocesses(active_subprocesses):
    """Collect all active subprocess.Popen instances."""
    for obj in gc.get_objects():
        if isinstance(obj, Popen) and obj.poll() is None:  # Still running
            if obj not in active_subprocesses:
                active_subprocesses.append(obj)

def _cleanup_subprocesses(active_subprocesses):
    """Terminate all tracked subprocess instances."""
    for proc in active_subprocesses:
        try:
            if proc.poll() is None:  # Process still running
                proc.terminate()
                try:
                    proc.wait(timeout=0.5)  # Short timeout
                except:
                    proc.kill()  # Force kill if terminate doesn't work
        except Exception as e:
            print(f"[red]Error cleaning up subprocess:[/red] {e}")

def _cleanup_transports():
    """Clean up transports in the event loop."""
    try:
        loop = asyncio.get_event_loop()
        if not loop.is_closed():
            # Clean up transports
            for transport in getattr(loop, '_transports', set()):
                if hasattr(transport, 'close'):
                    try:
                        transport.close()
                    except Exception:
                        pass
                    
            # Find and clean up subprocess transports specifically
            for obj in gc.get_objects():
                if hasattr(obj, '__class__') and 'SubprocessTransport' in obj.__class__.__name__:
                    # Disable the pipe to prevent EOF writing
                    if hasattr(obj, '_protocol') and obj._protocol is not None:
                        if hasattr(obj._protocol, 'pipe'):
                            obj._protocol.pipe = None
    except Exception as e:
        print(f"[red]Error cleaning up transports:[/red] {e}")