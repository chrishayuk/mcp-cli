# src/cli/commands/chat.py
import os
import typer
import asyncio
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel

# imports
from cli.chat.chat_handler import handle_chat_mode

# app
app = typer.Typer(help="Chat commands")

@app.command("run")
async def chat_run(server_streams: list):
    """Enter chat mode."""
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    os.system("cls" if os.name == "nt" else "clear")
    chat_info_text = (
        "Welcome to the Chat!\n\n"
        f"**Provider:** {provider}  |  **Model:** {model}\n\n"
        "Type 'exit' to quit."
    )
    print(Panel(Markdown(chat_info_text), style="bold cyan", title="Chat Mode", title_align="center"))
    
    try:
        # Create a task for the chat handler
        chat_task = asyncio.create_task(handle_chat_mode(server_streams, provider, model))
        
        # Await the task with proper exception handling
        await chat_task
    except KeyboardInterrupt:
        print("\nChat interrupted by user.")
    except Exception as e:
        print(f"\nError in chat mode: {e}")
    
    # Don't print exit message here, since it's already printed in the handler
    
    # Make sure any pending tasks are properly cancelled
    if 'chat_task' in locals() and not chat_task.done():
        chat_task.cancel()
        try:
            # Give it a short time to cancel gracefully
            await asyncio.wait_for(asyncio.shield(chat_task), timeout=1.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            # Expected when cancelling or timing out
            pass
        except Exception as e:
            # Only print this if it's not a typical cancellation error
            if not isinstance(e, (asyncio.CancelledError, RuntimeError)):
                print(f"Error during chat cleanup: {e}")
    
    # Signal a clean exit to the main process
    return True
