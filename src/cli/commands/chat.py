# src/cli/commands/chat.py
import os
import typer
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel

# imports
from mcpcli.chat_handler import handle_chat_mode

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
    await handle_chat_mode(server_streams, provider, model)
