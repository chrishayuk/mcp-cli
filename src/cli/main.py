# src/cli/main.py
import logging
import sys
import typer
import atexit
import os

# mcp imports
from mcp.transport.stdio.stdio_client import stdio_client
from mcp.messages.initialize.send_messages import send_initialize

# cli imports
from cli.commands.register_commands import register_commands, chat_command
from cli.cli_options import process_options

# host imports
from host.server_manager import run_command

# Configure logging
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

# Ensure terminal is reset on exit.
def restore_terminal():
    # restor the terminal
    os.system("stty sane")

# restore the terminal on exist
atexit.register(restore_terminal)

# Create Typer app instance
app = typer.Typer()

# Register commands by passing in the helper functions.
register_commands(app, process_options, run_command)

# Global Options Callback
@app.callback(invoke_without_command=True)
def common_options(
    ctx: typer.Context,
    config_file: str = "server_config.json",
    server: str = None,
    provider: str = "openai",
    model: str = None,
    disable_filesystem: bool = False,
):
    """
    MCP Command-Line Tool

    Global options are specified here.
    If no subcommand is provided, chat mode is launched by default.
    """
    # process the options, getting the servers etc
    servers, user_specified = process_options(server, disable_filesystem, provider, model)
    
    # set the context
    ctx.obj = {
        "config_file": config_file,
        "servers": servers,
        "user_specified": user_specified,
    }
    
    # check if a sub command is invoked
    if ctx.invoked_subcommand is None:
        # Call the chat command (imported from the commands module)
        chat_command(
            config_file=config_file,
            server=server,
            provider=provider,
            model=model,
            disable_filesystem=disable_filesystem,
        )

        # exit
        raise typer.Exit()

if __name__ == "__main__":
    try:
        # start the app
        app()
    finally:
        # restore the terminal
        restore_terminal()
