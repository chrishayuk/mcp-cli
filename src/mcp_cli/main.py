# mcp_cli/main.py
import logging
import sys
import typer
import atexit
import os
import asyncio
import signal
import gc

# mcp imports
from chuk_mcp.mcp_client.transport.stdio.stdio_client import stdio_client
from chuk_mcp.mcp_client.messages.initialize.send_messages import send_initialize

# cli imports
from mcp_cli.commands.register_commands import register_commands, chat_command
from mcp_cli.cli_options import process_options

# Import our improved run_command and StreamManager
from mcp_cli.run_command import run_command
from mcp_cli.stream_manager import StreamManager

# Configure logging without setting a fixed level here.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

# Ensure terminal is reset on exit.
def restore_terminal():
    """Restore terminal settings and clean up asyncio resources."""
    # Restore the terminal settings to normal.
    os.system("stty sane")
    
    # Then clean up asyncio tasks and the event loop
    try:
        if hasattr(asyncio, "all_tasks"):
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if not loop.is_closed():
                tasks = asyncio.all_tasks(loop=loop)
                if tasks:
                    for task in tasks:
                        task.cancel()
                    try:
                        if sys.version_info >= (3, 7):
                            loop.run_until_complete(asyncio.wait(tasks, timeout=0.5))
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                try:
                    loop.run_until_complete(asyncio.sleep(0))
                except (RuntimeError, asyncio.CancelledError):
                    pass
                loop.close()
    except Exception as e:
        logging.debug(f"Error during asyncio cleanup: {e}")
    
    gc.collect()

# Register terminal restore on exit.
atexit.register(restore_terminal)

# Create Typer app instance
app = typer.Typer()

# Register commands by passing in the helper functions.
register_commands(app, process_options, run_command)

# Set up signal handlers for cleaner shutdown
def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(sig, frame):
        logging.debug(f"Received signal {sig}")
        restore_terminal()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if hasattr(signal, 'SIGQUIT'):
        signal.signal(signal.SIGQUIT, signal_handler)

# Global Options Callback with logging_level CLI option
@app.callback(invoke_without_command=True)
def common_options(
    ctx: typer.Context,
    config_file: str = "server_config.json",
    server: str = None,
    provider: str = "openai",
    model: str = None,
    disable_filesystem: bool = True,
    logging_level: str = typer.Option(
        "WARNING",
        help="Set the logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    ),
):
    """
    MCP Command-Line Tool

    Global options are specified here.
    If no subcommand is provided, chat mode is launched by default.
    """
    # Convert the logging level string to a numeric value.
    numeric_level = getattr(logging, logging_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {logging_level}")
    logging.getLogger().setLevel(numeric_level)
    logging.debug(f"Logging level set to {logging_level.upper()}")

    # Process options to get servers and related configuration.
    servers, user_specified, server_names = process_options(server, disable_filesystem, provider, model, config_file)
    
    # Set the context.
    ctx.obj = {
        "config_file": config_file,
        "servers": servers,
        "user_specified": user_specified,
        "server_names": server_names,
    }
    
    # If no subcommand was invoked, launch chat mode.
    if ctx.invoked_subcommand is None:
        setup_signal_handlers()
        chat_command(
            config_file=config_file,
            server=server,
            provider=provider,
            model=model,
            disable_filesystem=disable_filesystem,
        )
        restore_terminal()
        raise typer.Exit()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        setup_signal_handlers()
        app()
    except KeyboardInterrupt:
        logging.debug("KeyboardInterrupt received")
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
    finally:
        restore_terminal()
