# mcp_cli/chat/__main__.py
import logging
import sys
import typer
import atexit
import os
import asyncio
import signal
import gc

# Updated imports for new chuk-mcp APIs

# cli imports
from mcp_cli.commands.register_commands import register_commands, chat_command
from mcp_cli.cli_options import process_options

# host imports - this may need updating depending on the new API structure
# Note: You may need to verify if this import path is still correct in the new APIs
from chuk_mcp.mcp_client.host.server_manager import run_command

# Configure logging
logging.basicConfig(
    level=logging.CRITICAL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)


# Ensure terminal is reset on exit.
def restore_terminal():
    """Restore terminal settings and clean up asyncio resources with special
    attention to subprocess transports."""
    # Restore the terminal settings to normal.
    os.system("stty sane")

    # First, try to explicitly clean up subprocess transports
    try:
        # Find and close all subprocess transports before the event loop is closed
        for obj in gc.get_objects():
            if (
                hasattr(obj, "__class__")
                and "SubprocessTransport" in obj.__class__.__name__
            ):
                if hasattr(obj, "_proc") and obj._proc is not None:
                    try:
                        # Close the subprocess if it's still running
                        if obj._proc.poll() is None:
                            obj._proc.kill()
                            obj._proc.wait(timeout=0.5)  # Short timeout
                    except Exception as e:
                        logging.debug(f"Error killing subprocess: {e}")

                # Mark internal pipe as closed to prevent EOF writing
                if hasattr(obj, "_protocol") and obj._protocol is not None:
                    if hasattr(obj._protocol, "pipe"):
                        obj._protocol.pipe = None
    except Exception as e:
        logging.debug(f"Error during subprocess cleanup: {e}")

    # Then clean up asyncio tasks and the event loop
    try:
        # Only attempt to clean up asyncio resources if we're in the main thread
        if hasattr(asyncio, "all_tasks"):
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if not loop.is_closed():
                # Get and cancel all tasks
                tasks = asyncio.all_tasks(loop=loop)
                if tasks:
                    for task in tasks:
                        task.cancel()

                    # Wait for tasks to be cancelled (with timeout)
                    try:
                        if sys.version_info >= (3, 7):
                            loop.run_until_complete(asyncio.wait(tasks, timeout=0.5))
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass  # Expected during cancellation

                # Run the loop one last time to complete any pending callbacks
                try:
                    loop.run_until_complete(asyncio.sleep(0))
                except (RuntimeError, asyncio.CancelledError):
                    pass

                # Close the loop
                loop.close()
    except Exception as e:
        logging.debug(f"Error during asyncio cleanup: {e}")

    # Force garbage collection to ensure __del__ methods run while we can still handle them
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

    # Register the signal handler for common termination signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # On non-Windows platforms, also handle SIGQUIT
    if hasattr(signal, "SIGQUIT"):
        signal.signal(signal.SIGQUIT, signal_handler)


# Global Options Callback
@app.callback(invoke_without_command=True)
def common_options(
    ctx: typer.Context,
    config_file: str = "server_config.json",
    server: str = None,
    provider: str = "openai",
    model: str = None,
    disable_filesystem: bool = True,
):
    """
    MCP Command-Line Tool

    Global options are specified here.
    If no subcommand is provided, chat mode is launched by default.
    """
    # Process the options, getting the servers, etc.
    servers, user_specified = process_options(
        server, disable_filesystem, provider, model
    )

    # Set the context.
    ctx.obj = {
        "config_file": config_file,
        "servers": servers,
        "user_specified": user_specified,
    }

    # Check if a subcommand was invoked.
    if ctx.invoked_subcommand is None:
        # Set up signal handlers before entering chat mode
        setup_signal_handlers()

        # Call the chat command (imported from the commands module)
        chat_command(
            config_file=config_file,
            server=server,
            provider=provider,
            model=model,
            disable_filesystem=disable_filesystem,
        )

        # Make sure any asyncio cleanup is done
        restore_terminal()

        # Exit chat mode.
        raise typer.Exit()


if __name__ == "__main__":
    # Set up platform-specific event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        # Set up signal handlers
        setup_signal_handlers()

        # Start the Typer app.
        app()
    except KeyboardInterrupt:
        logging.debug("KeyboardInterrupt received")
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
    finally:
        # Restore the terminal upon exit.
        restore_terminal()
