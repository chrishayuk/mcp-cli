"""mcp_cli.run_command
Utilities for running commands with proper setup and cleanup.
"""
import asyncio
import logging
from typing import Callable, List, Dict, Any, Optional

from mcp_cli.stream_manager import StreamManager

async def run_command_async(
    command_func: Callable[..., Any],
    config_file: str,
    servers: List[str],
    user_specified: List[str],
    extra_params: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Run an async command with StreamManager setup and teardown.

    Args:
        command_func: The command coroutine to execute.
        config_file: Path to the server configuration file.
        servers: List of server names to connect to.
        user_specified: User-specified server list (for logging or future use).
        extra_params: Extra parameters to pass to the command function.

    Returns:
        The result of command_func, or False if no servers were provided.
    """
    logging.info("Running command: %s", command_func.__name__)
    logging.debug("Servers: %s", servers)

    if not servers:
        logging.warning("No servers specified!")
        return False

    logging.info("Initializing servers: %s", servers)
    stream_manager = await StreamManager.create(
        config_file=config_file,
        servers=servers,
        server_names={i: name for i, name in enumerate(servers)},
    )

    try:
        params = dict(extra_params or {})
        params["stream_manager"] = stream_manager
        return await command_func(**params)
    finally:
        logging.debug("Tearing down StreamManager")
        await stream_manager.close()


def run_command(
    command_func: Callable[..., Any],
    config_file: str,
    servers: List[str],
    user_specified: List[str],
    extra_params: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Synchronous wrapper for run_command_async.

    Uses asyncio.run when possible, otherwise falls back to run_until_complete.
    """
    try:
        # Try to run in a new event loop
        return asyncio.run(
            run_command_async(
                command_func,
                config_file,
                servers,
                user_specified,
                extra_params,
            )
        )
    except RuntimeError as e:
        # Distinguish between command errors and nested loop errors
        msg = str(e)
        if "asyncio.run() cannot be called from a running event loop" not in msg:
            logging.error("Error running command (asyncio.run error): %s", e)
            return False
        # Fallback to existing loop
        logging.debug("Existing event loop detected; using run_until_complete")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logging.error("No running event loop available for fallback")
            return False
        try:
            return loop.run_until_complete(
                run_command_async(
                    command_func,
                    config_file,
                    servers,
                    user_specified,
                    extra_params,
                )
            )
        except KeyboardInterrupt:
            logging.debug("KeyboardInterrupt received in sync wrapper")
            return False
        except Exception as e2:
            logging.error("Error running command in existing loop: %s", e2)
            return False
    except KeyboardInterrupt:
        logging.debug("KeyboardInterrupt received in sync wrapper")
        return False
    except Exception as e:
        logging.error("Error running command: %s", e)
        return False
