# src/mcp_cli/main.py
"""Entry-point for the MCP CLI"""
from __future__ import annotations

import asyncio
import atexit
import signal
import sys

# Early environment setup
from mcp_cli.core.environment import setup_environment
from mcp_cli.core.app import create_app
from mcp_cli.ui.ui_helpers import restore_terminal

# Setup environment before any other imports
setup_environment()


def setup_signal_handlers() -> None:
    """Setup signal handlers for clean shutdown."""
    def handler(sig, _frame):
        restore_terminal()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    if hasattr(signal, "SIGQUIT"):
        signal.signal(signal.SIGQUIT, handler)


def main():
    """Main entry point."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    setup_signal_handlers()
    atexit.register(restore_terminal)
    
    # Create and run the app
    app = create_app()
    
    try:
        app()
    finally:
        # restore the terminal
        restore_terminal()

        # garbage collect
        import gc
        gc.collect()


if __name__ == "__main__":
    main()