#!/usr/bin/env python3
"""
Entry point for python -m mcp_cli - FIXED VERSION
Updated to properly handle the new OpenAI client and chuk-tool-processor APIs.
"""

if __name__ == "__main__":
    import sys
    import asyncio

    # Set up proper event loop policy on Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        from mcp_cli.main import app

        app()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting MCP CLI: {e}")
        sys.exit(1)
