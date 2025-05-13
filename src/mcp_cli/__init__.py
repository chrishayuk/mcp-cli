# mcp_cli/__init__.py
"""
MCP-CLI package root.

Early-loads environment variables from a .env file so that provider
adapters (OpenAI, Groq, Anthropic, …) can read API keys via `os.getenv`
without the caller having to export them in the shell.

If python-dotenv isn’t installed, we just continue silently.

Nothing else should be imported from here to keep side-effects minimal.
"""
from __future__ import annotations

import logging

try:
    from dotenv import load_dotenv
    loaded = load_dotenv()  # returns True if a .env file was found
    if loaded:
        logging.getLogger(__name__).debug(".env loaded successfully")
except ModuleNotFoundError:
    # python-dotenv not installed — .env support disabled
    logging.getLogger(__name__).debug("python-dotenv not installed; skipping .env load")
