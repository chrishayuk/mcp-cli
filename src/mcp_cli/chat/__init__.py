# mcp_cli/chat/__init__.py
"""Chat module for mcp-cli.

Exports are lazy to avoid circular imports with tools.manager.
Use: from mcp_cli.chat.chat_handler import handle_chat_mode
"""


def __getattr__(name: str):
    """Lazy import to avoid circular dependencies."""
    if name == "handle_chat_mode":
        from mcp_cli.chat.chat_handler import handle_chat_mode

        return handle_chat_mode
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["handle_chat_mode"]
