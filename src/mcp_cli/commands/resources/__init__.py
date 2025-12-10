"""Resource management commands."""

from mcp_cli.commands.resources.resources import ResourcesCommand
from mcp_cli.commands.resources.prompts import PromptsCommand

__all__ = [
    "ResourcesCommand",
    "PromptsCommand",
]
