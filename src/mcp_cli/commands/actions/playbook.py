"""Playbook command actions."""

from __future__ import annotations

import logging
from typing import Any

from chuk_term.ui import output
from mcp_cli.commands.models.playbook import PlaybookActionParams
from mcp_cli.utils.preferences import get_preference_manager
from mcp_cli.context import get_context

logger = logging.getLogger(__name__)


async def playbook_action_async(params: PlaybookActionParams) -> str | dict[str, Any]:
    """Execute playbook action asynchronously.

    Args:
        params: Playbook action parameters

    Returns:
        Result string or dictionary
    """
    pref_manager = get_preference_manager()
    action = params.action.lower()

    # Handle enable/disable actions
    if action == "enable":
        pref_manager.set_playbook_enabled(True)
        output.success("✓ Playbook integration enabled")
        output.warning(
            "Note: Please restart your session for changes to take effect"
        )
        output.hint("The playbook server will be bound on next startup")
        return "Playbook enabled (restart required)"

    elif action == "disable":
        pref_manager.set_playbook_enabled(False)
        output.success("✓ Playbook integration disabled")
        output.warning(
            "Note: Please restart your session for changes to take effect"
        )
        output.hint("The playbook server will not be bound on next startup")
        return "Playbook disabled (restart required)"

    elif action == "status":
        enabled = pref_manager.is_playbook_enabled()
        server_name = pref_manager.get_playbook_server_name()
        top_k = pref_manager.get_playbook_top_k()
        local_dir = pref_manager.get_local_playbook_dir()

        status_str = "enabled" if enabled else "disabled"
        output.print(f"\n[bold]Playbook Status:[/bold]")
        output.print(f"  Status: [{'green' if enabled else 'red'}]{status_str}[/]")
        output.print(f"  Server: {server_name}")
        output.print(f"  Top K: {top_k}")
        if local_dir:
            output.print(f"  Local playbooks: {local_dir}")
        else:
            output.print(f"  Local playbooks: [dim]not configured[/dim]")

        if not enabled:
            output.hint("Use '/playbook enable' to enable playbook integration")

        return f"Playbook is {status_str}"

    # For actions that need the playbook service
    elif action == "list":
        if not pref_manager.is_playbook_enabled():
            output.error("Playbook integration is disabled")
            output.hint("Use '/playbook enable' to enable it")
            return "Playbook integration is disabled"

        # Get tool manager from context
        context = get_context()
        tool_manager = context.get("tool_manager")

        if not tool_manager:
            output.error("No tool manager available")
            output.hint("Playbook commands require an active session")
            return "No tool manager available"

        # Import and use PlaybookService
        from mcp_cli.services import PlaybookService

        playbook_service = PlaybookService(tool_manager)

        playbooks = await playbook_service.list_playbooks()
        if not playbooks:
            output.warning("No playbooks found")
            return "No playbooks available"

        output.print(f"\n[bold]Available Playbooks ({len(playbooks)}):[/bold]\n")
        for idx, title in enumerate(playbooks, 1):
            output.print(f"  {idx}. {title}")

        output.hint("\nTo use a playbook, just ask the LLM your question")
        output.hint("Example: 'How do I get sunset times?'")

        return f"Found {len(playbooks)} playbooks"

    else:
        output.error(f"Unknown action: {action}")
        output.hint(
            "Valid actions: enable, disable, status, list"
        )
        return f"Unknown action: {action}"
