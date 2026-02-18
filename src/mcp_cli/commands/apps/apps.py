# src/mcp_cli/commands/apps/apps.py
"""Unified /apps command — list and manage MCP Apps (interactive tool UIs)."""

from __future__ import annotations

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandParameter,
    CommandResult,
)


class AppsCommand(UnifiedCommand):
    """List tools with interactive UIs and manage running MCP Apps."""

    @property
    def name(self) -> str:
        return "apps"

    @property
    def aliases(self) -> list[str]:
        return []

    @property
    def description(self) -> str:
        return "List tools with interactive UIs and manage running MCP Apps"

    @property
    def help_text(self) -> str:
        return """
List tools that have MCP Apps interactive UIs and manage running apps.

Usage:
  /apps              - List tools with available UI apps
  /apps running      - Show currently running apps
  /apps stop         - Stop all running app servers

Examples:
  /apps              - See which tools have interactive UIs
  /apps running      - Check what app servers are active
  /apps stop         - Stop all running app servers
"""

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="subcommand",
                type=str,
                required=False,
                default="list",
                help="Subcommand: list (default), running, stop",
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the apps command."""
        from mcp_cli.context import get_context

        try:
            context = get_context()
            if not context or not context.tool_manager:
                return CommandResult(
                    success=False,
                    error="No tool manager available. Please connect to a server first.",
                )

            # Parse subcommand from args or subcommand parameter
            subcommand = kwargs.get("subcommand", "list")
            args = kwargs.get("args", "")
            if args:
                subcommand = args.strip().split()[0] if args.strip() else "list"

            if subcommand == "running":
                return await self._show_running(context.tool_manager)
            elif subcommand == "stop":
                return await self._stop_all(context.tool_manager)
            else:
                return await self._list_app_tools(context.tool_manager)

        except Exception as e:
            return CommandResult(
                success=False,
                error=f"Failed to execute apps command: {e}",
            )

    async def _list_app_tools(self, tool_manager) -> CommandResult:
        """List all tools that have MCP Apps interactive UIs."""
        from chuk_term.ui import output, format_table

        all_tools = await tool_manager.get_all_tools()
        app_tools = [t for t in all_tools if t.has_app_ui]

        if not app_tools:
            return CommandResult(
                success=True,
                output="No tools with MCP Apps interactive UIs found.",
            )

        table_data = []
        for tool in app_tools:
            table_data.append(
                {
                    "Server": tool.namespace,
                    "Tool": tool.name,
                    "UI Resource": tool.app_resource_uri or "unknown",
                    "Description": (tool.description or "")[:60],
                }
            )

        table = format_table(
            table_data,
            title=f"{len(app_tools)} Tools with Interactive UIs",
            columns=["Server", "Tool", "UI Resource", "Description"],
        )
        output.print_table(table)

        return CommandResult(success=True, data=table_data)

    async def _show_running(self, tool_manager) -> CommandResult:
        """Show currently running MCP App instances."""
        from chuk_term.ui import output, format_table

        if tool_manager._app_host is None:
            return CommandResult(
                success=True,
                output="No MCP Apps have been launched.",
            )

        running = tool_manager.app_host.get_running_apps()
        if not running:
            return CommandResult(
                success=True,
                output="No MCP Apps currently running.",
            )

        table_data = []
        for app in running:
            table_data.append(
                {
                    "Tool": app.tool_name,
                    "URL": app.url,
                    "State": app.state.value,
                    "Server": app.server_name,
                }
            )

        table = format_table(
            table_data,
            title=f"{len(running)} Running MCP Apps",
            columns=["Tool", "URL", "State", "Server"],
        )
        output.print_table(table)

        return CommandResult(success=True, data=table_data)

    async def _stop_all(self, tool_manager) -> CommandResult:
        """Stop all running MCP App servers."""
        from chuk_term.ui import output

        if tool_manager._app_host is None:
            return CommandResult(
                success=True,
                output="No MCP Apps to stop.",
            )

        count = len(tool_manager.app_host.get_running_apps())
        await tool_manager.app_host.close_all()

        msg = f"Stopped {count} MCP App(s)." if count else "No MCP Apps were running."
        output.info(msg)

        return CommandResult(success=True, output=msg)
