# src/mcp_cli/commands/tools/tool_history.py
"""
Unified tool history command implementation (chat mode only).

Reads from procedural memory (ToolMemoryManager.tool_log) instead of
maintaining a redundant in-memory list.
"""

from __future__ import annotations

import json

from mcp_cli.commands.base import (
    UnifiedCommand,
    CommandMode,
    CommandParameter,
    CommandResult,
)
from chuk_term.ui import output, format_table


class ToolHistoryCommand(UnifiedCommand):
    """View history of tool calls in this session."""

    @property
    def name(self) -> str:
        return "toolhistory"

    @property
    def aliases(self) -> list[str]:
        return ["th"]

    @property
    def description(self) -> str:
        return "View history of tool calls in this session"

    @property
    def help_text(self) -> str:
        return """
Inspect the history of tool calls executed during this chat session.

Usage:
  /toolhistory              - Show all tool calls in a table
  /toolhistory <row>        - Show detailed view of specific call
  /toolhistory -n 10        - Show last 10 calls only
  /toolhistory --json       - Export as JSON

Options:
  <row>         - Row number for detailed view (e.g., 1, 2, 3)
  -n <count>    - Limit to last N entries
  --json        - Output as JSON

Examples:
  /toolhistory              - Table of all calls
  /toolhistory 3            - Full details for call #3
  /toolhistory -n 5         - Last five calls
  /toolhistory --json       - JSON dump of all calls

Note: This command is only available in chat mode.
"""

    @property
    def modes(self) -> CommandMode:
        """This is a chat-only command."""
        return CommandMode.CHAT

    @property
    def parameters(self) -> list[CommandParameter]:
        return [
            CommandParameter(
                name="row",
                type=int,
                required=False,
                help="Row number for detailed view",
            ),
            CommandParameter(
                name="n",
                type=int,
                required=False,
                help="Limit to last N entries",
            ),
            CommandParameter(
                name="json",
                type=bool,
                default=False,
                help="Output as JSON",
                is_flag=True,
            ),
        ]

    async def execute(self, **kwargs) -> CommandResult:
        """Execute the tool history command."""
        # Get chat context
        chat_context = kwargs.get("chat_context")
        if not chat_context:
            return CommandResult(
                success=False,
                error="Tool history command requires chat context.",
            )

        # Get tool log from procedural memory
        if not hasattr(chat_context, "tool_memory"):
            return CommandResult(
                success=True,
                output="No tool history available.",
            )

        tool_log = chat_context.tool_memory.memory.tool_log
        if not tool_log:
            return CommandResult(
                success=True,
                output="No tool calls have been made yet.",
            )

        # Get parameters
        row_num = kwargs.get("row")
        limit = kwargs.get("n")
        show_json = kwargs.get("json", False)

        # Handle positional argument for row number
        if row_num is None and "args" in kwargs:
            args_val = kwargs["args"]
            if isinstance(args_val, list) and args_val:
                try:
                    row_num = int(args_val[0])
                except (ValueError, TypeError):
                    pass
            elif isinstance(args_val, str):
                try:
                    row_num = int(args_val)
                except (ValueError, TypeError):
                    pass

        # Apply limit if specified
        display_log = tool_log
        if limit and limit > 0:
            display_log = tool_log[-limit:]

        # Handle JSON output
        if show_json:
            json_data = [
                {
                    "tool": entry.tool_name,
                    "arguments": entry.arguments,
                    "outcome": entry.outcome.value,
                    "result_summary": entry.result_summary,
                    "timestamp": entry.timestamp.isoformat(),
                }
                for entry in display_log
            ]
            json_output = json.dumps(json_data, indent=2, default=str)
            return CommandResult(
                success=True,
                output=json_output,
            )

        # Handle row detail view
        if row_num is not None:
            if 1 <= row_num <= len(tool_log):
                entry = tool_log[row_num - 1]
                output.panel(
                    f"Tool: {entry.tool_name}\n"
                    f"Outcome: {entry.outcome.value}\n"
                    f"Arguments:\n{json.dumps(entry.arguments, indent=2)}\n"
                    f"Result: {entry.result_summary}",
                    title=f"Tool Call #{row_num}",
                    style="cyan",
                )
                return CommandResult(success=True)
            else:
                return CommandResult(
                    success=False,
                    error=f"Invalid row number: {row_num}. Valid range: 1-{len(tool_log)}",
                )

        # Default table view
        table_data = []
        for i, entry in enumerate(display_log, 1):
            # Truncate arguments for display
            args_str = json.dumps(entry.arguments)
            if len(args_str) > 50:
                args_str = args_str[:47] + "..."

            table_data.append(
                {
                    "#": str(i),
                    "Tool": entry.tool_name,
                    "Arguments": args_str,
                    "Status": entry.format_compact().split("]")[0] + "]",
                }
            )

        table = format_table(
            table_data,
            title="Tool Call History",
            columns=["#", "Tool", "Arguments", "Status"],
        )
        output.print_table(table)

        if limit and limit < len(tool_log):
            output.hint(f"Showing last {limit} of {len(tool_log)} total calls")

        return CommandResult(success=True)
