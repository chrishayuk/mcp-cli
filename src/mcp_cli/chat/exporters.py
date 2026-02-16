# mcp_cli/chat/exporters.py
"""Conversation export formatters.

Supports Markdown and JSON export with full metadata and tool call details.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


class MarkdownExporter:
    """Export conversation as formatted Markdown."""

    @staticmethod
    def export(
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Export messages to Markdown format.

        Args:
            messages: Conversation messages (list of dicts with role, content, etc.)
            metadata: Optional session metadata (provider, model, session_id, etc.)

        Returns:
            Formatted Markdown string
        """
        lines: list[str] = []
        lines.append("# Chat Export")
        lines.append("")

        # Metadata section
        if metadata:
            lines.append("## Metadata")
            lines.append("")
            for key, value in metadata.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Messages
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "system":
                lines.append("### System")
                lines.append("")
                lines.append(f"> {content}")
                lines.append("")
            elif role == "user":
                lines.append("### User")
                lines.append("")
                lines.append(content)
                lines.append("")
            elif role == "assistant":
                lines.append("### Assistant")
                lines.append("")
                if content:
                    lines.append(content)
                    lines.append("")

                # Tool calls
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        name = func.get("name", "unknown")
                        args = func.get("arguments", "{}")
                        lines.append(f"**Tool Call**: `{name}`")
                        lines.append("")
                        lines.append("```json")
                        # Pretty-print arguments
                        try:
                            parsed = json.loads(args) if isinstance(args, str) else args
                            lines.append(json.dumps(parsed, indent=2))
                        except (json.JSONDecodeError, TypeError):
                            lines.append(str(args))
                        lines.append("```")
                        lines.append("")
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                lines.append(f"### Tool Result (`{tool_call_id}`)")
                lines.append("")
                lines.append("```")
                lines.append(str(content)[:2000])  # Truncate long results
                lines.append("```")
                lines.append("")

        return "\n".join(lines)


class JSONExporter:
    """Export conversation as structured JSON."""

    @staticmethod
    def export(
        messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        token_usage: dict[str, Any] | None = None,
    ) -> str:
        """Export messages to JSON format.

        Args:
            messages: Conversation messages
            metadata: Optional session metadata
            token_usage: Optional token usage data

        Returns:
            JSON string
        """
        export_data: dict[str, Any] = {
            "version": "1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

        if metadata:
            export_data["metadata"] = metadata

        if token_usage:
            export_data["token_usage"] = token_usage

        export_data["messages"] = messages

        return json.dumps(export_data, indent=2, default=str)
