"""Playbook service for querying and managing playbooks via MCP."""

from __future__ import annotations

import logging
from typing import Any

from mcp_cli.utils.preferences import get_preference_manager

logger = logging.getLogger(__name__)


class PlaybookService:
    """Service for interacting with the playbook MCP server."""

    def __init__(self, tool_manager: Any = None):
        """Initialize the playbook service.

        Args:
            tool_manager: Optional ToolManager instance for making tool calls
        """
        self.tool_manager = tool_manager
        self.pref_manager = get_preference_manager()

    def is_enabled(self) -> bool:
        """Check if playbook integration is enabled."""
        return self.pref_manager.is_playbook_enabled()

    def get_server_name(self) -> str:
        """Get the configured playbook server name."""
        return self.pref_manager.get_playbook_server_name()

    async def query_playbook(
        self, question: str, top_k: int | None = None
    ) -> str | None:
        """Query the playbook repository.

        Args:
            question: Natural language question
            top_k: Number of results to consider (default from preferences)

        Returns:
            Markdown content of the most relevant playbook, or None if not found/disabled
        """
        if not self.is_enabled():
            logger.debug("Playbook service is disabled")
            return None

        if not self.tool_manager:
            logger.warning("No tool manager available for playbook queries")
            return None

        if top_k is None:
            top_k = self.pref_manager.get_playbook_top_k()

        server_name = self.get_server_name()

        try:
            # Call the query_playbook tool on the playbook server
            result = await self.tool_manager.call_tool(
                tool_name="query_playbook",
                arguments={"question": question, "top_k": top_k},
                server_name=server_name,
            )

            if isinstance(result, dict):
                # Extract content from result
                return result.get("content") or result.get("result")
            elif isinstance(result, str):
                return result
            else:
                logger.warning(
                    f"Unexpected result type from query_playbook: {type(result)}"
                )
                return None

        except Exception as e:
            logger.error(f"Error querying playbook: {e}")
            return None

    async def list_playbooks(self) -> list[str]:
        """List all available playbooks.

        Returns:
            List of playbook titles
        """
        if not self.is_enabled() or not self.tool_manager:
            return []

        server_name = self.get_server_name()

        try:
            result = await self.tool_manager.call_tool(
                tool_name="list_playbooks",
                arguments={},
                server_name=server_name,
            )

            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                playbooks: list[str] = result.get("playbooks", [])
                return playbooks
            else:
                return []

        except Exception as e:
            logger.error(f"Error listing playbooks: {e}")
            return []

    async def get_playbook(self, title: str) -> str | None:
        """Get a specific playbook by title.

        Args:
            title: Exact playbook title

        Returns:
            Playbook markdown content or None if not found
        """
        if not self.is_enabled() or not self.tool_manager:
            return None

        server_name = self.get_server_name()

        try:
            result = await self.tool_manager.call_tool(
                tool_name="get_playbook",
                arguments={"title": title},
                server_name=server_name,
            )

            if isinstance(result, str):
                return result
            elif isinstance(result, dict):
                return result.get("content") or result.get("result")
            else:
                return None

        except Exception as e:
            logger.error(f"Error getting playbook '{title}': {e}")
            return None

    async def submit_playbook(
        self,
        title: str,
        content: str,
        description: str,
        tags: list[str] | None = None,
    ) -> bool:
        """Submit a new playbook to the repository.

        This is for user-submitted playbooks (different from ingest which is for
        loading playbooks from files/directories).

        Args:
            title: Playbook title
            content: Markdown content
            description: Brief description
            tags: Optional tags for categorization

        Returns:
            True if successful, False otherwise
        """
        if not self.is_enabled() or not self.tool_manager:
            return False

        server_name = self.get_server_name()

        try:
            await self.tool_manager.call_tool(
                tool_name="ingest_playbook",
                arguments={
                    "title": title,
                    "content": content,
                    "description": description,
                    "tags": tags or [],
                    "author": "mcp-cli-user",
                },
                server_name=server_name,
            )
            return True

        except Exception as e:
            logger.error(f"Error submitting playbook: {e}")
            return False
