# mcp_cli/chat/session_store.py
"""Session persistence — save and restore conversation sessions.

Pydantic-native. Sessions are stored as JSON files in a configurable directory.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SessionMetadata(BaseModel):
    """Metadata for a saved session."""

    session_id: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    provider: str = ""
    model: str = ""
    message_count: int = 0
    description: str = ""


class SessionData(BaseModel):
    """Complete session data for persistence."""

    metadata: SessionMetadata
    messages: list[dict[str, Any]] = Field(default_factory=list)
    token_usage: dict[str, Any] | None = None


class SessionStore:
    """File-based session persistence.

    Stores sessions as JSON files in a configurable directory.
    """

    def __init__(self, sessions_dir: Path | None = None):
        if sessions_dir is None:
            sessions_dir = Path.home() / ".mcp-cli" / "sessions"
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """Get the file path for a session."""
        # Sanitize session_id to prevent path traversal
        safe_id = session_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self.sessions_dir / f"{safe_id}.json"

    def save(self, data: SessionData) -> Path:
        """Save session data to disk.

        Args:
            data: Session data to save

        Returns:
            Path to the saved file
        """
        data.metadata.updated_at = datetime.now(timezone.utc).isoformat()
        data.metadata.message_count = len(data.messages)

        path = self._session_path(data.metadata.session_id)
        path.write_text(data.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Session saved: {path}")
        return path

    def load(self, session_id: str) -> SessionData | None:
        """Load session data from disk.

        Args:
            session_id: Session ID to load

        Returns:
            SessionData or None if not found
        """
        path = self._session_path(session_id)
        if not path.exists():
            logger.warning(f"Session not found: {session_id}")
            return None

        try:
            raw = path.read_text(encoding="utf-8")
            data: SessionData = SessionData.model_validate_json(raw)
            return data
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None

    def list_sessions(self) -> list[SessionMetadata]:
        """List all saved sessions.

        Returns:
            List of session metadata, sorted by updated_at (newest first)
        """
        sessions: list[SessionMetadata] = []
        for path in self.sessions_dir.glob("*.json"):
            try:
                raw = path.read_text(encoding="utf-8")
                data = SessionData.model_validate_json(raw)
                sessions.append(data.metadata)
            except Exception as e:
                logger.warning(f"Skipping corrupt session file {path}: {e}")

        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def delete(self, session_id: str) -> bool:
        """Delete a saved session.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted, False if not found
        """
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            logger.info(f"Session deleted: {session_id}")
            return True
        return False
