# tests/chat/test_session_store.py
"""Tests for session persistence."""

import pytest
from mcp_cli.chat.session_store import SessionStore, SessionData, SessionMetadata


@pytest.fixture
def store(tmp_path):
    """Session store with temporary directory."""
    return SessionStore(sessions_dir=tmp_path)


@pytest.fixture
def sample_data():
    """Sample session data for testing."""
    return SessionData(
        metadata=SessionMetadata(
            session_id="test-abc123",
            provider="openai",
            model="gpt-4",
        ),
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
    )


class TestSessionStore:
    def test_save_creates_file(self, store, sample_data):
        path = store.save(sample_data)
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_updates_metadata(self, store, sample_data):
        store.save(sample_data)
        loaded = store.load("test-abc123")
        assert loaded is not None
        assert loaded.metadata.message_count == 3

    def test_load_returns_data(self, store, sample_data):
        store.save(sample_data)
        loaded = store.load("test-abc123")
        assert loaded is not None
        assert loaded.metadata.session_id == "test-abc123"
        assert loaded.metadata.provider == "openai"
        assert loaded.metadata.model == "gpt-4"
        assert len(loaded.messages) == 3

    def test_load_missing_returns_none(self, store):
        result = store.load("nonexistent")
        assert result is None

    def test_list_sessions_empty(self, store):
        sessions = store.list_sessions()
        assert sessions == []

    def test_list_sessions(self, store, sample_data):
        store.save(sample_data)

        # Save another session
        data2 = SessionData(
            metadata=SessionMetadata(
                session_id="test-def456",
                provider="anthropic",
                model="claude-3",
            ),
            messages=[{"role": "user", "content": "Test"}],
        )
        store.save(data2)

        sessions = store.list_sessions()
        assert len(sessions) == 2
        ids = {s.session_id for s in sessions}
        assert "test-abc123" in ids
        assert "test-def456" in ids

    def test_delete_existing(self, store, sample_data):
        store.save(sample_data)
        assert store.delete("test-abc123") is True
        assert store.load("test-abc123") is None

    def test_delete_missing(self, store):
        assert store.delete("nonexistent") is False

    def test_save_with_token_usage(self, store):
        data = SessionData(
            metadata=SessionMetadata(
                session_id="test-tokens",
                provider="openai",
                model="gpt-4",
            ),
            messages=[{"role": "user", "content": "Hello"}],
            token_usage={"total_input": 100, "total_output": 50},
        )
        store.save(data)
        loaded = store.load("test-tokens")
        assert loaded is not None
        assert loaded.token_usage is not None
        assert loaded.token_usage["total_input"] == 100

    def test_path_traversal_prevention(self, store):
        """Ensure session IDs can't escape the sessions directory."""
        data = SessionData(
            metadata=SessionMetadata(
                session_id="../../../etc/passwd",
                provider="openai",
                model="gpt-4",
            ),
            messages=[],
        )
        path = store.save(data)
        # Path should be within sessions_dir
        assert str(store.sessions_dir) in str(path)


class TestAutoSave:
    def test_auto_save_triggers(self):
        """Auto-save fires after N turns."""
        from unittest.mock import MagicMock, patch

        from mcp_cli.chat.chat_context import ChatContext

        tool_manager = MagicMock()
        model_manager = MagicMock()
        model_manager.provider = "openai"
        model_manager.model = "gpt-4"
        model_manager.api_base = None
        model_manager.api_key = None

        ctx = ChatContext(tool_manager=tool_manager, model_manager=model_manager)

        with patch.object(ctx, "save_session") as mock_save:
            # Simulate turns below threshold
            for _ in range(9):
                ctx.auto_save_check()
            mock_save.assert_not_called()

            # 10th call should trigger auto-save
            ctx.auto_save_check()
            mock_save.assert_called_once()
