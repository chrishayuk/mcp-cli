# tests/chat/test_init.py
"""Tests for chat/__init__.py lazy import."""

import pytest


class TestChatModuleLazyImport:
    """Tests for lazy import in chat module."""

    def test_getattr_handle_chat_mode(self):
        """Test lazy import of handle_chat_mode."""
        from mcp_cli import chat

        # Access the lazy-loaded attribute
        func = chat.handle_chat_mode
        assert callable(func)

    def test_getattr_invalid_attribute(self):
        """Test AttributeError for invalid attribute."""
        from mcp_cli import chat

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = chat.nonexistent_attribute

    def test_all_exports(self):
        """Test __all__ exports."""
        from mcp_cli import chat

        assert "handle_chat_mode" in chat.__all__
