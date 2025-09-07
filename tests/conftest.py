"""Common test fixtures and utilities for MCP CLI tests."""

import pytest
from mcp_cli.context import ContextManager, initialize_context


@pytest.fixture(autouse=True)
def reset_context():
    """Reset the context manager before each test."""
    ContextManager().reset()
    yield
    # Clean up after test
    ContextManager().reset()


def setup_test_context(tool_manager=None, **kwargs):
    """
    Helper to set up test context with common defaults.

    Args:
        tool_manager: Optional tool manager to use
        **kwargs: Additional context parameters

    Returns:
        The initialized ApplicationContext
    """
    # Reset first to ensure clean state
    ContextManager().reset()

    # Set defaults if not provided
    if "provider" not in kwargs:
        kwargs["provider"] = "openai"
    if "model" not in kwargs:
        kwargs["model"] = "gpt-4"

    # Initialize with test defaults
    context = initialize_context(tool_manager=tool_manager, **kwargs)

    return context
