"""
Unit tests for the prompt-toolkit completer used in chat / interactive mode.

The only thing we really care about here is that **no circular-import** or
other initialisation error is triggered when completions are requested.
"""

from __future__ import annotations

import pytest
from prompt_toolkit.document import Document

from mcp_cli.chat.command_completer import ChatCommandCompleter


@pytest.fixture()
def completer():
    """A completer instance with a minimal dummy context."""
    return ChatCommandCompleter(context={})


@pytest.mark.parametrize(
    "text, expect_some",
    [
        ("/to", True),          # should suggest /tools, /tools-all, …
        ("/xyz", False),        # no command starts with /xyz
        ("plain text", False),  # not a slash-command → no completions
    ],
)
def test_get_completions(completer, text: str, expect_some: bool):
    """
    • must **not** raise  
    • returns an iterator; convert to list so we can inspect the suggestions
    """
    comps = list(completer.get_completions(Document(text=text), None))
    if expect_some:
        assert comps, f"expected suggestions for {text!r}"
    else:
        assert not comps, f"unexpected suggestions for {text!r}"
