# tests/interactive/test_interactive_shell.py
import sys
import types
import pytest

# ─── Stub out mcp_cli.interactive.commands so shell can import ────────────
dummy_commands = types.ModuleType("mcp_cli.interactive.commands")
# Provide the needed register_all_commands function
dummy_commands.register_all_commands = lambda: None
sys.modules["mcp_cli.interactive.commands"] = dummy_commands

# Also stub out the base so there are no missing imports
dummy_base = types.ModuleType("mcp_cli.interactive.commands.base")
# Minimal InteractiveCommand for type consistency (not actually used here)
class InteractiveCommand:
    def __init__(self, name, help_text="", aliases=None):
        self.name = name
        self.aliases = aliases or []
dummy_base.InteractiveCommand = InteractiveCommand
sys.modules["mcp_cli.interactive.commands.base"] = dummy_base

# ─── Now import the function under test ───────────────────────────────────
from mcp_cli.interactive.shell import interactive_mode


def test_interactive_mode_is_callable():
    # Simply verify that interactive_mode was imported
    assert callable(interactive_mode)
