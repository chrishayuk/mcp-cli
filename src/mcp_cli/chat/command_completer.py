# mcp_cli/chat/command_completer.py
from prompt_toolkit.completion import Completer, Completion


class ChatCommandCompleter(Completer):
    """Completer for chat/interactive slash-commands using unified command system."""

    def __init__(self, context):
        self.context = context

    def get_completions(self, document, complete_event):
        """Get completions from unified command registry."""
        from mcp_cli.commands.registry import UnifiedCommandRegistry
        from mcp_cli.commands.base import CommandMode
        from mcp_cli.commands import register_all_commands

        # Ensure commands are registered
        register_all_commands()

        # Get text before cursor
        text = document.text_before_cursor.lstrip()

        # Only complete if we're typing a command (starts with /)
        if not text or not text.startswith("/"):
            return

        # Get unified commands
        registry = UnifiedCommandRegistry()
        commands = registry.list_commands(mode=CommandMode.CHAT)

        # Calculate start position - how far back to replace
        start_pos = -len(text)

        # Show all matching commands
        for cmd in commands:
            cmd_text = f"/{cmd.name}"
            # Check if this command matches what's been typed
            if cmd_text.startswith(text):
                yield Completion(
                    cmd_text,
                    start_position=start_pos,
                    display=cmd_text,
                    display_meta=cmd.description,
                )

            # Also check aliases
            for alias in cmd.aliases:
                alias_text = f"/{alias}"
                if alias_text.startswith(text) and alias != cmd.name:
                    yield Completion(
                        alias_text,
                        start_position=start_pos,
                        display=alias_text,
                        display_meta=f"→ /{cmd.name}",
                    )
