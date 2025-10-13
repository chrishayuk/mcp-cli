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

        txt = document.text.lstrip()
        if not txt.startswith("/"):
            return

        # Get unified commands
        registry = UnifiedCommandRegistry()
        commands = registry.list_commands(mode=CommandMode.CHAT)

        # Generate completions
        for cmd in commands:
            # Check if this command matches the partial text
            if f"/{cmd.name}".startswith(txt):
                # Calculate the replacement text (only the part not yet typed)
                replacement = f"/{cmd.name}"[len(txt) :]

                yield Completion(
                    replacement,
                    start_position=0,
                    display=f"/{cmd.name}",
                    display_meta=cmd.description[:40]
                    if len(cmd.description) > 40
                    else cmd.description,
                )

            # Also check aliases
            for alias in cmd.aliases:
                if f"/{alias}".startswith(txt) and alias != cmd.name:
                    replacement = f"/{alias}"[len(txt) :]
                    yield Completion(
                        replacement,
                        start_position=0,
                        display=f"/{alias}",
                        display_meta=f"→ /{cmd.name}",
                    )
