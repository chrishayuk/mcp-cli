# mcp_cli/chat/command_completer.py
from prompt_toolkit.completion import Completer, Completion


class ChatCommandCompleter(Completer):
    """Completer for chat/interactive slash-commands."""

    def __init__(self, context):
        self.context = context

    # ↓↓↓  import happens *after* all modules are fully initialised
    def get_completions(self, document, complete_event):
        from mcp_cli.chat.commands import get_command_completions  # ← moved

        txt = document.text.lstrip()
        if not txt.startswith("/"):
            return

        word_before = document.get_word_before_cursor()
        for cand in get_command_completions(txt):
            if txt == cand:
                continue  # already typed
            if " " not in cand:  # plain command
                yield Completion(
                    cand,
                    start_position=-len(txt),
                    style="fg:goldenrod",
                )
            else:  # completing an argument
                yield Completion(
                    cand.split()[-1],
                    start_position=-len(word_before),
                    style="fg:goldenrod",
                )
