# src/cli/chat/command_completer.py
from prompt_toolkit.completion import Completer, Completion
from mcp_cli.chat.commands import get_command_completions

class ChatCommandCompleter(Completer):
    """Completer for chat commands with slash prefix."""
    
    def __init__(self, context):
        self.context = context
        
    def get_completions(self, document, complete_event):
        text = document.text
        
        # Only suggest completions for slash commands
        if text.lstrip().startswith('/'):
            word_before_cursor = document.get_word_before_cursor()
            
            # Get completions from command system
            completions = get_command_completions(text.lstrip())
            
            for completion in completions:
                # If completion already matches what's there, don't suggest it
                if text.lstrip() == completion:
                    continue
                    
                # For simple command completion, just return the command
                if ' ' not in completion:
                    yield Completion(
                        completion, 
                        start_position=-len(text.lstrip()),
                        style='fg:goldenrod'  # Darker gold/yellow color
                    )
                # For argument completion, provide the full arg
                else:
                    yield Completion(
                        completion.split()[-1], 
                        start_position=-len(word_before_cursor),
                        style='fg:goldenrod'
                    )