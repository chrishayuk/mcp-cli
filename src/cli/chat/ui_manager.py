# src/cli/chat/ui_manager.py
import os
import json
from rich import print
from rich.markdown import Markdown
from rich.panel import Panel
from rich.console import Console

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style

from cli.chat.command_completer import ChatCommandCompleter
from cli.chat.commands import handle_command

class ChatUIManager:
    """Class to manage the chat UI interface."""
    
    def __init__(self, context):
        self.context = context
        self.console = Console()
        
        # Set up prompt_toolkit session with history and tab completion
        history_file = os.path.expanduser("~/.mcp_chat_history")
        self.style = Style.from_dict({
            # Don't highlight the completion menu background
            'completion-menu': 'bg:default',
            'completion-menu.completion': 'bg:default fg:goldenrod',
            'completion-menu.completion.current': 'bg:default fg:goldenrod bold',
            # Set auto-suggestion color to a very subtle shade
            'auto-suggestion': 'fg:ansibrightblack',
        })
        
        self.session = PromptSession(
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory(),
            completer=ChatCommandCompleter(context.to_dict()),
            complete_while_typing=True,
            style=self.style
        )
    
    async def get_user_input(self):
        """Get user input with prompt toolkit."""
        # Print the prompt with Rich to ensure consistent color
        print("[bold yellow]>[/bold yellow] ", end="", flush=True)
        
        # Use prompt_toolkit for enhanced input
        user_message = await self.session.prompt_async("")
        return user_message.strip()
    
    def print_user_message(self, message):
        """Print formatted user message."""
        user_panel_text = message if message else "[No Message]"
        print(Panel(user_panel_text, style="bold yellow", title="You"))
    
    def print_tool_call(self, tool_name, raw_arguments):
        """Print formatted tool call."""
        # Handle JSON arguments
        if isinstance(raw_arguments, str):
            try:
                raw_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                # If it's not valid JSON, just display as is
                pass

        # Format as JSON
        tool_args_str = json.dumps(raw_arguments, indent=2)
        tool_md = f"**Tool Call:** {tool_name}\n\n```json\n{tool_args_str}\n```"
        print(Panel(Markdown(tool_md), style="bold magenta", title="Tool Invocation"))
    
    def print_assistant_response(self, response_content, response_time):
        """Print formatted assistant response."""
        assistant_panel_text = response_content if response_content else "[No Response]"
        footer = f"Response time: {response_time:.2f}s"
        print(
            Panel(
                Markdown(assistant_panel_text), 
                style="bold blue", 
                title="Assistant",
                subtitle=footer
            )
        )
        
    async def handle_command(self, command):
        """Handle a command and update context if needed."""
        # Convert context to dict for command handler
        context_dict = self.context.to_dict()
        
        # Pass to command handler
        handled = await handle_command(command, context_dict)
        
        # Update context with any changes made by commands
        self.context.update_from_dict(context_dict)
        
        return handled