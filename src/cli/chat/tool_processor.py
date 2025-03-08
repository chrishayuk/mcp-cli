# src/cli/chat/tool_processor.py
from rich.console import Console
from mcpcli.tools_handler import handle_tool_call

class ToolProcessor:
    """Class to handle tool processing."""
    
    def __init__(self, context, ui_manager):
        self.context = context
        self.ui_manager = ui_manager
    
    async def process_tool_calls(self, tool_calls):
        """Process a list of tool calls."""
        for tool_call in tool_calls:
            # Extract tool_name and raw_arguments
            if hasattr(tool_call, "function"):
                tool_name = getattr(tool_call.function, "name", "unknown tool")
                raw_arguments = getattr(tool_call.function, "arguments", {})
            elif isinstance(tool_call, dict) and "function" in tool_call:
                fn_info = tool_call["function"]
                tool_name = fn_info.get("name", "unknown tool")
                raw_arguments = fn_info.get("arguments", {})
            else:
                tool_name = "unknown tool"
                raw_arguments = {}

            # Display the tool call
            self.ui_manager.print_tool_call(tool_name, raw_arguments)

            # Process tool call
            with Console().status("[cyan]Executing tool...[/cyan]", spinner="dots"):
                await handle_tool_call(
                    tool_call, 
                    self.context.conversation_history, 
                    self.context.server_streams
                )