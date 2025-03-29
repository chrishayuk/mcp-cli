from rich.console import Console
from rich import print

# mcp_cli
from mcp_cli.llm.tools_handler import handle_tool_call

class ToolProcessor:
    """Class to handle tool processing."""
    
    def __init__(self, context, ui_manager):
        self.context = context
        self.ui_manager = ui_manager
    
    async def process_tool_calls(self, tool_calls):
        """Process a list of tool calls."""
        if not tool_calls:
            print("[yellow]Warning: Empty tool_calls list received.[/yellow]")
            return
            
        if not self.context.server_streams:
            print("[red]Error: No server streams available for tool calls.[/red]")
            # Add a failed tool response to the conversation history
            self.context.conversation_history.append({
                "role": "tool",
                "name": "system",
                "content": "Error: No server connections available to process tool calls."
            })
            return
            
        for tool_call in tool_calls:
            try:
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
                    try:
                        await handle_tool_call(
                            tool_call, 
                            self.context.conversation_history, 
                            self.context.server_streams
                        )
                    except Exception as e:
                        print(f"[red]Error executing tool {tool_name}: {e}[/red]")
                        
                        # Add a failed tool response to maintain conversation flow
                        tool_call_id = getattr(tool_call, "id", "unknown_id")
                        if isinstance(tool_call, dict) and "id" in tool_call:
                            tool_call_id = tool_call["id"]
                            
                        # Add a placeholder tool call to history
                        self.context.conversation_history.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": str(raw_arguments)
                                    }
                                }
                            ]
                        })
                        
                        # Add error response
                        self.context.conversation_history.append({
                            "role": "tool",
                            "name": tool_name,
                            "content": f"Error: Could not execute tool. {str(e)}",
                            "tool_call_id": tool_call_id
                        })
                        
            except Exception as e:
                print(f"[red]Error processing tool call: {e}[/red]")