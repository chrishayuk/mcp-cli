# mcp_cli/chat/tool_processor.py
from rich.console import Console
from rich import print
import json

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
            
        if not hasattr(self.context, 'stream_manager') or not self.context.stream_manager:
            print("[red]Error: No StreamManager available for tool calls.[/red]")
            # Add a failed tool response to the conversation history
            self.context.conversation_history.append({
                "role": "tool",
                "name": "system",
                "content": "Error: No StreamManager available to process tool calls."
            })
            return
            
        for tool_call in tool_calls:
            try:
                # Extract tool_name and raw_arguments
                if hasattr(tool_call, "function"):
                    tool_name = getattr(tool_call.function, "name", "unknown tool")
                    raw_arguments = getattr(tool_call.function, "arguments", {})
                    tool_call_id = getattr(tool_call, "id", f"call_{tool_name}")
                elif isinstance(tool_call, dict) and "function" in tool_call:
                    fn_info = tool_call["function"]
                    tool_name = fn_info.get("name", "unknown tool")
                    raw_arguments = fn_info.get("arguments", {})
                    tool_call_id = tool_call.get("id", f"call_{tool_name}")
                else:
                    tool_name = "unknown tool"
                    raw_arguments = {}
                    tool_call_id = f"call_{tool_name}"

                # Display the tool call
                self.ui_manager.print_tool_call(tool_name, raw_arguments)

                # Process tool call using StreamManager
                with Console().status("[cyan]Executing tool...[/cyan]", spinner="dots"):
                    try:
                        # Parse arguments if they're a string
                        if isinstance(raw_arguments, str):
                            try:
                                arguments = json.loads(raw_arguments)
                            except json.JSONDecodeError:
                                arguments = raw_arguments
                        else:
                            arguments = raw_arguments
                            
                        # Call the tool using StreamManager
                        result = await self.context.stream_manager.call_tool(
                            tool_name=tool_name,
                            arguments=arguments
                        )
                        
                        # Add the tool call to conversation history
                        self.context.conversation_history.append({
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": json.dumps(arguments) if isinstance(arguments, dict) else str(arguments)
                                    }
                                }
                            ]
                        })
                        
                        # Extract content from result
                        if isinstance(result, dict):
                            if result.get("isError"):
                                content = f"Error: {result.get('error', 'Unknown error')}"
                            else:
                                content = result.get("content", "No content returned")
                                if isinstance(content, (list, dict)):
                                    # Format structured content as JSON string
                                    content = json.dumps(content, indent=2)
                        else:
                            content = str(result)
                            
                        # Add the tool response to conversation history
                        self.context.conversation_history.append({
                            "role": "tool",
                            "name": tool_name,
                            "content": content,
                            "tool_call_id": tool_call_id
                        })
                        
                    except Exception as e:
                        print(f"[red]Error executing tool {tool_name}: {e}[/red]")
                        
                        # Add a failed tool response to maintain conversation flow
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
                                        "arguments": json.dumps(raw_arguments) if isinstance(raw_arguments, dict) else str(raw_arguments)
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