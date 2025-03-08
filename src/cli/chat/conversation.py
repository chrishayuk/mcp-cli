# src/cli/chat/conversation.py
import time
import asyncio
from rich import print

from cli.chat.tool_processor import ToolProcessor

class ConversationProcessor:
    """Class to handle LLM conversation processing."""
    
    def __init__(self, context, ui_manager):
        self.context = context
        self.ui_manager = ui_manager
        self.tool_processor = ToolProcessor(context, ui_manager)
    
    async def process_conversation(self):
        """Process the conversation loop, handling tool calls and responses."""
        try:
            while True:
                try:
                    start_time = time.time()
                    
                    completion = self.context.client.create_completion(
                        messages=self.context.conversation_history,
                        tools=self.context.openai_tools,
                    )

                    response_content = completion.get("response", "No response")
                    tool_calls = completion.get("tool_calls", [])
                    
                    # Calculate response time
                    response_time = time.time() - start_time

                    # Process tool calls if any
                    if tool_calls:
                        await self.tool_processor.process_tool_calls(tool_calls)
                        continue

                    # Display assistant response
                    self.ui_manager.print_assistant_response(response_content, response_time)
                    self.context.conversation_history.append(
                        {"role": "assistant", "content": response_content}
                    )
                    break
                except asyncio.CancelledError:
                    # Handle cancellation during API calls
                    raise
                except Exception as e:
                    print(f"[red]Error during conversation processing:[/red] {e}")
                    self.context.conversation_history.append(
                        {"role": "assistant", "content": f"I encountered an error: {str(e)}"}
                    )
                    break
        except asyncio.CancelledError:
            # Propagate cancellation up
            raise