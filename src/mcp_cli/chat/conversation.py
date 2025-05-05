# mcp_cli/chat/conversation.py
import time
import asyncio
from rich import print

# mcp cli imports
from mcp_cli.chat.tool_processor import ToolProcessor


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

                    # Skip slash commands (already handled by UI)
                    last_msg = (
                        self.context.conversation_history[-1]
                        if self.context.conversation_history
                        else {}
                    )
                    content = last_msg.get("content", "")
                    if last_msg.get("role") == "user" and content.startswith("/"):
                        return

                    # Ensure OpenAI tools are loaded for function calling
                    if not getattr(self.context, "openai_tools", None):
                        self.context.openai_tools = (
                            await self._maybe_async_get_tools()
                        )
                        print(f"[DEBUG] Loaded OpenAI tools: {len(self.context.openai_tools)} tools")

                    # Attempt LLM call with function-calling
                    try:
                        completion = await self.context.client.create_completion(
                            messages=self.context.conversation_history,
                            tools=self.context.openai_tools,
                        )
                    except Exception as e:
                        # If tools spec invalid, retry without tools
                        err = str(e)
                        if "Invalid 'tools[0].function.name'" in err:
                            print("[yellow]Warning: tool definitions rejected by model, retrying without tools...[/yellow]")
                            completion = await self.context.client.create_completion(
                                messages=self.context.conversation_history
                            )
                        else:
                            raise

                    response_content = completion.get("response", "No response")
                    tool_calls = completion.get("tool_calls", [])

                    # If model requested tool calls, execute them
                    if tool_calls:
                        await self.tool_processor.process_tool_calls(tool_calls)
                        continue

                    # Otherwise, display the assistant's reply
                    elapsed = time.time() - start_time
                    self.ui_manager.print_assistant_response(response_content, elapsed)
                    self.context.conversation_history.append(
                        {"role": "assistant", "content": response_content}
                    )
                    break

                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    print(f"[red]Error during conversation processing:[/red] {exc}")
                    import traceback; traceback.print_exc()
                    self.context.conversation_history.append(
                        {"role": "assistant", "content": f"I encountered an error: {exc}"}
                    )
                    break
        except asyncio.CancelledError:
            raise

    async def _maybe_async_get_tools(self):
        """Helper to fetch tools, awaiting if necessary."""
        tools = self.context.tool_manager.get_tools_for_llm()
        if asyncio.iscoroutine(tools):
            return await tools
        return tools
