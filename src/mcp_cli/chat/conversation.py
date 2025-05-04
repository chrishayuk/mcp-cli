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

                    # ── 1.  Quick NL-to-tool shortcut ────────────────────
                    if self.context.conversation_history:
                        last_message = self.context.conversation_history[-1]
                        if last_message.get("role") == "user":
                            content = last_message.get("content", "")
                            tool_name = self.context.tool_manager.parse_natural_language_tool(content)
                            if tool_name:
                                print(f"[DEBUG] Detected direct tool command: {content} -> {tool_name}")
                                fake_tool_call = {
                                    "function": {"name": tool_name, "arguments": {}}
                                }
                                await self.tool_processor.process_tool_calls([fake_tool_call])
                                return

                    # ── 2.  Refresh tools for the LLM if needed ──────────
                    if not getattr(self.context, "openai_tools", None):
                        self.context.openai_tools = self.context.tool_manager.get_tools_for_llm()
                        print(f"[DEBUG] Refreshed OpenAI tools: {len(self.context.openai_tools)} tools")

                    # ── 3.  Call the LLM (ASYNC) ─────────────────────────
                    completion = await self.context.client.create_completion(   # ▲ await
                        messages=self.context.conversation_history,
                        tools=self.context.openai_tools,
                    )

                    response_content = completion.get("response", "No response")
                    tool_calls = completion.get("tool_calls", [])

                    # ── 4.  Handle tool-calls if requested ───────────────
                    if tool_calls:
                        await self.tool_processor.process_tool_calls(tool_calls)
                        continue  # loop again after tools execute

                    # ── 5.  Normal assistant reply ───────────────────────
                    response_time = time.time() - start_time
                    self.ui_manager.print_assistant_response(response_content, response_time)
                    self.context.conversation_history.append(
                        {"role": "assistant", "content": response_content}
                    )
                    break

                except asyncio.CancelledError:
                    raise  # bubble up
                except Exception as e:
                    print(f"[red]Error during conversation processing:[/red] {e}")
                    import traceback
                    traceback.print_exc()
                    self.context.conversation_history.append(
                        {"role": "assistant", "content": f"I encountered an error: {str(e)}"}
                    )
                    break
        except asyncio.CancelledError:
            raise
