# mcp_cli/chat/conversation.py
import time
import asyncio
import logging
from rich import print

# mcp cli imports
from mcp_cli.chat.tool_processor import ToolProcessor

log = logging.getLogger(__name__)

class ConversationProcessor:
    """Class to handle LLM conversation processing."""

    def __init__(self, context, ui_manager):
        self.context = context
        self.ui_manager = ui_manager
        self.tool_processor = ToolProcessor(context, ui_manager)

    # mcp_cli/chat/conversation.py - update the process_conversation method
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
                        try:
                            # Get adapted tools that will work with OpenAI
                            if hasattr(self.context.tool_manager, "get_adapted_tools_for_llm"):
                                # EXPLICITLY specify OpenAI provider for proper adaptation
                                tools_and_mapping = await self.context.tool_manager.get_adapted_tools_for_llm("openai")
                                self.context.openai_tools = tools_and_mapping[0]
                                self.context.tool_name_mapping = tools_and_mapping[1]
                                log.debug(f"Loaded {len(self.context.openai_tools)} adapted tools for OpenAI")
                                
                                # Validate all tool names
                                import re
                                has_invalid = False
                                for i, tool in enumerate(self.context.openai_tools):
                                    name = tool["function"]["name"]
                                    is_valid = re.match(r'^[a-zA-Z0-9_-]+$', name) is not None
                                    print(f"[DEBUG] Tool {i}: '{name}' valid = {is_valid}")
                                    if not is_valid:
                                        has_invalid = True
                                
                                if has_invalid:
                                    print("[CRITICAL] Found invalid tool names that will cause OpenAI API errors!")
                                else:
                                    print("[DEBUG] All tool names are valid for OpenAI")
                        except Exception as exc:
                            log.error(f"Error loading tools: {exc}")
                            self.context.openai_tools = []
                            self.context.tool_name_mapping = {}
                    
                    # Sanitize conversation history before making API call
                    self._sanitize_conversation_history()

                    # Attempt LLM call with function-calling
                    try:
                        completion = await self.context.client.create_completion(
                            messages=self.context.conversation_history,
                            tools=self.context.openai_tools,
                        )
                    except Exception as e:
                        # If tools spec invalid, retry without tools
                        err = str(e)
                        if "Invalid 'tools" in err:
                            log.error(f"Tool definition error: {err}")
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
                        # Pass tool name mapping if available
                        name_mapping = getattr(self.context, "tool_name_mapping", {})
                        await self.tool_processor.process_tool_calls(tool_calls, name_mapping)
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
    
    def _sanitize_conversation_history(self):
        """Ensure all tool names in conversation history follow OpenAI's pattern."""
        import re
        
        # Only process if we have history
        if not self.context.conversation_history:
            return
        
        sanitized_count = 0
        
        # Go through all messages in history
        for msg in self.context.conversation_history:
            # Fix tool calls in assistant messages
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    if tc.get("function") and "name" in tc["function"]:
                        name = tc["function"]["name"]
                        # If name contains a dot or doesn't match pattern, sanitize it
                        if '.' in name or not re.match(r'^[a-zA-Z0-9_-]+$', name):
                            # This name has dots or other invalid chars, sanitize it
                            sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
                            log.debug(f"Sanitizing tool name in history: {name} -> {sanitized}")
                            tc["function"]["name"] = sanitized
                            sanitized_count += 1
            
            # Fix tool messages
            if msg.get("role") == "tool" and "name" in msg:
                name = msg["name"]
                if '.' in name or not re.match(r'^[a-zA-Z0-9_-]+$', name):
                    # Sanitize
                    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
                    log.debug(f"Sanitizing tool message name in history: {name} -> {sanitized}")
                    msg["name"] = sanitized
                    sanitized_count += 1
        
        if sanitized_count > 0:
            log.debug(f"Sanitized {sanitized_count} tool name(s) in conversation history")
        
    async def _maybe_async_get_tools(self):
        """Helper to fetch tools, awaiting if necessary."""
        tools = self.context.tool_manager.get_tools_for_llm()
        if asyncio.iscoroutine(tools):
            return await tools
        return tools