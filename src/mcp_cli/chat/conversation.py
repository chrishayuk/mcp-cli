# mcp_cli/chat/conversation_v2.py
"""
Simplified conversation processor that leverages chuk-llm's streaming and tool handling.
"""

import asyncio
import logging
import time
from typing import Optional

from chuk_term.ui import output
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown

from mcp_cli.chat.tool_processor import ToolProcessor

logger = logging.getLogger(__name__)


class ConversationProcessor:
    """
    Simplified conversation processor that uses chuk-llm's native capabilities.

    This version removes redundant streaming handler and tool call extraction,
    leveraging chuk-llm's built-in streaming and tool handling.
    """

    def __init__(self, context, ui_manager):
        self.context = context
        self.ui_manager = ui_manager
        self.tool_processor = ToolProcessor(context, ui_manager)
        self._current_live_display: Optional[Live] = None

    async def process_conversation(self):
        """Process conversation using chuk-llm's native streaming."""
        try:
            # The user message has already been added to context by chat_handler
            # We just need to get the response from the LLM
            # No need to get the message from history - just process the conversation
            
            # Check if we should use streaming
            if self._should_use_streaming():
                # Pass None since the message is already in the conversation
                await self._handle_streaming_conversation(None)
            else:
                await self._handle_regular_conversation(None)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Error during conversation processing")
            output.error(f"Error processing message: {exc}")
            # Add error to context for recovery
            self.context.add_assistant_message(f"I encountered an error: {exc}")

    def _should_use_streaming(self) -> bool:
        """Check if streaming should be used."""
        # Check UI preference
        if hasattr(self.ui_manager, "streaming_enabled"):
            return self.ui_manager.streaming_enabled
        # Default to streaming
        return True

    async def _handle_streaming_conversation(self, user_message: Optional[str] = None):
        """Handle streaming conversation using chuk-llm's native streaming."""
        start_time = time.time()
        collected_response = ""
        collected_tool_calls = []

        # Signal UI that streaming is starting
        self.ui_manager.start_streaming_response()

        try:
            # Create live display for streaming
            self._current_live_display = Live(
                Panel("", title="⚡ Streaming", border_style="blue"),
                console=self.ui_manager.console,
                transient=True,
                refresh_per_second=4,
            )

            with self._current_live_display:
                chunk_count = 0

                # Get the last user message from conversation history if not provided
                if user_message is None:
                    # Get the last user message from the conversation
                    for msg in reversed(self.context.conversation_history):
                        if msg.get("role") == "user":
                            user_message = msg.get("content", "")
                            break
                    
                    if not user_message:
                        logger.warning("No user message found in conversation history")
                        return

                # Use chuk-llm's native streaming
                async for chunk in self.context.stream_with_tools(user_message):
                    chunk_count += 1
                    
                    # Debug: log what we're getting from the LLM
                    logger.debug(f"Stream chunk {chunk_count}: {type(chunk)} - {str(chunk)[:100]}")

                    # Check for interruption
                    if self.ui_manager.interrupt_requested:
                        logger.debug("Streaming interrupted by user")
                        break

                    # Process chunk based on type
                    if isinstance(chunk, dict):
                        # Handle response content
                        if "response" in chunk:
                            content = chunk["response"]
                            if content:
                                collected_response += content
                                self._update_streaming_display(
                                    collected_response, chunk_count, start_time
                                )

                        # Handle tool calls
                        if "tool_calls" in chunk:
                            tool_calls = chunk["tool_calls"]
                            if tool_calls:
                                collected_tool_calls.extend(tool_calls)

                    elif isinstance(chunk, str):
                        # Simple string chunk
                        collected_response += chunk
                        self._update_streaming_display(
                            collected_response, chunk_count, start_time
                        )

                    # Small delay for smooth display
                    await asyncio.sleep(0.01)

            # Streaming complete
            elapsed = time.time() - start_time

            # Process any tool calls
            if collected_tool_calls:
                logger.debug(
                    f"Processing {len(collected_tool_calls)} tool calls from stream"
                )
                await self.tool_processor.process_tool_calls(
                    collected_tool_calls, self.context.tool_name_mapping
                )
            else:
                # Display final response
                self._display_final_response(collected_response, elapsed, chunk_count)

            # Update conversation (chuk-llm handles this internally)
            self.context.add_assistant_message(collected_response)

        finally:
            self.ui_manager.stop_streaming_response()
            self._current_live_display = None

    async def _handle_regular_conversation(self, user_message: Optional[str] = None):
        """Handle non-streaming conversation using chuk-llm's ask."""
        start_time = time.time()

        # Get the last user message from conversation history if not provided
        if user_message is None:
            # Get the last user message from the conversation
            for msg in reversed(self.context.conversation_history):
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            if not user_message:
                logger.warning("No user message found in conversation history")
                return

        with output.loading("Generating response..."):
            # Use chuk-llm's ask with tools
            result = await self.context.ask_with_tools(user_message)

        elapsed = time.time() - start_time

        # Extract response and tool calls
        response_content = result.get("response", "")
        tool_calls = result.get("tool_calls", [])

        # Process tool calls if any
        if tool_calls:
            logger.debug(f"Processing {len(tool_calls)} tool calls")
            await self.tool_processor.process_tool_calls(
                tool_calls, self.context.tool_name_mapping
            )
        else:
            # Display response
            self.ui_manager.print_assistant_response(response_content, elapsed)

        # Update conversation (chuk-llm handles this internally)
        self.context.add_assistant_message(response_content)

    def _update_streaming_display(self, content: str, chunks: int, start_time: float):
        """Update the streaming display with current content."""
        if not self._current_live_display:
            return

        elapsed = time.time() - start_time

        # Create formatted content
        try:
            display_content = Markdown(content + " ▌")  # Add cursor
        except Exception:
            display_content = content + " ▌"

        # Create status text
        status = f"⚡ Streaming • {chunks} chunks • {elapsed:.1f}s"

        # Update panel
        panel = Panel(
            display_content,
            title=status,
            border_style="blue",
            padding=(0, 1),
        )

        self._current_live_display.update(panel)

    def _display_final_response(self, content: str, elapsed: float, chunks: int):
        """Display the final streamed response."""
        # Calculate stats
        words = len(content.split())

        # Create subtitle with stats
        subtitle_parts = [f"Response time: {elapsed:.2f}s"]
        if chunks > 1:
            subtitle_parts.append(f"Streamed: {chunks} chunks")
        if elapsed > 0:
            subtitle_parts.append(f"{words / elapsed:.1f} words/s")

        subtitle = " | ".join(subtitle_parts)

        # Format content
        try:
            formatted_content = Markdown(content)
        except Exception:
            formatted_content = content

        # Display final panel
        output.print(
            Panel(
                formatted_content,
                title="Assistant",
                subtitle=subtitle,
                padding=(0, 1),
            )
        )
