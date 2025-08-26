# mcp_cli/chat/streaming_handler.py - COMPLETE FINAL VERSION
"""
Enhanced streaming response handler for MCP CLI chat interface.
Handles async chunk yielding from chuk-llm with live UI updates and better integration.
Now includes proper tool call extraction from streaming chunks.

FINAL FIX: Proper parameter accumulation across multiple streaming chunks.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from chuk_term.ui import output
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown

from mcp_cli.logging_config import get_logger

logger = get_logger("streaming")


class StreamingResponseHandler:
    """Enhanced streaming handler with better UI integration and error handling."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or output._console
        self.current_response = ""
        self.live_display: Optional[Live] = None
        self.start_time = 0.0
        self.chunk_count = 0
        self.is_streaming = False
        self._response_complete = False
        self._interrupted = False

        # Tool call tracking for streaming
        self._accumulated_tool_calls = []
        self._current_tool_call = None

    async def stream_response(
        self,
        client,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Stream response from LLM with live UI updates and enhanced error handling.

        Args:
            client: LLM client with streaming support
            messages: Conversation messages
            tools: Available tools for function calling
            **kwargs: Additional arguments for completion

        Returns:
            Complete response dictionary
        """
        self.current_response = ""
        self.chunk_count = 0
        self.start_time = time.time()
        self.is_streaming = True
        self._response_complete = False
        self._interrupted = False
        self._accumulated_tool_calls = []
        self._current_tool_call = None

        try:
            # Check if client supports streaming via create_completion with stream=True
            if hasattr(client, "create_completion"):
                return await self._handle_chuk_llm_streaming(
                    client, messages, tools, **kwargs
                )
            else:
                # Client doesn't support completion, fallback
                logger.debug(
                    "Client doesn't support create_completion, falling back to regular completion"
                )
                return await self._handle_regular_completion(
                    client, messages, tools, **kwargs
                )

        finally:
            self.is_streaming = False
            if self.live_display:
                # Show final response if not already shown
                if not self._response_complete:
                    # Stop live display and show final response.  If not done in this order, the live display
                    # transient removal can cause issues with overwriting the final response.
                    self.live_display.stop()
                    self.live_display = None
                    self._show_final_response()
                else:
                    self.live_display.stop()
                    self.live_display = None

    def interrupt_streaming(self):
        """Interrupt the current streaming operation."""
        self._interrupted = True
        logger.debug("Streaming interrupted by user")

    def _show_final_response(self):
        """Display the final complete response with enhanced formatting."""
        if self._response_complete or not self.current_response:
            return

        elapsed = time.time() - self.start_time

        # Calculate stats
        words = len(self.current_response.split())
        chars = len(self.current_response)

        # Create subtitle with stats
        subtitle_parts = [f"Response time: {elapsed:.2f}s"]
        if self.chunk_count > 1:
            subtitle_parts.append(f"Streamed: {self.chunk_count} chunks")
        if elapsed > 0:
            subtitle_parts.append(f"{words / elapsed:.1f} words/s")

        subtitle = " | ".join(subtitle_parts)

        # Format content
        try:
            # Use Markdown for formatted text
            content = Markdown(self.current_response)
        except Exception as e:
            # Fallback to Text if Markdown parsing fails
            logger.debug(f"Markdown parsing failed: {e}")
            content = Text(self.current_response)

        # Display final panel
        output.print(
            Panel(
                content,
                title="Assistant",
                subtitle=subtitle,
                # style="bold blue",
                padding=(0, 1),
            )
        )
        self._response_complete = True

    async def _handle_chuk_llm_streaming(
        self,
        client,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Handle chuk-llm's streaming with proper tool call accumulation."""
        tool_calls = []

        # Start live display
        self._start_live_display()

        try:
            # Use chuk-llm's streaming approach
            async for chunk in client.create_completion(
                messages=messages, tools=tools, stream=True, **kwargs
            ):
                if self._interrupted:
                    logger.debug("Breaking from stream due to interruption")
                    break

                await self._process_chunk(chunk, tool_calls)

            # IMPORTANT: After streaming is complete, finalize any remaining tool calls
            await self._finalize_streaming_tool_calls(tool_calls)

        except asyncio.CancelledError:
            logger.debug("Streaming cancelled")
            self._interrupted = True
            raise
        except Exception as e:
            logger.error(f"Streaming error in chuk-llm streaming: {e}")
            raise

        # Build final response
        elapsed = time.time() - self.start_time
        result = {
            "response": self.current_response,
            "tool_calls": tool_calls,
            "chunks_received": self.chunk_count,
            "elapsed_time": elapsed,
            "streaming": True,
            "interrupted": self._interrupted,
        }

        logger.debug(f"Streaming completed: {len(tool_calls)} tool calls extracted")
        for i, tc in enumerate(tool_calls):
            logger.debug(
                f"Tool call {i}: {tc['function']['name']} with args: {tc['function']['arguments']}"
            )

        return result

    async def _handle_stream_completion(
        self,
        client,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Handle alternative stream_completion method."""
        tool_calls = []

        # Start live display
        self._start_live_display()

        try:
            async for chunk in client.stream_completion(
                messages=messages, tools=tools, **kwargs
            ):
                if self._interrupted:
                    logger.debug("Breaking from stream due to interruption")
                    break

                await self._process_chunk(chunk, tool_calls)

        except asyncio.CancelledError:
            logger.debug("Streaming cancelled")
            self._interrupted = True
            raise
        except Exception as e:
            logger.error(f"Streaming error in stream_completion: {e}")
            raise

        # Build final response
        elapsed = time.time() - self.start_time
        return {
            "response": self.current_response,
            "tool_calls": tool_calls,
            "chunks_received": self.chunk_count,
            "elapsed_time": elapsed,
            "streaming": True,
            "interrupted": self._interrupted,
        }

    async def _handle_regular_completion(
        self,
        client,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Fallback to regular non-streaming completion."""
        logger.debug("Using non-streaming completion")

        # Show a simple loading indicator
        with output.loading("Generating response..."):
            result = await client.create_completion(
                messages=messages, tools=tools, **kwargs
            )

        return {
            "response": result.get("response", ""),
            "tool_calls": result.get("tool_calls", []),
            "chunks_received": 1,
            "elapsed_time": time.time() - self.start_time,
            "streaming": False,
            "interrupted": False,
        }

    def _start_live_display(self):
        """Start the live display for streaming updates."""
        if not self.live_display:
            self.live_display = Live(
                self._create_display_content(),
                console=self.console,
                transient=True,
                refresh_per_second=4,  # 10 FPS for smooth updates
                vertical_overflow="visible",
            )
            self.live_display.start()

    async def _process_chunk(
        self, chunk: Dict[str, Any], tool_calls: List[Dict[str, Any]]
    ):
        """Process a single streaming chunk with enhanced error handling and tool call support."""
        self.chunk_count += 1

        try:
            # Log chunk for debugging
            logger.debug(f"Processing chunk {self.chunk_count}: {chunk}")

            # Extract content from chunk
            content = self._extract_chunk_content(chunk)
            if content:
                self.current_response += content

            # Handle tool calls in chunks
            tool_call_data = self._extract_tool_calls_from_chunk(chunk)
            if tool_call_data:
                logger.debug(f"Extracted tool call data: {tool_call_data}")
                self._process_tool_call_chunk(tool_call_data, tool_calls)

            # Update live display
            if self.live_display and not self._interrupted:
                self.live_display.update(self._create_display_content())

            # Small delay to prevent overwhelming the terminal
            await asyncio.sleep(0.01)

        except Exception as e:
            logger.warning(f"Error processing chunk: {e}")
            # Continue processing other chunks

    def _extract_chunk_content(self, chunk: Dict[str, Any]) -> str:
        """Extract text content from a chuk-llm streaming chunk."""
        try:
            # chuk-llm streaming format - chunk has "response" field with content
            if isinstance(chunk, dict):
                # Primary format for chuk-llm
                if "response" in chunk:
                    return (
                        str(chunk["response"]) if chunk["response"] is not None else ""
                    )

                # Alternative formats (for compatibility)
                elif "content" in chunk:
                    return str(chunk["content"])
                elif "text" in chunk:
                    return str(chunk["text"])
                elif "delta" in chunk and isinstance(chunk["delta"], dict):
                    delta_content = chunk["delta"].get("content")
                    return str(delta_content) if delta_content is not None else ""
                elif "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    if "delta" in choice and "content" in choice["delta"]:
                        delta_content = choice["delta"]["content"]
                        return str(delta_content) if delta_content is not None else ""
            elif isinstance(chunk, str):
                return chunk

        except Exception as e:
            logger.debug(f"Error extracting content from chunk: {e}")

        return ""

    def _extract_tool_calls_from_chunk(
        self, chunk: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract tool call data from a streaming chunk."""
        try:
            if isinstance(chunk, dict):
                # Direct tool_calls field
                if "tool_calls" in chunk:
                    logger.debug(
                        f"Found direct tool_calls in chunk: {chunk['tool_calls']}"
                    )
                    return chunk["tool_calls"]

                # OpenAI-style delta format
                if "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    if "delta" in choice:
                        delta = choice["delta"]
                        if "tool_calls" in delta:
                            logger.debug(
                                f"Found tool_calls in delta: {delta['tool_calls']}"
                            )
                            return delta["tool_calls"]
                        # Sometimes tool_calls come in function_call format
                        if "function_call" in delta:
                            logger.debug(
                                f"Found function_call in delta: {delta['function_call']}"
                            )
                            return {"function_call": delta["function_call"]}

                # Alternative formats
                if "function_call" in chunk:
                    logger.debug(
                        f"Found direct function_call: {chunk['function_call']}"
                    )
                    return {"function_call": chunk["function_call"]}

        except Exception as e:
            logger.debug(f"Error extracting tool calls from chunk: {e}")

        return None

    def _process_tool_call_chunk(
        self, tool_call_data: Dict[str, Any], tool_calls: List[Dict[str, Any]]
    ):
        """Process tool call chunk data and accumulate complete tool calls."""
        try:
            if isinstance(tool_call_data, list):
                # Array of tool calls
                for tc_item in tool_call_data:
                    self._accumulate_tool_call(tc_item, tool_calls)
            elif isinstance(tool_call_data, dict):
                # Single tool call or function call
                if "function_call" in tool_call_data:
                    # Legacy function_call format - convert to tool_calls format
                    fc = tool_call_data["function_call"]
                    converted = {
                        "id": f"call_{len(self._accumulated_tool_calls)}",
                        "type": "function",
                        "function": fc,
                    }
                    self._accumulate_tool_call(converted, tool_calls)
                else:
                    # Direct tool call
                    self._accumulate_tool_call(tool_call_data, tool_calls)

        except Exception as e:
            logger.warning(f"Error processing tool call chunk: {e}")

    def _accumulate_tool_call(
        self, tool_call_item: Dict[str, Any], tool_calls: List[Dict[str, Any]]
    ):
        """
        Accumulate streaming tool call data into complete tool calls.

        FINAL FIX: Proper accumulation that waits for complete parameters.
        """
        try:
            tc_id = tool_call_item.get("id")
            tc_index = tool_call_item.get("index", 0)

            logger.debug(
                f"Accumulating tool call: id={tc_id}, index={tc_index}, item={tool_call_item}"
            )

            # Find existing tool call or create new one
            existing_tc = None
            for tc in self._accumulated_tool_calls:
                if tc.get("id") == tc_id or (
                    tc_id is None and tc.get("index") == tc_index
                ):
                    existing_tc = tc
                    break

            if existing_tc is None:
                # Create new tool call
                existing_tc = {
                    "id": tc_id or f"call_{len(self._accumulated_tool_calls)}",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                    "index": tc_index,
                    "_streaming_state": {
                        "chunks_received": 0,
                        "name_complete": False,
                        "args_started": False,
                        "args_complete": False,
                    },
                }
                self._accumulated_tool_calls.append(existing_tc)
                logger.debug(f"Created new tool call: {existing_tc}")

            # Update streaming state
            existing_tc["_streaming_state"]["chunks_received"] += 1

            if "type" in tool_call_item:
                existing_tc["type"] = tool_call_item["type"]

            if "function" in tool_call_item:
                func_data = tool_call_item["function"]
                existing_func = existing_tc["function"]

                # Accumulate function name
                if "name" in func_data and func_data["name"] is not None:
                    new_name = str(func_data["name"])
                    if new_name and not existing_func["name"]:
                        existing_func["name"] += new_name
                        existing_tc["_streaming_state"]["name_complete"] = True
                        logger.debug(f"Accumulated name: '{existing_func['name']}'")

                # CRITICAL: Accumulate arguments properly
                if "arguments" in func_data:
                    new_args = func_data["arguments"]
                    current_args = existing_func["arguments"]

                    logger.debug(
                        f"Accumulating arguments: current='{current_args}', new='{new_args}'"
                    )

                    if new_args is not None:
                        existing_tc["_streaming_state"]["args_started"] = True

                        if isinstance(new_args, dict):
                            # Complete arguments object received
                            existing_func["arguments"] = json.dumps(new_args)
                            existing_tc["_streaming_state"]["args_complete"] = True
                            logger.debug(
                                f"Complete arguments dict received for {existing_func['name']}"
                            )
                        elif isinstance(new_args, str):
                            if new_args.strip():
                                # Non-empty string - accumulate
                                if not current_args:
                                    existing_func["arguments"] = new_args
                                else:
                                    merged = self._merge_argument_strings(
                                        current_args, new_args
                                    )
                                    existing_func["arguments"] = merged

                                # Check if we now have complete JSON
                                if self._is_complete_json(existing_func["arguments"]):
                                    existing_tc["_streaming_state"]["args_complete"] = (
                                        True
                                    )
                                    logger.debug(
                                        f"Arguments appear complete for {existing_func['name']}"
                                    )
                            # Empty string might indicate completion
                            elif existing_tc["_streaming_state"]["args_started"]:
                                existing_tc["_streaming_state"]["args_complete"] = True
                                logger.debug(
                                    f"Empty args received - marking complete for {existing_func['name']}"
                                )
                        else:
                            # Other type - convert to string
                            existing_func["arguments"] += str(new_args)

                    logger.debug(
                        f"Final accumulated arguments: '{existing_func['arguments']}'"
                    )

            # Check if this tool call is ready to be finalized
            # Don't finalize during streaming - wait for end

        except Exception as e:
            logger.error(f"Error accumulating tool call: {e}")
            import traceback

            traceback.print_exc()

    def _is_complete_json(self, json_str: str) -> bool:
        """Check if a string contains complete, valid JSON."""
        try:
            if not json_str or not json_str.strip():
                return False
            parsed = json.loads(json_str)
            return isinstance(parsed, dict)  # We expect objects for tool arguments
        except json.JSONDecodeError:
            return False

    def _merge_argument_strings(self, current: str, new: str) -> str:
        """Intelligently merge argument strings from streaming chunks."""
        try:
            # If both are empty, return empty
            if not current.strip() and not new.strip():
                return ""

            # If one is empty, return the other
            if not current.strip():
                return new
            if not new.strip():
                return current

            # Try to parse both as JSON first
            try:
                current_json = json.loads(current)
                new_json = json.loads(new)

                # Both are valid JSON - merge them
                if isinstance(current_json, dict) and isinstance(new_json, dict):
                    current_json.update(new_json)
                    return json.dumps(current_json)
                else:
                    # One is not a dict - just use the newer one
                    return new

            except json.JSONDecodeError:
                # At least one is not valid JSON - try concatenation
                combined = current + new

                # Test if concatenation creates valid JSON
                try:
                    json.loads(combined)
                    return combined
                except json.JSONDecodeError:
                    # Concatenation didn't work - try with fixes
                    return self._fix_concatenated_json(combined)

        except Exception as e:
            logger.warning(f"Error merging argument strings: {e}")
            # Fallback - just concatenate
            return current + new

    def _fix_concatenated_json(self, json_str: str) -> str:
        """Attempt to fix concatenated JSON strings from streaming."""
        try:
            # Common fixes
            fixed = json_str

            # Fix missing opening brace
            if not fixed.strip().startswith("{") and ":" in fixed:
                fixed = "{" + fixed

            # Fix missing closing brace
            if not fixed.strip().endswith("}") and ":" in fixed:
                fixed = fixed + "}"

            # Fix concatenated objects: }{"key": "value"} -> },{"key": "value"}
            fixed = fixed.replace("}{", "},{")

            # Try to parse the fixed version
            json.loads(fixed)
            return fixed

        except json.JSONDecodeError:
            # Still invalid - return as-is and let validation handle it
            logger.debug(f"Could not fix concatenated JSON: {json_str}")
            return json_str

    async def _finalize_streaming_tool_calls(self, tool_calls: List[Dict[str, Any]]):
        """
        Finalize accumulated tool calls after streaming is complete.

        This is where we decide which tool calls are complete and ready to execute.
        """
        logger.debug("Finalizing streaming tool calls after completion")

        for tc in self._accumulated_tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", "")
            state = tc.get("_streaming_state", {})

            logger.debug(f"Finalizing tool call: {name}")
            logger.debug(f"  State: {state}")
            logger.debug(f"  Args: '{args}'")

            # Skip if already added
            if any(existing_tc.get("id") == tc.get("id") for existing_tc in tool_calls):
                continue

            # Must have a name
            if not name:
                logger.debug("Skipping tool call without name")
                continue

            # Generic parameter validation - no hard-coded tool names
            if not args or args.strip() == "":
                # No arguments provided
                func["arguments"] = "{}"
                logger.debug(f"Finalizing tool with empty args: {name}")

            elif args.strip() == "{}":
                # Empty JSON object - this could be valid for some tools
                logger.debug(f"Finalizing tool with empty object: {name}")

            else:
                # Has some arguments - validate they're proper JSON
                try:
                    parsed = json.loads(args)
                    if isinstance(parsed, dict):
                        logger.debug(f"Finalizing tool with valid args: {name}")
                    else:
                        logger.warning(
                            f"Tool {name} has non-object arguments: {type(parsed)}"
                        )
                        # Still allow it - some tools might accept non-object args
                except json.JSONDecodeError:
                    logger.warning(f"Tool {name} has invalid JSON arguments: {args}")
                    # Try to fix it or skip
                    fixed_args = self._fix_concatenated_json(args)
                    try:
                        json.loads(fixed_args)
                        func["arguments"] = fixed_args
                        logger.debug(f"Fixed JSON for tool: {name}")
                    except json.JSONDecodeError:
                        logger.error(f"Cannot fix JSON for tool {name}, skipping")
                        continue

            # Clean up and add to final list
            final_tc = self._clean_tool_call_for_final_list(tc)
            tool_calls.append(final_tc)
            logger.info(f"✅ Finalized tool call: {final_tc['function']['name']}")

    def _clean_tool_call_for_final_list(
        self, tool_call: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Clean up tool call for final list by removing internal tracking fields."""
        cleaned = dict(tool_call)

        # Remove streaming state
        if "_streaming_state" in cleaned:
            del cleaned["_streaming_state"]

        # Ensure proper structure
        if "function" in cleaned and "arguments" in cleaned["function"]:
            args = cleaned["function"]["arguments"]
            if isinstance(args, str):
                # Ensure it's valid JSON
                try:
                    parsed = json.loads(args)
                    cleaned["function"]["arguments"] = json.dumps(parsed)
                except json.JSONDecodeError:
                    # Invalid JSON - use empty object
                    cleaned["function"]["arguments"] = "{}"
            elif isinstance(args, dict):
                cleaned["function"]["arguments"] = json.dumps(args)
            else:
                cleaned["function"]["arguments"] = "{}"

        return cleaned

    def _create_display_content(self):
        """Create enhanced content for live display."""
        elapsed = time.time() - self.start_time

        # Create enhanced status line
        status_text = Text()
        status_text.append("⚡ Streaming", style="cyan bold")
        status_text.append(f" • {self.chunk_count} chunks", style="dim")
        status_text.append(f" • {elapsed:.1f}s", style="dim")

        # Show tool call info if any are accumulating
        if self._accumulated_tool_calls:
            # Count tools with names
            named_tools = sum(
                1
                for tc in self._accumulated_tool_calls
                if tc.get("function", {}).get("name")
            )
            status_text.append(
                f" • {named_tools}/{len(self._accumulated_tool_calls)} tools",
                style="dim magenta",
            )

        # Show performance metrics if we have enough data
        if elapsed > 1.0 and self.current_response:
            words = len(self.current_response.split())
            chars = len(self.current_response)
            words_per_sec = words / elapsed
            chars_per_sec = chars / elapsed

            status_text.append(f" • {words_per_sec:.1f} words/s", style="dim green")
            status_text.append(f" • {chars_per_sec:.0f} chars/s", style="dim green")

        # Handle interruption state
        if self._interrupted:
            status_text.append(" • INTERRUPTED", style="red bold")

        # Response content with typing cursor
        if self.current_response:
            try:
                # Progressive markdown rendering with cursor
                display_text = self.current_response
                if not self._interrupted:
                    display_text += " ▌"  # Add typing cursor
                response_content = Markdown(markup=display_text)
            except Exception as e:
                # Fallback to plain text if markdown fails
                logger.debug(f"Markdown rendering failed: {e}")
                display_text = self.current_response
                if not self._interrupted:
                    display_text += " ▌"
                response_content = Text(display_text)
        else:
            # Show just cursor when no content yet
            cursor_style = "dim" if not self._interrupted else "red"
            response_content = Text("▌", style=cursor_style)

        # Create panel with dynamic styling
        border_style = "blue" if not self._interrupted else "red"

        return Panel(
            response_content,
            title=status_text,
            title_align="left",
            border_style=border_style,
            padding=(0, 1),
        )
