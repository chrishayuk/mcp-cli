# mcp_cli/chat/streaming_handler.py
"""
Enhanced streaming response handler for MCP CLI chat interface.
Handles async chunk yielding from chuk-llm with live UI updates and better integration.
Now includes proper tool call extraction from streaming chunks.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, AsyncIterator

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown

from mcp_cli.logging_config import get_logger

logger = get_logger("streaming")


class StreamingResponseHandler:
    """Enhanced streaming handler with better UI integration and error handling."""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
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
        **kwargs
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
            if hasattr(client, 'create_completion'):
                return await self._handle_chuk_llm_streaming(client, messages, tools, **kwargs)
            else:
                # Client doesn't support completion, fallback
                logger.debug("Client doesn't support create_completion, falling back to regular completion")
                return await self._handle_regular_completion(client, messages, tools, **kwargs)
                
        finally:
            self.is_streaming = False
            if self.live_display:
                # Show final response if not already shown
                if not self._response_complete:
                    self._show_final_response()
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
            subtitle_parts.append(f"{words/elapsed:.1f} words/s")
        
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
        self.console.print(
            Panel(
                content,
                title="Assistant",
                subtitle=subtitle,
                style="bold blue",
                padding=(0, 1)
            )
        )
        self._response_complete = True
    
    async def _handle_chuk_llm_streaming(
        self, 
        client, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle chuk-llm's streaming with create_completion(stream=True)."""
        tool_calls = []
        
        # Start live display
        self._start_live_display()
        
        try:
            # Use chuk-llm's streaming approach
            async for chunk in client.create_completion(
                messages=messages, 
                tools=tools,
                stream=True,
                **kwargs
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
            "interrupted": self._interrupted
        }
        
        return result
    
    async def _handle_stream_completion(
        self, 
        client, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Handle alternative stream_completion method."""
        tool_calls = []
        
        # Start live display
        self._start_live_display()
        
        try:
            async for chunk in client.stream_completion(
                messages=messages, 
                tools=tools, 
                **kwargs
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
            "interrupted": self._interrupted
        }
    
    async def _handle_regular_completion(
        self, 
        client, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Fallback to regular non-streaming completion."""
        logger.debug("Using non-streaming completion")
        
        # Show a simple loading indicator
        with self.console.status("[cyan]Generating response...[/cyan]", spinner="dots"):
            result = await client.create_completion(messages=messages, tools=tools, **kwargs)
        
        return {
            "response": result.get("response", ""),
            "tool_calls": result.get("tool_calls", []),
            "chunks_received": 1,
            "elapsed_time": time.time() - self.start_time,
            "streaming": False,
            "interrupted": False
        }
    
    def _start_live_display(self):
        """Start the live display for streaming updates."""
        if not self.live_display:
            self.live_display = Live(
                self._create_display_content(),
                console=self.console,
                refresh_per_second=10,  # 10 FPS for smooth updates
                vertical_overflow="visible"
            )
            self.live_display.start()
    
    async def _process_chunk(self, chunk: Dict[str, Any], tool_calls: List[Dict[str, Any]]):
        """Process a single streaming chunk with enhanced error handling and tool call support."""
        self.chunk_count += 1
        
        try:
            # Extract content from chunk
            content = self._extract_chunk_content(chunk)
            if content:
                self.current_response += content
            
            # Handle tool calls in chunks - ENHANCED TOOL CALL PROCESSING
            tool_call_data = self._extract_tool_calls_from_chunk(chunk)
            if tool_call_data:
                # Process tool call data and accumulate complete tool calls
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
                    return str(chunk["response"]) if chunk["response"] is not None else ""
                
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
    
    def _extract_tool_calls_from_chunk(self, chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract tool call data from a streaming chunk."""
        try:
            if isinstance(chunk, dict):
                # Direct tool_calls field
                if "tool_calls" in chunk:
                    return chunk["tool_calls"]
                
                # OpenAI-style delta format
                if "choices" in chunk and chunk["choices"]:
                    choice = chunk["choices"][0]
                    if "delta" in choice:
                        delta = choice["delta"]
                        if "tool_calls" in delta:
                            return delta["tool_calls"]
                        # Sometimes tool_calls come in function_call format
                        if "function_call" in delta:
                            return {"function_call": delta["function_call"]}
                
                # Alternative formats
                if "function_call" in chunk:
                    return {"function_call": chunk["function_call"]}
                    
        except Exception as e:
            logger.debug(f"Error extracting tool calls from chunk: {e}")
            
        return None
    
    def _process_tool_call_chunk(self, tool_call_data: Dict[str, Any], tool_calls: List[Dict[str, Any]]):
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
                        "function": fc
                    }
                    self._accumulate_tool_call(converted, tool_calls)
                else:
                    # Direct tool call
                    self._accumulate_tool_call(tool_call_data, tool_calls)
                    
        except Exception as e:
            logger.warning(f"Error processing tool call chunk: {e}")
    
    def _accumulate_tool_call(self, tool_call_item: Dict[str, Any], tool_calls: List[Dict[str, Any]]):
        """Accumulate streaming tool call data into complete tool calls."""
        try:
            tc_id = tool_call_item.get("id")
            tc_index = tool_call_item.get("index", 0)
            
            # Find existing tool call or create new one
            existing_tc = None
            for tc in self._accumulated_tool_calls:
                if tc.get("id") == tc_id or (tc_id is None and tc.get("index") == tc_index):
                    existing_tc = tc
                    break
            
            if existing_tc is None:
                # Create new tool call
                existing_tc = {
                    "id": tc_id or f"call_{len(self._accumulated_tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": "",
                        "arguments": ""
                    },
                    "index": tc_index
                }
                self._accumulated_tool_calls.append(existing_tc)
            
            # Update the tool call with new data
            if "type" in tool_call_item:
                existing_tc["type"] = tool_call_item["type"]
            
            if "function" in tool_call_item:
                func_data = tool_call_item["function"]
                existing_func = existing_tc["function"]
                
                # Accumulate function name
                if "name" in func_data:
                    if func_data["name"] is not None:
                        existing_func["name"] += str(func_data["name"])
                
                # Accumulate function arguments
                if "arguments" in func_data:
                    if func_data["arguments"] is not None:
                        existing_func["arguments"] += str(func_data["arguments"])
            
            # Check if this tool call is complete and add to final list
            if self._is_tool_call_complete(existing_tc) and existing_tc not in tool_calls:
                # Validate the tool call before adding
                if self._validate_tool_call(existing_tc):
                    tool_calls.append(dict(existing_tc))  # Make a copy
                    logger.debug(f"Complete tool call accumulated: {existing_tc['function']['name']}")
                
        except Exception as e:
            logger.warning(f"Error accumulating tool call: {e}")
    
    def _is_tool_call_complete(self, tool_call: Dict[str, Any]) -> bool:
        """Check if a tool call has all required fields and appears complete."""
        try:
            if not tool_call.get("function"):
                return False
            
            func = tool_call["function"]
            name = func.get("name", "")
            args = func.get("arguments", "")
            
            # Tool call is complete if:
            # 1. Has a name
            # 2. Arguments appear to be valid JSON or empty
            if not name:
                return False
            
            # Try to parse arguments as JSON if not empty
            if args.strip():
                try:
                    json.loads(args)
                except json.JSONDecodeError:
                    # Still accumulating arguments
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error checking tool call completeness: {e}")
            return False
    
    def _validate_tool_call(self, tool_call: Dict[str, Any]) -> bool:
        """Validate that a tool call has valid structure."""
        try:
            if not isinstance(tool_call, dict):
                return False
            
            if "function" not in tool_call:
                return False
            
            func = tool_call["function"]
            if not isinstance(func, dict):
                return False
            
            name = func.get("name", "")
            if not name or not isinstance(name, str):
                return False
            
            # Validate arguments if present
            args = func.get("arguments", "")
            if args:
                try:
                    json.loads(args)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in tool call arguments: {args}")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating tool call: {e}")
            return False
    
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
            complete_calls = sum(1 for tc in self._accumulated_tool_calls if self._is_tool_call_complete(tc))
            total_calls = len(self._accumulated_tool_calls)
            status_text.append(f" • {complete_calls}/{total_calls} tools", style="dim magenta")
        
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
                response_content = Markdown(display_text)
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
            padding=(0, 1)
        )