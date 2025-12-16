# mcp_cli/chat/conversation.py - FIXED VERSION
"""
from __future__ import annotations

FIXED: Updated to work with the new OpenAI client universal tool compatibility system.
"""

import time
import asyncio
import logging
import os
from chuk_term.ui import output

# mcp cli imports
from mcp_cli.chat.models import Message, MessageRole
from mcp_cli.chat.tool_processor import ToolProcessor
from mcp_cli.llm.toon_optimizer import (
    ToonOptimizer,
    format_token_comparison,
    get_format_decision_message,
)

log = logging.getLogger(__name__)


class ConversationProcessor:
    """
    Class to handle LLM conversation processing with streaming support.

    Updated to work with universal tool compatibility system.
    """

    def __init__(self, context, ui_manager):
        self.context = context
        self.ui_manager = ui_manager
        self.tool_processor = ToolProcessor(context, ui_manager)

        # Initialize TOON optimizer based on config and provider
        # TOON optimization now supported for all providers
        toon_enabled = False
        provider = getattr(context, 'provider', '').lower()
        model = getattr(context, 'model', '')

        try:
            from mcp_cli.config import get_config

            config = get_config()
            # Enable TOON if config allows (now works with all providers)
            toon_enabled = config.enable_toon_optimization
            if config.enable_toon_optimization:
                log.info(f"TOON optimization enabled for provider '{provider}' with model '{model}'")
        except Exception as e:
            log.warning(f"Could not load TOON config, TOON optimization disabled: {e}")

        self.toon_optimizer = ToonOptimizer(
            enabled=toon_enabled,
            provider=getattr(context, 'provider', 'openai'),
            model=model
        )

    async def process_conversation(self, max_turns: int = 30):
        """Process the conversation loop, handling tool calls and responses with streaming.

        Args:
            max_turns: Maximum number of conversation turns before forcing exit (default: 30)
        """
        turn_count = 0
        last_tool_signature = None  # Track last tool call to detect duplicates
        tools_for_completion = None  # Will be set based on context
        try:
            while turn_count < max_turns:
                try:
                    turn_count += 1
                    start_time = time.time()

                    # Skip slash commands (already handled by UI)
                    last_msg = (
                        self.context.conversation_history[-1]
                        if self.context.conversation_history
                        else None
                    )
                    if last_msg:
                        content = last_msg.content or ""
                        if last_msg.role == MessageRole.USER and content.startswith(
                            "/"
                        ):
                            return

                    # Ensure OpenAI tools are loaded for function calling
                    if not getattr(self.context, "openai_tools", None):
                        await self._load_tools()

                    # REMOVED: Sanitization logic - now handled by universal tool compatibility
                    # The OpenAI client automatically handles tool name sanitization and restoration

                    # Always pass tools - let the model decide what to do
                    tools_for_completion = self.context.openai_tools
                    log.debug(
                        f"Passing {len(tools_for_completion) if tools_for_completion else 0} tools to completion"
                    )

                    # Log conversation history size for debugging
                    history_size = len(self.context.conversation_history)
                    log.debug(f"Conversation history has {history_size} messages")

                    # Log last few messages for debugging (truncated)
                    for i, msg in enumerate(self.context.conversation_history[-3:]):
                        role = msg.role if hasattr(msg, 'role') else msg.get('role', 'unknown')
                        content_preview = str(msg.content)[:100] if hasattr(msg, 'content') else str(msg.get('content', ''))[:100]
                        log.debug(f"  Message {history_size - 3 + i}: role={role}, content_preview={content_preview}")

                    # Check if client supports streaming
                    client = self.context.client

                    # For chuk-llm, check if create_completion accepts stream parameter
                    supports_streaming = hasattr(client, "create_completion")

                    if supports_streaming:
                        # Check if create_completion accepts stream parameter
                        import inspect

                        try:
                            sig = inspect.signature(client.create_completion)
                            has_stream_param = "stream" in sig.parameters
                            supports_streaming = has_stream_param
                        except Exception as e:
                            log.debug(f"Could not inspect signature: {e}")
                            supports_streaming = False

                    completion = None

                    # Prepare messages for sending
                    messages_to_send = [
                        msg.to_dict() for msg in self.context.conversation_history
                    ]

                    # Compare formats to show potential savings (messages only, not tools)
                    toon_comparison = self.toon_optimizer.compare_formats(
                        messages_to_send, None
                    )

                    # Display token comparison
                    if toon_comparison and toon_comparison.get("json_tokens", 0) > 0:
                        comparison_msg = format_token_comparison(toon_comparison)
                        output.info(comparison_msg)

                        # Apply TOON compression if enabled and beneficial
                        if self.toon_optimizer.enabled and toon_comparison.get("saved_tokens", 0) > 0:
                            # Compress messages using TOON (whitespace removal, content optimization)
                            messages_to_send = self.toon_optimizer.convert_to_toon_dict(messages_to_send)
                            output.info("Using TOON compression to reduce tokens")
                            log.info(f"TOON compression applied: {toon_comparison['saved_tokens']} tokens saved")
                        elif self.toon_optimizer.enabled:
                            output.info("Using JSON (no TOON savings for this request)")
                        else:
                            # TOON is disabled in config
                            output.info("Using JSON (TOON optimization disabled in config)")

                    # Always log API request payloads to file for debugging
                    try:
                        import json
                        import datetime

                        # Write to log file for inspection
                        log_file_path = os.path.expanduser("~/.mcp-cli/api_requests.log")
                        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

                        with open(log_file_path, "a") as logf:
                            timestamp = datetime.datetime.now().isoformat()

                            output_lines = [
                                "\n" + "=" * 80,
                                f"API REQUEST PAYLOAD [{timestamp}]",
                                "=" * 80,
                                f"Provider: {getattr(self.context, 'provider', 'unknown')}",
                                f"Model: {getattr(self.context, 'model', 'unknown')}",
                                f"Number of messages: {len(messages_to_send)}",
                                f"Number of tools: {len(tools_for_completion) if tools_for_completion else 0}",
                                "-" * 80,
                                "Messages being sent to LLM:",
                            ]

                            for idx, msg in enumerate(messages_to_send):
                                output_lines.append(f"\n[Message {idx}]")
                                output_lines.append(json.dumps(msg, indent=2, ensure_ascii=False))

                            if tools_for_completion:
                                output_lines.append("-" * 80)
                                output_lines.append(f"Tools being sent to LLM ({len(tools_for_completion)} tools):")
                                for idx, tool in enumerate(tools_for_completion[:3]):
                                    output_lines.append(f"\n[Tool {idx}]")
                                    output_lines.append(json.dumps(tool, indent=2, ensure_ascii=False))
                                if len(tools_for_completion) > 3:
                                    output_lines.append(f"\n... and {len(tools_for_completion) - 3} more tools")

                            output_lines.append("=" * 80 + "\n")

                            full_output = "\n".join(output_lines)
                            logf.write(full_output)
                            logf.flush()

                        log.debug(f"API request logged to: {log_file_path}")
                    except Exception as e:
                        log.error(f"Failed to write API request log: {e}")

                    if supports_streaming:
                        # Use streaming response handler
                        try:
                            completion = await self._handle_streaming_completion(
                                tools=tools_for_completion,
                                messages=messages_to_send,
                            )
                        except Exception as e:
                            log.warning(
                                f"Streaming failed, falling back to regular completion: {e}"
                            )
                            output.warning(
                                f"Streaming failed, falling back to regular completion: {e}"
                            )
                            completion = await self._handle_regular_completion(
                                tools=tools_for_completion,
                                messages=messages_to_send,
                            )
                    else:
                        # Regular completion
                        completion = await self._handle_regular_completion(
                            tools=tools_for_completion,
                            messages=messages_to_send,
                        )

                    response_content = completion.get("response", "No response")
                    tool_calls = completion.get("tool_calls", [])

                    # Log API response to file
                    try:
                        import json
                        import datetime

                        log_file_path = os.path.expanduser("~/.mcp-cli/api_requests.log")
                        with open(log_file_path, "a") as logf:
                            timestamp = datetime.datetime.now().isoformat()

                            output_lines = [
                                "\n" + "=" * 80,
                                f"API RESPONSE [{timestamp}]",
                                "=" * 80,
                                f"Provider: {getattr(self.context, 'provider', 'unknown')}",
                                f"Model: {getattr(self.context, 'model', 'unknown')}",
                                f"Response length: {len(response_content)} chars",
                                f"Tool calls: {len(tool_calls)}",
                                "-" * 80,
                            ]

                            if response_content:
                                output_lines.append("Response content:")
                                output_lines.append(response_content[:500])  # First 500 chars
                                if len(response_content) > 500:
                                    output_lines.append(f"... ({len(response_content) - 500} more chars)")

                            if tool_calls:
                                output_lines.append("-" * 80)
                                output_lines.append(f"Tool calls ({len(tool_calls)}):")
                                for idx, tc in enumerate(tool_calls):
                                    output_lines.append(f"\n[Tool Call {idx}]")
                                    output_lines.append(json.dumps(tc, indent=2, ensure_ascii=False))

                            output_lines.append("=" * 80 + "\n")

                            full_output = "\n".join(output_lines)
                            logf.write(full_output)
                            logf.flush()

                        log.debug(f"API response logged to: {log_file_path}")
                    except Exception as e:
                        log.error(f"Failed to write API response log: {e}")

                    # If model requested tool calls, execute them
                    if tool_calls and len(tool_calls) > 0:
                        log.debug(f"Processing {len(tool_calls)} tool calls from LLM")

                        # Check if we're at max turns
                        if turn_count >= max_turns:
                            output.warning(
                                f"Maximum conversation turns ({max_turns}) reached. Stopping to prevent infinite loop."
                            )
                            self.context.conversation_history.append(
                                Message(
                                    role=MessageRole.ASSISTANT,
                                    content="I've reached the maximum number of conversation turns. The tool results have been provided above.",
                                )
                            )
                            break

                        # Create signature to detect duplicate tool calls
                        import json

                        current_signature = []
                        for tc in tool_calls:
                            if hasattr(tc, "function"):
                                name = getattr(tc.function, "name", "")
                                args = getattr(tc.function, "arguments", "")
                            elif isinstance(tc, dict) and "function" in tc:
                                name = tc["function"].get("name", "")
                                args = tc["function"].get("arguments", "")
                            else:
                                continue
                            if isinstance(args, dict):
                                args = json.dumps(args, sort_keys=True)
                            current_signature.append(f"{name}:{args}")

                        current_sig_str = "|".join(sorted(current_signature))

                        # If this is a duplicate, stop looping and return control to user
                        if (
                            last_tool_signature
                            and current_sig_str == last_tool_signature
                        ):
                            log.warning(
                                f"Duplicate tool call detected: {current_sig_str}"
                            )
                            output.info(
                                "Tool has already been executed. Results are shown above."
                            )
                            break

                        last_tool_signature = current_sig_str

                        # Log the tool calls for debugging
                        for i, tc in enumerate(tool_calls):
                            log.debug(f"Tool call {i}: {tc}")

                        # FIXED: Get name mapping from universal tool compatibility system
                        name_mapping = getattr(self.context, "tool_name_mapping", {})
                        log.debug(f"Using name mapping: {name_mapping}")

                        # Process tool calls - this will handle streaming display
                        await self.tool_processor.process_tool_calls(
                            tool_calls, name_mapping
                        )
                        continue

                    # Reset duplicate tracking on text response
                    last_tool_signature = None

                    # Display assistant response (if not already displayed by streaming)
                    elapsed = completion.get("elapsed_time", time.time() - start_time)

                    if not completion.get("streaming", False):
                        # Non-streaming response, display normally
                        self.ui_manager.print_assistant_response(
                            response_content, elapsed
                        )
                    else:
                        # Streaming response - final display already handled by finish_streaming()
                        # Just mark streaming as stopped and clean up
                        self.ui_manager.stop_streaming_response()
                        # Clear streaming handler reference
                        if hasattr(self.ui_manager, "streaming_handler"):
                            self.ui_manager.streaming_handler = None

                    # Add to conversation history
                    self.context.conversation_history.append(
                        Message(role=MessageRole.ASSISTANT, content=response_content)
                    )
                    break

                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    output.error(f"Error during conversation processing: {exc}")
                    import traceback

                    traceback.print_exc()
                    self.context.conversation_history.append(
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=f"I encountered an error: {exc}",
                        )
                    )
                    break
        except asyncio.CancelledError:
            raise

    async def _handle_streaming_completion(
        self, tools: list | None = None, messages: list | None = None
    ) -> dict:
        """Handle streaming completion with UI integration.

        Args:
            tools: Tool definitions to pass to the LLM, or None to disable tools
            messages: Optimized messages to send (can be TOON format or regular)
        """
        from mcp_cli.chat.streaming_handler import StreamingResponseHandler

        # Use provided messages or convert from conversation history
        if messages is None:
            messages = [msg.to_dict() for msg in self.context.conversation_history]

        # Signal UI that streaming is starting
        self.ui_manager.start_streaming_response()

        # Set the streaming handler reference in UI manager for interruption support
        streaming_handler = StreamingResponseHandler(
            console=self.ui_manager.console, chat_display=self.ui_manager.display
        )
        self.ui_manager.streaming_handler = streaming_handler

        # Log the actual API call being made
        if log.isEnabledFor(logging.DEBUG):
            log.debug("=" * 80)
            log.debug("MAKING STREAMING API CALL TO LLM")
            log.debug("=" * 80)
            log.debug(f"Calling: streaming_handler.stream_response()")
            log.debug(f"Messages count: {len(messages)}")
            log.debug(f"Tools count: {len(tools) if tools else 0}")
            log.debug("=" * 80)

        try:
            completion = await streaming_handler.stream_response(
                client=self.context.client,
                messages=messages,
                tools=tools,
            )

            # Log streaming response completion
            if log.isEnabledFor(logging.DEBUG):
                log.debug("=" * 80)
                log.debug("STREAMING RESPONSE COMPLETE")
                log.debug("=" * 80)
                log.debug(f"Response keys: {list(completion.keys())}")
                if 'response' in completion:
                    log.debug(f"Response text length: {len(completion['response'])} chars")
                if 'tool_calls' in completion and completion['tool_calls']:
                    log.debug(f"Tool calls returned: {len(completion['tool_calls'])}")
                log.debug("=" * 80)

            # Enhanced tool call validation and logging
            if completion.get("tool_calls"):
                log.debug(
                    f"Streaming completion returned {len(completion['tool_calls'])} tool calls"
                )
                for i, tc in enumerate(completion["tool_calls"]):
                    log.debug(f"Streamed tool call {i}: {tc}")

                    # Validate tool call structure
                    if not self._validate_streaming_tool_call(tc):
                        log.warning(f"Invalid tool call structure from streaming: {tc}")
                        # Try to fix common issues
                        fixed_tc = self._fix_tool_call_structure(tc)
                        if fixed_tc:
                            completion["tool_calls"][i] = fixed_tc
                            log.debug(f"Fixed tool call {i}: {fixed_tc}")
                        else:
                            log.error(
                                f"Could not fix tool call {i}, removing from list"
                            )
                            completion["tool_calls"].pop(i)

            return completion

        finally:
            # Keep streaming handler reference for finalization
            # Will be cleared after finalization in main conversation loop
            pass

    async def _handle_regular_completion(
        self, tools: list | None = None, messages: list | None = None
    ) -> dict:
        """Handle regular (non-streaming) completion.

        Args:
            tools: Tool definitions to pass to the LLM, or None to disable tools
            messages: Optimized messages to send (can be TOON format or regular)
        """
        start_time = time.time()

        # Use provided messages or convert from conversation history
        if messages is None:
            messages = [msg.to_dict() for msg in self.context.conversation_history]

        # Log the actual API call being made
        if log.isEnabledFor(logging.DEBUG):
            log.debug("=" * 80)
            log.debug("MAKING API CALL TO LLM")
            log.debug("=" * 80)
            log.debug(f"Calling: client.create_completion()")
            log.debug(f"Messages count: {len(messages)}")
            log.debug(f"Tools count: {len(tools) if tools else 0}")
            log.debug("=" * 80)

        try:
            completion = await self.context.client.create_completion(
                messages=messages,
                tools=tools,
            )

            # Log the response
            if log.isEnabledFor(logging.DEBUG):
                log.debug("=" * 80)
                log.debug("RECEIVED API RESPONSE")
                log.debug("=" * 80)
                log.debug(f"Response keys: {list(completion.keys())}")
                if 'response' in completion:
                    log.debug(f"Response text length: {len(completion['response'])} chars")
                if 'tool_calls' in completion and completion['tool_calls']:
                    log.debug(f"Tool calls returned: {len(completion['tool_calls'])}")
                log.debug("=" * 80)
        except Exception as e:
            # If tools spec invalid, retry without tools
            err = str(e)
            if "Invalid 'tools" in err:
                log.error(f"Tool definition error: {err}")
                output.warning(
                    "Tool definitions rejected by model, retrying without tools..."
                )
                completion = await self.context.client.create_completion(
                    messages=messages
                )
            else:
                raise

        elapsed = time.time() - start_time
        completion["elapsed_time"] = elapsed
        completion["streaming"] = False

        result: dict = completion
        return result

    def _validate_streaming_tool_call(self, tool_call: dict) -> bool:
        """Validate that a tool call from streaming has the required structure."""
        try:
            if not isinstance(tool_call, dict):
                return False  # type: ignore[unreachable]

            # Check for required fields
            if "function" not in tool_call:
                return False

            function = tool_call["function"]
            if not isinstance(function, dict):
                return False

            # Check function has name
            if "name" not in function or not function["name"]:
                return False

            # Validate arguments if present
            if "arguments" in function:
                args = function["arguments"]
                if isinstance(args, str):
                    # Try to parse as JSON
                    try:
                        if args.strip():  # Don't try to parse empty strings
                            import json

                            json.loads(args)
                    except json.JSONDecodeError:
                        log.warning(f"Invalid JSON arguments in tool call: {args}")
                        return False
                elif not isinstance(args, dict):
                    # Arguments should be string or dict
                    return False

            return True

        except Exception as e:
            log.error(f"Error validating streaming tool call: {e}")
            return False

    def _fix_tool_call_structure(self, tool_call: dict) -> dict | None:
        """Try to fix common issues with tool call structure from streaming."""
        try:
            fixed = dict(tool_call)  # Make a copy

            # Ensure we have required fields
            if "id" not in fixed:
                fixed["id"] = f"call_{hash(str(tool_call)) % 10000}"

            if "type" not in fixed:
                fixed["type"] = "function"

            if "function" not in fixed:
                return None  # Can't fix this

            function = fixed["function"]

            # Fix empty name
            if not function.get("name"):
                return None  # Can't fix missing name

            # Fix arguments
            if "arguments" not in function:
                function["arguments"] = "{}"
            elif function["arguments"] is None:
                function["arguments"] = "{}"
            elif isinstance(function["arguments"], dict):
                # Convert dict to JSON string
                import json

                function["arguments"] = json.dumps(function["arguments"])
            elif not isinstance(function["arguments"], str):
                # Convert to string
                function["arguments"] = str(function["arguments"])

            # Validate the fixed version
            if self._validate_streaming_tool_call(fixed):
                return fixed
            else:
                return None

        except Exception as e:
            log.error(f"Error fixing tool call structure: {e}")
            return None

    async def _load_tools(self):
        """
        Load and adapt tools for the current provider.

        FIXED: Updated to use universal tool compatibility system.
        """
        try:
            if hasattr(self.context.tool_manager, "get_adapted_tools_for_llm"):
                # EXPLICITLY specify provider for proper adaptation
                provider = getattr(self.context, "provider", "openai")
                tools_and_mapping = (
                    await self.context.tool_manager.get_adapted_tools_for_llm(provider)
                )
                self.context.openai_tools = tools_and_mapping[0]
                self.context.tool_name_mapping = tools_and_mapping[1]
                log.debug(
                    f"Loaded {len(self.context.openai_tools)} adapted tools for {provider}"
                )

                # FIXED: No longer validate tool names here since universal compatibility handles it
                log.debug(f"Universal tool compatibility enabled for {provider}")

        except Exception as exc:
            log.error(f"Error loading tools: {exc}")
            self.context.openai_tools = []
            self.context.tool_name_mapping = {}
