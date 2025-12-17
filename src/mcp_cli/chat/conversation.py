# mcp_cli/chat/conversation.py - FIXED VERSION
"""
from __future__ import annotations

FIXED: Updated to work with the new OpenAI client universal tool compatibility system.
Clean Pydantic models - no dictionary goop!

ENHANCED: Added tool state management to prevent "model getting lost":
- Caches tool results so duplicates return cached values
- Injects compact state summaries back to the model
- Continues conversation instead of aborting on duplicate calls
"""

import time
import asyncio
import logging
from chuk_term.ui import output

# mcp cli imports - using chuk_llm canonical models
from mcp_cli.chat.response_models import (
    CompletionResponse,
    Message,
    MessageRole,
)
from mcp_cli.chat.tool_processor import ToolProcessor
from chuk_ai_session_manager.guards import get_tool_state

log = logging.getLogger(__name__)


class ConversationProcessor:
    """
    Class to handle LLM conversation processing with streaming support.

    Updated to work with universal tool compatibility system.

    ENHANCED: Now includes tool state management to prevent "model getting lost":
    - Tracks tool call results in a cache
    - Returns cached values on duplicate calls instead of aborting
    - Injects state summaries to help model track computed values
    """

    # Tool name patterns that are polling/status tools - exempt from loop detection
    # These tools are expected to be called repeatedly with the same args
    POLLING_TOOL_PATTERNS = frozenset({
        "status",
        "poll",
        "check",
        "monitor",
        "watch",
        "wait",
        "progress",
        "state",
    })

    def __init__(
        self,
        context,
        ui_manager,
        runtime_config=None,
    ):
        self.context = context
        self.ui_manager = ui_manager
        self.tool_processor = ToolProcessor(context, ui_manager)
        # Store runtime_config for passing to streaming handler
        self.runtime_config = runtime_config
        # Tool state manager for caching and variable binding
        self._tool_state = get_tool_state()
        # Counter for consecutive duplicate detections (for escalation)
        self._consecutive_duplicate_count = 0
        self._max_consecutive_duplicates = 5  # Abort after this many
        # Runtime uses adaptive policy: strict core with smooth wrapper
        # No mode selection needed - always enforces grounding with auto-repair

    def _is_polling_tool(self, tool_name: str) -> bool:
        """Check if a tool is a polling/status tool that should be exempt from loop detection.

        Polling tools (like render_status, check_progress, etc.) are expected to be called
        repeatedly with the same arguments to monitor changing state. These should not
        trigger the duplicate call detection.
        """
        tool_lower = tool_name.lower()
        for pattern in self.POLLING_TOOL_PATTERNS:
            if pattern in tool_lower:
                return True
        return False

    async def process_conversation(self, max_turns: int = 100):
        """Process the conversation loop, handling tool calls and responses with streaming.

        Args:
            max_turns: Maximum number of conversation turns before forcing exit (default: 100)
        """
        turn_count = 0
        last_tool_signature = None  # Track last tool call to detect true duplicates
        tools_for_completion = None  # Will be set based on context

        # Reset tool state for this new prompt
        self._tool_state.reset_for_new_prompt()

        # Advance search engine turn for session boosting
        # Tools used recently get boosted in search results
        from chuk_tool_processor.discovery import get_search_engine

        search_engine = get_search_engine()
        search_engine.advance_turn()

        # Register user literals from the latest user message
        # This whitelists numbers from the user prompt so they pass ungrounded checks
        self._register_user_literals_from_history()

        try:
            while turn_count < max_turns:
                try:
                    turn_count += 1

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
                        role = (
                            msg.role if isinstance(msg, Message) else MessageRole.USER
                        )
                        content_preview = str(msg.content)[:100] if msg.content else ""
                        log.debug(
                            f"  Message {history_size - 3 + i}: role={role}, content_preview={content_preview}"
                        )

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

                    completion: CompletionResponse | None = None

                    if supports_streaming:
                        # Use streaming response handler
                        try:
                            completion = await self._handle_streaming_completion(
                                tools=tools_for_completion
                            )
                        except Exception as e:
                            log.warning(
                                f"Streaming failed, falling back to regular completion: {e}"
                            )
                            output.warning(
                                f"Streaming failed, falling back to regular completion: {e}"
                            )
                            completion = await self._handle_regular_completion(
                                tools=tools_for_completion
                            )
                    else:
                        # Regular completion
                        completion = await self._handle_regular_completion(
                            tools=tools_for_completion
                        )

                    # Use Pydantic model properties instead of dict.get()
                    response_content = completion.response or "No response"
                    tool_calls = completion.tool_calls
                    reasoning_content = completion.reasoning_content

                    # DEBUG: Log what we got from the model
                    log.info("=== COMPLETION RESULT ===")
                    log.info(
                        f"Response length: {len(response_content) if response_content else 0}"
                    )
                    log.info(
                        f"Tool calls count: {len(tool_calls) if tool_calls else 0}"
                    )
                    log.info(
                        f"Reasoning length: {len(reasoning_content) if reasoning_content else 0}"
                    )
                    if response_content and response_content != "No response":
                        log.info(f"Response preview: {response_content[:200]}")
                    if tool_calls:
                        for i, tc in enumerate(tool_calls):
                            # ToolCall is a Pydantic model from chuk_llm
                            log.info(
                                f"Tool call {i}: {tc.function.name} args={tc.function.arguments}"
                            )

                    # If model requested tool calls, execute them
                    if tool_calls and len(tool_calls) > 0:
                        log.debug(f"Processing {len(tool_calls)} tool calls from LLM")

                        # Check split budgets for each tool call type
                        # Get name mapping for looking up actual tool names
                        name_mapping = getattr(self.context, "tool_name_mapping", {})

                        # Check if any discovery tools would exceed budget
                        # Uses behavior-based classification (pattern matching + result shape)
                        discovery_tools_requested = []
                        execution_tools_requested = []

                        for tc in tool_calls:
                            tool_name = name_mapping.get(
                                tc.function.name, tc.function.name
                            )
                            if self._tool_state.is_discovery_tool(tool_name):
                                discovery_tools_requested.append(tool_name)
                            elif self._tool_state.is_execution_tool(tool_name):
                                execution_tools_requested.append(tool_name)

                        # Check discovery budget first
                        if discovery_tools_requested:
                            disc_status = self._tool_state.check_runaway(
                                discovery_tools_requested[0]
                            )
                            if disc_status.should_stop and "Discovery" in (
                                disc_status.reason or ""
                            ):
                                log.warning(
                                    f"Discovery budget exhausted: {disc_status.reason}"
                                )
                                output.warning(
                                    "⚠ Discovery budget exhausted - no more searching"
                                )

                                stop_msg = self._tool_state.format_discovery_exhausted_message()
                                self.context.inject_assistant_message(stop_msg)

                                if self.ui_manager.is_streaming_response:
                                    await self.ui_manager.stop_streaming_response()
                                if hasattr(self.ui_manager, "streaming_handler"):
                                    self.ui_manager.streaming_handler = None
                                continue

                        # Check execution budget
                        if execution_tools_requested:
                            exec_status = self._tool_state.check_runaway(
                                execution_tools_requested[0]
                            )
                            if exec_status.should_stop and "Execution" in (
                                exec_status.reason or ""
                            ):
                                log.warning(
                                    f"Execution budget exhausted: {exec_status.reason}"
                                )
                                output.warning(
                                    "⚠ Execution budget exhausted - no more tool calls"
                                )

                                stop_msg = self._tool_state.format_execution_exhausted_message()
                                self.context.inject_assistant_message(stop_msg)

                                if self.ui_manager.is_streaming_response:
                                    await self.ui_manager.stop_streaming_response()
                                if hasattr(self.ui_manager, "streaming_handler"):
                                    self.ui_manager.streaming_handler = None
                                continue

                        # Check general runaway status (combined budget, saturation, etc.)
                        runaway_status = self._tool_state.check_runaway()
                        if runaway_status.should_stop:
                            log.warning(f"Runaway detected: {runaway_status.reason}")
                            output.warning(f"⚠ {runaway_status.message}")

                            # Generate appropriate stop message
                            if runaway_status.budget_exhausted:
                                stop_msg = (
                                    self._tool_state.format_budget_exhausted_message()
                                )
                            elif runaway_status.saturation_detected:
                                last_val = (
                                    self._tool_state._recent_numeric_results[-1]
                                    if self._tool_state._recent_numeric_results
                                    else 0.0
                                )
                                stop_msg = self._tool_state.format_saturation_message(
                                    last_val
                                )
                            else:
                                stop_msg = (
                                    f"**Tool execution stopped**: {runaway_status.reason}\n\n"
                                    f"{self._tool_state.format_state_for_model()}\n\n"
                                    "Please provide your final answer using the computed values above."
                                )

                            # Inject stop message and continue without tools
                            self.context.inject_assistant_message(stop_msg)

                            # Stop streaming UI and continue to get final answer
                            if self.ui_manager.is_streaming_response:
                                await self.ui_manager.stop_streaming_response()
                            if hasattr(self.ui_manager, "streaming_handler"):
                                self.ui_manager.streaming_handler = None

                            # Continue to next iteration - model will see stop message
                            # and should provide final answer
                            continue

                        # Check if we're at max turns
                        if turn_count >= max_turns:
                            output.warning(
                                f"Maximum conversation turns ({max_turns}) reached. Stopping to prevent infinite loop."
                            )
                            self.context.inject_assistant_message(
                                "I've reached the maximum number of conversation turns. The tool results have been provided above."
                            )
                            # Stop streaming UI before breaking
                            if self.ui_manager.is_streaming_response:
                                await self.ui_manager.stop_streaming_response()
                            if hasattr(self.ui_manager, "streaming_handler"):
                                self.ui_manager.streaming_handler = None
                            break

                        # Create signature to detect duplicate tool calls
                        # ToolCall is a Pydantic model from chuk_llm with frozen function
                        current_signature = []
                        tool_names = []
                        for tc in tool_calls:
                            name = tc.function.name
                            args = tc.function.arguments  # JSON string from chuk_llm
                            current_signature.append(f"{name}:{args}")
                            tool_names.append(name)

                        current_sig_str = "|".join(sorted(current_signature))

                        # Check if ALL tools in this call are polling tools
                        # If so, exempt from duplicate detection
                        all_polling = all(self._is_polling_tool(n) for n in tool_names)

                        # Detect TRUE duplicates: same tool(s) with exact same args
                        # Different args = different computation, not stuck
                        # Polling tools are exempt - they're meant to be called repeatedly
                        is_true_duplicate: bool = bool(
                            last_tool_signature
                            and current_sig_str == last_tool_signature
                            and not all_polling
                        )

                        log.debug(
                            f"Duplicate check: sig={current_sig_str[:50]}, "
                            f"is_dup={is_true_duplicate}, all_polling={all_polling}"
                        )

                        if is_true_duplicate:
                            # True duplicate: same tool with same args
                            self._consecutive_duplicate_count += 1
                            log.warning(
                                f"Duplicate tool call detected ({self._consecutive_duplicate_count}x): {current_sig_str[:100]}"
                            )

                            # Check if we've exceeded max duplicates (safety valve)
                            if (
                                self._consecutive_duplicate_count
                                >= self._max_consecutive_duplicates
                            ):
                                output.warning(
                                    f"Model called exact same tool {self._consecutive_duplicate_count} times in a row.\n"
                                    "This indicates the model is stuck. Returning to prompt."
                                )
                                # CRITICAL: Stop streaming UI before breaking
                                if self.ui_manager.is_streaming_response:
                                    await self.ui_manager.stop_streaming_response()
                                if hasattr(self.ui_manager, "streaming_handler"):
                                    self.ui_manager.streaming_handler = None
                                break

                            # Inject state summary to help model use cached values
                            output.info(
                                "Detected repeated tool call. Using cached results and providing state summary."
                            )
                            state_summary = self._tool_state.format_state_for_model()
                            if state_summary:
                                state_msg = (
                                    "**Previously computed values (use these directly):**\n\n"
                                    f"{state_summary}\n\n"
                                    "Continue with the calculation using these stored values. "
                                    "Do not re-call tools for values already computed."
                                )
                                self.context.inject_assistant_message(state_msg)
                                log.info(
                                    f"Injected state summary: {state_summary[:200]}"
                                )

                            # Continue to next iteration - model will see the state
                            continue
                        else:
                            # Not a duplicate, reset counter
                            self._consecutive_duplicate_count = 0

                        last_tool_signature = current_sig_str

                        # Log the tool calls for debugging
                        for i, tc in enumerate(tool_calls):
                            log.debug(f"Tool call {i}: {tc}")

                        # FIXED: Get name mapping from universal tool compatibility system
                        name_mapping = getattr(self.context, "tool_name_mapping", {})
                        log.debug(f"Using name mapping: {name_mapping}")

                        # Process tool calls - this will handle streaming display
                        await self.tool_processor.process_tool_calls(
                            tool_calls,
                            name_mapping,
                            reasoning_content=reasoning_content,
                        )
                        continue

                    # Reset duplicate tracking on text response
                    last_tool_signature = None

                    # Display assistant response (if not already displayed by streaming)
                    elapsed = completion.elapsed_time

                    if not completion.streaming:
                        # Non-streaming response, display normally
                        await self.ui_manager.print_assistant_message(
                            response_content, elapsed
                        )
                    else:
                        # Streaming response - final display already handled by streaming_handler
                        # Just clean up
                        # NOTE: Don't call stop_streaming_response() here - it was already called
                        # by streaming_handler.stream_response()
                        # Clear streaming handler reference
                        if hasattr(self.ui_manager, "streaming_handler"):
                            self.ui_manager.streaming_handler = None

                    # Check for unused tool results (dataflow hygiene warning)
                    # NOTE: Disabled for cleaner demo output - models often compute
                    # analytically without referencing tool results explicitly
                    unused_warning = self._tool_state.format_unused_warning()
                    if unused_warning:
                        log.info("Unused tool results detected at end of turn")
                        # output.info(unused_warning)  # Disabled - too noisy for demos

                    # Extract and register any value bindings from assistant text
                    # This allows values like "σ_d = 5" to become referenceable via $vN
                    if response_content and response_content != "No response":
                        new_bindings = self._tool_state.extract_bindings_from_text(
                            response_content
                        )
                        if new_bindings:
                            log.info(
                                f"Extracted {len(new_bindings)} value bindings from assistant response"
                            )
                            for binding in new_bindings:
                                log.debug(
                                    f"  ${binding.id} = {binding.raw_value} (aliases: {binding.aliases})"
                                )

                    # Add to conversation history via SessionManager
                    # Include reasoning_content if present (for DeepSeek reasoner and similar models)
                    await self.context.add_assistant_message(response_content)
                    break

                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    output.error(f"Error during conversation processing: {exc}")
                    import traceback

                    traceback.print_exc()
                    self.context.inject_assistant_message(f"I encountered an error: {exc}")
                    # Stop streaming UI before breaking
                    if self.ui_manager.is_streaming_response:
                        await self.ui_manager.stop_streaming_response()
                    if hasattr(self.ui_manager, "streaming_handler"):
                        self.ui_manager.streaming_handler = None
                    break
        except asyncio.CancelledError:
            raise

    async def _handle_streaming_completion(
        self, tools: list | None = None
    ) -> CompletionResponse:
        """Handle streaming completion with UI integration.

        Args:
            tools: Tool definitions to pass to the LLM, or None to disable tools

        Returns:
            CompletionResponse with streaming metadata
        """
        from mcp_cli.chat.streaming_handler import StreamingResponseHandler

        # Signal UI that streaming is starting
        await self.ui_manager.start_streaming_response()

        # Set the streaming handler reference in UI manager for interruption support
        streaming_handler = StreamingResponseHandler(
            display=self.ui_manager.display, runtime_config=self.runtime_config
        )
        self.ui_manager.streaming_handler = streaming_handler

        try:
            # stream_response returns dict, convert to CompletionResponse
            completion_dict = await streaming_handler.stream_response(
                client=self.context.client,
                messages=[msg.to_dict() for msg in self.context.conversation_history],
                tools=tools,
            )

            # Convert dict to CompletionResponse Pydantic model
            completion = CompletionResponse.from_dict(completion_dict)

            # Enhanced tool call validation and logging
            if completion.tool_calls:
                log.debug(
                    f"Streaming completion returned {len(completion.tool_calls)} tool calls"
                )
                for i, tc in enumerate(completion.tool_calls):
                    log.debug(f"Streamed tool call {i}: {tc}")

            return completion

        finally:
            # Keep streaming handler reference for finalization
            # Will be cleared after finalization in main conversation loop
            pass

    async def _handle_regular_completion(
        self, tools: list | None = None
    ) -> CompletionResponse:
        """Handle regular (non-streaming) completion.

        Args:
            tools: Tool definitions to pass to the LLM, or None to disable tools

        Returns:
            CompletionResponse with timing metadata
        """
        start_time = time.time()

        try:
            messages_as_dicts = [
                msg.to_dict() for msg in self.context.conversation_history
            ]
            completion_dict = await self.context.client.create_completion(
                messages=messages_as_dicts,
                tools=tools,
            )
        except Exception as e:
            # If tools spec invalid, retry without tools
            err = str(e)
            if "Invalid 'tools" in err:
                log.error(f"Tool definition error: {err}")
                output.warning(
                    "Tool definitions rejected by model, retrying without tools..."
                )
                messages_as_dicts = [
                    msg.to_dict() for msg in self.context.conversation_history
                ]
                completion_dict = await self.context.client.create_completion(
                    messages=messages_as_dicts
                )
            else:
                raise

        elapsed = time.time() - start_time

        # Add timing and streaming metadata to the dict before converting to Pydantic
        completion_dict["elapsed_time"] = elapsed
        completion_dict["streaming"] = False

        # Convert to CompletionResponse Pydantic model
        return CompletionResponse.from_dict(completion_dict)

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

    def _register_user_literals_from_history(self) -> int:
        """Extract and register numeric literals from recent user messages.

        Scans conversation history for the most recent user message(s) and
        registers any numeric literals found. This whitelists user-provided
        numbers so they pass ungrounded call detection.

        Returns:
            Number of literals registered
        """
        total_registered = 0

        # Scan recent messages for user content
        for msg in reversed(self.context.conversation_history):
            if msg.role == MessageRole.USER and msg.content:
                count = self._tool_state.register_user_literals(msg.content)
                total_registered += count
                log.debug(f"Registered {count} user literals from message")
                # Only process the most recent user message
                break

        if total_registered > 0:
            log.info(
                f"Registered {total_registered} user literals for ungrounded check whitelist"
            )

        return total_registered
