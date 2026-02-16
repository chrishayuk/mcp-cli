#!/usr/bin/env python3
"""Tier 1: Context Safety Mechanisms Demo

Demonstrates all five Tier 1 foundation features that prevent crashes
from large payloads, unbounded accumulation, and context overflow:

  1.1 Tool result truncation
  1.2 Reasoning content stripping
  1.3 Conversation history sliding window
  1.4 Infinite context configuration
  1.5 Streaming buffer caps

No API keys or MCP servers required — runs fully self-contained.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Allow running from the examples/ directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def demo_tool_result_truncation() -> None:
    """1.1 — Truncate large tool results before they enter conversation history."""
    from mcp_cli.config.defaults import DEFAULT_MAX_TOOL_RESULT_CHARS

    print("=" * 70)
    print("1.1  TOOL RESULT TRUNCATION")
    print("=" * 70)

    # Simulate a massive API response (e.g., maritime data)
    huge_result = '{"data": "' + "X" * 200_000 + '"}'
    print(f"  Original result size:  {len(huge_result):>10,} chars")
    print(f"  Configured max:        {DEFAULT_MAX_TOOL_RESULT_CHARS:>10,} chars")

    # Import and use the truncation method
    from unittest.mock import MagicMock
    from mcp_cli.chat.tool_processor import ToolProcessor

    ctx = MagicMock()
    ui = MagicMock()
    processor = ToolProcessor(ctx, ui)

    truncated = processor._truncate_tool_result(
        huge_result, DEFAULT_MAX_TOOL_RESULT_CHARS
    )
    print(f"  After truncation:      {len(truncated):>10,} chars")
    print(f"  Contains notice:       {'TRUNCATED' in truncated}")
    print(f"  Head preserved:        {truncated[:20]!r}...")
    print(f"  Tail preserved:        ...{truncated[-20:]!r}")

    # Show it's a no-op for small results
    small = '{"status": "ok"}'
    same = processor._truncate_tool_result(small, DEFAULT_MAX_TOOL_RESULT_CHARS)
    print(f"  Small result unchanged: {same == small}")
    print()


def demo_reasoning_stripping() -> None:
    """1.2 — Strip old reasoning content before API calls."""
    print("=" * 70)
    print("1.2  REASONING CONTENT STRIPPING")
    print("=" * 70)

    from mcp_cli.chat.conversation import ConversationProcessor

    # Simulate a conversation with thinking models
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": "4",
            "reasoning_content": "Let me analyze this... " * 5000,  # ~100K chars
        },
        {"role": "user", "content": "And 3+3?"},
        {
            "role": "assistant",
            "content": "6",
            "reasoning_content": "Simple arithmetic... " * 100,  # ~2K chars
        },
    ]

    total_reasoning_before = sum(
        len(m.get("reasoning_content", ""))
        for m in messages
        if m.get("role") == "assistant"
    )
    print(f"  Messages:              {len(messages)}")
    print(f"  Total reasoning chars: {total_reasoning_before:,}")

    stripped = ConversationProcessor._strip_old_reasoning_content(messages)

    total_reasoning_after = sum(
        len(m.get("reasoning_content", ""))
        for m in stripped
        if m.get("role") == "assistant"
    )
    print(f"  After stripping:       {total_reasoning_after:,}")
    print(
        f"  Savings:               {total_reasoning_before - total_reasoning_after:,} chars"
    )
    print(f"  Latest reasoning kept: {'reasoning_content' in stripped[-1]}")
    print(f"  Old reasoning removed: {'reasoning_content' not in stripped[1]}")
    print()


async def demo_sliding_window() -> None:
    """1.3 — Conversation history sliding window."""
    print("=" * 70)
    print("1.3  CONVERSATION HISTORY SLIDING WINDOW")
    print("=" * 70)

    from unittest.mock import Mock
    from mcp_cli.chat.chat_context import ChatContext
    from mcp_cli.model_management import ModelManager

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "demo"
    mock_manager.get_active_model.return_value = "demo-model"

    mock_tm = Mock()

    # Create context with a sliding window of 5 messages
    ctx = ChatContext(
        tool_manager=mock_tm,
        model_manager=mock_manager,
        max_history_messages=5,
    )
    ctx._system_prompt = "You are a helpful assistant."
    await ctx._initialize_session()

    # Add 20 messages
    for i in range(20):
        await ctx.add_user_message(f"Message {i}")

    history = ctx.conversation_history
    print("  Messages added:        20")
    print("  Window size:           5")
    print(f"  History returned:      {len(history)} (1 system + 5 windowed)")
    print(f"  System prompt kept:    {history[0].content[:30]}...")
    print(f"  Oldest in window:      {history[1].content}")
    print(f"  Newest in window:      {history[-1].content}")
    print()


async def demo_infinite_context_config() -> None:
    """1.4 — Infinite context mode configuration."""
    print("=" * 70)
    print("1.4  INFINITE CONTEXT CONFIGURATION")
    print("=" * 70)

    from unittest.mock import Mock
    from mcp_cli.chat.chat_context import ChatContext
    from mcp_cli.model_management import ModelManager

    mock_manager = Mock(spec=ModelManager)
    mock_manager.get_client.return_value = None
    mock_manager.get_active_provider.return_value = "demo"
    mock_manager.get_active_model.return_value = "demo-model"

    mock_tm = Mock()

    # Default: infinite_context=False
    ctx_default = ChatContext(
        tool_manager=mock_tm,
        model_manager=mock_manager,
    )
    print(f"  Default infinite_context:   {ctx_default._infinite_context}")

    # Configured: infinite_context=True
    ctx_enabled = ChatContext(
        tool_manager=mock_tm,
        model_manager=mock_manager,
        infinite_context=True,
        token_threshold=8000,
        max_turns_per_segment=30,
    )
    print(f"  Enabled infinite_context:   {ctx_enabled._infinite_context}")
    print(f"  Token threshold:            {ctx_enabled._token_threshold}")
    print(f"  Max turns per segment:      {ctx_enabled._max_turns_per_segment}")

    # Verify it gets passed to SessionManager
    ctx_enabled._system_prompt = "SYS"
    await ctx_enabled._initialize_session()
    print(f"  SessionManager created:     {ctx_enabled.session is not None}")
    print()


def demo_streaming_buffer_caps() -> None:
    """1.5 — Streaming buffer caps prevent OOM from huge responses."""
    print("=" * 70)
    print("1.5  STREAMING BUFFER CAPS")
    print("=" * 70)

    from mcp_cli.display.models import StreamingState, StreamingChunk

    # Create a state with a small cap for demo purposes
    state = StreamingState(max_accumulated_chars=500, max_chunks=10)
    print(f"  Max content chars:     {state.max_accumulated_chars}")
    print(f"  Max chunks:            {state.max_chunks}")

    # Simulate streaming chunks
    for i in range(15):
        chunk = StreamingChunk(content=f"Chunk-{i:02d} " + "x" * 50)
        state.add_chunk(chunk)

    print(f"  Chunks received:       {state.chunks_received}")
    print(f"  Content capped:        {state.content_capped}")
    print(f"  Content length:        {state.content_length} chars")
    print(f"  Finish reason:         {state.finish_reason}")

    # Show the truncation notice in content
    if state.content_capped:
        # Find the truncation notice
        content = state.accumulated_content
        if "buffer limit" in content:
            idx = content.index("[Content truncated")
            print(f"  Truncation notice at:  char {idx}")

    # Show default (1MB) caps
    default_state = StreamingState()
    print(f"\n  Default max chars:     {default_state.max_accumulated_chars:,} (1 MB)")
    print(f"  Default max chunks:    {default_state.max_chunks:,}")
    print()


async def main() -> None:
    """Run all Tier 1 demos."""
    print()
    print("  MCP-CLI Tier 1: Context Safety Mechanisms")
    print("  These features prevent crashes from large payloads")
    print()

    demo_tool_result_truncation()
    demo_reasoning_stripping()
    await demo_sliding_window()
    await demo_infinite_context_config()
    demo_streaming_buffer_caps()

    print("=" * 70)
    print("  All Tier 1 demos completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
