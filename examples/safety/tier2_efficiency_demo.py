#!/usr/bin/env python3
"""Tier 2: Efficiency & Resilience Demo

Demonstrates all nine Tier 2 features that improve performance and
error handling in production workloads:

  2.1 Eliminate triple tool result storage
  2.2 Enforce procedural memory limits
  2.3 System prompt optimization (caching + tool summary)
  2.4 Stale connection recovery (error classification)
  2.5 Tool batch timeout
  2.6 Narrower exception handlers
  2.7 Provider validation
  2.8 LLM-visible context management notices
  2.9 Adaptive first-chunk timeout for thinking models

No API keys or MCP servers required — runs fully self-contained.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from the examples/ directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def demo_single_source_tool_history() -> None:
    """2.1 — Tool history reads from procedural memory, not redundant list."""
    print("=" * 70)
    print("2.1  SINGLE SOURCE OF TRUTH: TOOL HISTORY")
    print("=" * 70)

    from mcp_cli.config.defaults import DEFAULT_CONTEXT_NOTICES_ENABLED

    print(f"  Context notices enabled:  {DEFAULT_CONTEXT_NOTICES_ENABLED}")
    print("  Tool history source:      tool_memory.memory.tool_log")
    print("  Removed:                  chat_context.tool_history (was unbounded)")
    print("  Benefit:                  No more 3x storage of tool results")
    print()


def demo_system_prompt_optimization() -> None:
    """2.3 — System prompt caching + tool summary for large server sets."""
    print("=" * 70)
    print("2.3  SYSTEM PROMPT OPTIMIZATION")
    print("=" * 70)

    from mcp_cli.chat.system_prompt import _build_server_section
    from mcp_cli.config.defaults import (
        DEFAULT_SYSTEM_PROMPT_TOOL_SUMMARY_THRESHOLD,
        DEFAULT_SYSTEM_PROMPT_TOOL_PREVIEW_COUNT,
    )

    # Small server — all tools listed
    small_group = [
        {
            "name": "math-server",
            "description": "Math tools",
            "tools": ["add", "subtract", "multiply", "divide", "sqrt"],
        }
    ]
    small_section = _build_server_section(small_group)
    print(f"  Tool summary threshold:   {DEFAULT_SYSTEM_PROMPT_TOOL_SUMMARY_THRESHOLD}")
    print(f"  Tool preview count:       {DEFAULT_SYSTEM_PROMPT_TOOL_PREVIEW_COUNT}")
    print()
    print("  Small server (5 tools):   all listed")
    print(f"    {small_section.strip().split(chr(10))[-1]}")

    # Large server — tools summarized
    large_tools = [f"tool_{i}" for i in range(30)]
    large_group = [
        {
            "name": "mega-server",
            "description": "Many tools",
            "tools": large_tools,
        }
    ]
    large_section = _build_server_section(large_group)
    print("  Large server (30 tools):  summarized")
    print(f"    {large_section.strip().split(chr(10))[-1]}")

    # Demonstrate dirty flag caching
    print()
    print("  System prompt caching:    dirty flag pattern")
    print("    - _system_prompt_dirty=True  → regenerate")
    print("    - _system_prompt_dirty=False → return cached")
    print("    - Set True on: tool change, model change")
    print()


def demo_batch_timeout() -> None:
    """2.5 — Batch timeout prevents one hung tool from blocking everything."""
    print("=" * 70)
    print("2.5  TOOL BATCH TIMEOUT")
    print("=" * 70)

    from mcp_cli.config.defaults import (
        DEFAULT_BATCH_TIMEOUT_MULTIPLIER,
        DEFAULT_BATCH_TIMEOUT_FLOOR,
        DEFAULT_TOOL_EXECUTION_TIMEOUT,
    )

    per_tool = DEFAULT_TOOL_EXECUTION_TIMEOUT
    computed = max(
        per_tool * DEFAULT_BATCH_TIMEOUT_MULTIPLIER,
        DEFAULT_BATCH_TIMEOUT_FLOOR,
    )

    print(f"  Per-tool timeout:         {per_tool}s")
    print(f"  Batch multiplier:         {DEFAULT_BATCH_TIMEOUT_MULTIPLIER}x")
    print(f"  Batch floor:              {DEFAULT_BATCH_TIMEOUT_FLOOR}s")
    print(f"  Computed batch timeout:   {computed}s")
    print()
    print("  Behavior:")
    print("    - Fast tools return immediately")
    print("    - Slow tools cancelled when batch timeout fires")
    print("    - Partial results returned for completed tools")
    print()


def demo_narrower_exceptions() -> None:
    """2.6 — Specific exception handlers replace broad 'except Exception'."""
    print("=" * 70)
    print("2.6  NARROWER EXCEPTION HANDLERS")
    print("=" * 70)

    print("  Conversation loop now catches specifically:")
    print()
    print("    asyncio.CancelledError       → re-raised (not swallowed)")
    print("    asyncio.TimeoutError          → 'Request timed out' + assistant msg")
    print("    ConnectionError / OSError     → 'Connection error' + assistant msg")
    print("    ValueError / TypeError        → 'Configuration error' (no AI msg)")
    print("    Exception (fallback)          → 'Unexpected error' + assistant msg")
    print()
    print("  Benefit: each error type gets appropriate logging level,")
    print("  user-facing message, and recovery strategy.")
    print()


def demo_connection_error_detection() -> None:
    """2.4 — Connection error classification in ToolManager."""
    print("=" * 70)
    print("2.4  CONNECTION ERROR CLASSIFICATION")
    print("=" * 70)

    from mcp_cli.tools.manager import ToolManager

    tm = ToolManager(config_file="demo.json", servers=[])

    test_errors = [
        "Connection refused",
        "Transport not initialized",
        "Broken pipe while sending",
        "Connection reset by peer",
        "Server timed out",
        "File not found",  # Not a connection error
        "Invalid arguments",  # Not a connection error
    ]

    print("  Error message                    → Connection error?")
    print("  " + "-" * 50)
    for msg in test_errors:
        is_conn = tm._is_connection_error(msg)
        marker = "YES" if is_conn else "no"
        print(f"  {msg:<35} → {marker}")
    print()


def demo_context_notices() -> None:
    """2.8 — LLM-visible context management notices."""
    print("=" * 70)
    print("2.8  LLM-VISIBLE CONTEXT MANAGEMENT NOTICES")
    print("=" * 70)

    from mcp_cli.chat.conversation import ConversationProcessor
    from mcp_cli.chat.response_models import Message, MessageRole

    # Create a mock context with pending notices
    class DemoContext:
        def __init__(self):
            self._pending_context_notices = []

        def add_context_notice(self, notice):
            self._pending_context_notices.append(notice)

        def drain_context_notices(self):
            notices = self._pending_context_notices[:]
            self._pending_context_notices.clear()
            return notices

    ctx = DemoContext()
    ctx.add_context_notice(
        "Tool result from 'search_data' was truncated from 200,000 to 100,000 chars. "
        "Consider requesting less data."
    )
    ctx.add_context_notice(
        "5 older messages were evicted from context. "
        "Key context may need to be re-established."
    )

    print(f"  Pending notices:          {len(ctx._pending_context_notices)}")

    # Simulate _prepare_messages_for_api
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are helpful."),
        Message(role=MessageRole.USER, content="Search for data."),
    ]
    result = ConversationProcessor._prepare_messages_for_api(messages, context=ctx)

    print(f"  Messages before:          {len(messages)}")
    print(f"  Messages after:           {len(result)}")
    print("  Notice injected at:       index 1 (after system prompt)")
    print(f"  Notice role:              {result[1]['role']}")
    print()
    print("  Notice content:")
    for line in result[1]["content"].split("\n"):
        print(f"    {line}")
    print()
    print(f"  Notices drained:          {len(ctx._pending_context_notices)} remaining")
    print()


def demo_provider_validation() -> None:
    """2.7 — Provider validation at startup."""
    print("=" * 70)
    print("2.7  PROVIDER VALIDATION AT STARTUP")
    print("=" * 70)

    print("  ChatContext.initialize() now validates the provider:")
    print()
    print("    1. Attempts to create client (self.client)")
    print("    2. Fails fast if API key is missing/invalid")
    print("    3. Logs warning but does NOT block chat startup")
    print("    4. User sees clear message before typing anything")
    print()
    print("  Example output on invalid key:")
    print("    [yellow]Provider validation warning: No API key[/yellow]")
    print("    [yellow]Chat may fail when making API calls.[/yellow]")
    print()


def demo_memory_limits() -> None:
    """2.2 — Procedural memory pattern limits enforced."""
    print("=" * 70)
    print("2.2  PROCEDURAL MEMORY LIMITS")
    print("=" * 70)

    print("  ChatContext._enforce_memory_limits() runs after each tool call:")
    print()
    print("    - Reads max_patterns_per_tool from ToolMemoryManager")
    print("    - Trims error_patterns and success_patterns to limit")
    print("    - Keeps most recent patterns (LRU eviction)")
    print()
    print("  Prevents unbounded growth of:")
    print("    - tool_patterns.error_patterns")
    print("    - tool_patterns.success_patterns")
    print()


def demo_first_chunk_timeout() -> None:
    """Adaptive first-chunk timeout for thinking models after tool calls."""
    print("=" * 70)
    print("2.9  ADAPTIVE FIRST-CHUNK TIMEOUT")
    print("=" * 70)

    from mcp_cli.config.defaults import (
        DEFAULT_STREAMING_CHUNK_TIMEOUT,
        DEFAULT_STREAMING_FIRST_CHUNK_TIMEOUT,
        DEFAULT_STREAMING_FIRST_CHUNK_AFTER_TOOLS_TIMEOUT,
    )

    print(f"  Chunk timeout (per-chunk):            {DEFAULT_STREAMING_CHUNK_TIMEOUT}s")
    print(
        f"  First chunk timeout (normal):         {DEFAULT_STREAMING_FIRST_CHUNK_TIMEOUT}s"
    )
    print(
        f"  First chunk timeout (after tools):    {DEFAULT_STREAMING_FIRST_CHUNK_AFTER_TOOLS_TIMEOUT}s"
    )
    print()
    print("  Problem solved:")
    print("    Thinking models (kimi-k2.5, DeepSeek-R1) need extended")
    print("    processing time after receiving tool results before emitting")
    print("    their first response token. The standard 45s chunk timeout")
    print("    would fire prematurely, killing a valid response.")
    print()
    print("  Three-tier timeout strategy:")
    print("    1. First chunk (no tools):  60s — initial model warm-up")
    print("    2. First chunk (after tools): 180s — thinking model processing")
    print("    3. Subsequent chunks:       45s — normal streaming cadence")
    print()
    print("  Configurable via:")
    print("    MCP_STREAMING_FIRST_CHUNK_TIMEOUT=120")
    print("    timeouts.streamingFirstChunk = 120")
    print()


def demo_defaults_overview() -> None:
    """Show all Tier 2 defaults from config/defaults.py."""
    print("=" * 70)
    print("TIER 2 CONFIGURATION DEFAULTS")
    print("=" * 70)

    from mcp_cli.config.defaults import (
        DEFAULT_BATCH_TIMEOUT_MULTIPLIER,
        DEFAULT_BATCH_TIMEOUT_FLOOR,
        DEFAULT_SYSTEM_PROMPT_TOOL_SUMMARY_THRESHOLD,
        DEFAULT_SYSTEM_PROMPT_TOOL_PREVIEW_COUNT,
        DEFAULT_RECONNECT_ON_FAILURE,
        DEFAULT_MAX_RECONNECT_ATTEMPTS,
        DEFAULT_CONTEXT_NOTICES_ENABLED,
        DEFAULT_MAX_CONSECUTIVE_DUPLICATES,
        DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES,
        DEFAULT_STREAMING_FIRST_CHUNK_AFTER_TOOLS_TIMEOUT,
        DYNAMIC_TOOL_PROXY_NAME,
    )

    defaults = {
        "Batch timeout multiplier": DEFAULT_BATCH_TIMEOUT_MULTIPLIER,
        "Batch timeout floor": f"{DEFAULT_BATCH_TIMEOUT_FLOOR}s",
        "Tool summary threshold": DEFAULT_SYSTEM_PROMPT_TOOL_SUMMARY_THRESHOLD,
        "Tool preview count": DEFAULT_SYSTEM_PROMPT_TOOL_PREVIEW_COUNT,
        "Reconnect on failure": DEFAULT_RECONNECT_ON_FAILURE,
        "Max reconnect attempts": DEFAULT_MAX_RECONNECT_ATTEMPTS,
        "Context notices enabled": DEFAULT_CONTEXT_NOTICES_ENABLED,
        "Max consecutive duplicates": DEFAULT_MAX_CONSECUTIVE_DUPLICATES,
        "Max transport failures": DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES,
        "First chunk after tools": f"{DEFAULT_STREAMING_FIRST_CHUNK_AFTER_TOOLS_TIMEOUT}s",
        "Dynamic tool proxy name": DYNAMIC_TOOL_PROXY_NAME,
    }

    max_key_len = max(len(k) for k in defaults)
    for key, value in defaults.items():
        print(f"  {key:<{max_key_len}}  {value}")
    print()


def main() -> None:
    """Run all Tier 2 demos."""
    print()
    print("  MCP-CLI Tier 2: Efficiency & Resilience")
    print("  These features improve performance and error handling")
    print()

    demo_single_source_tool_history()
    demo_system_prompt_optimization()
    demo_batch_timeout()
    demo_narrower_exceptions()
    demo_connection_error_detection()
    demo_context_notices()
    demo_provider_validation()
    demo_memory_limits()
    demo_first_chunk_timeout()
    demo_defaults_overview()

    print("=" * 70)
    print("  All Tier 2 demos completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
