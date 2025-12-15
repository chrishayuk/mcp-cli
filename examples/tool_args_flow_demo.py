#!/usr/bin/env python3
"""Demo script to trace tool argument flow.

This shows how arguments flow from model response to tool execution,
helping identify where values might get corrupted or changed.
"""

import json
import logging
import sys

sys.path.insert(0, "src")

from mcp_cli.chat.streaming_handler import ToolCallAccumulator  # noqa: E402

# Enable debug logging to see all traces
logging.basicConfig(level=logging.DEBUG, format="%(name)s - %(message)s")


def demo_accumulator():
    """Demo the tool call accumulator to see how args are merged."""
    print("=" * 60)
    print("TOOL CALL ACCUMULATOR DEMO - Normal streaming")
    print("=" * 60)

    acc = ToolCallAccumulator()

    # Simulate streaming chunks from model
    # Chunk 1: Tool name and start of args
    chunk1 = [
        {
            "id": "call_123",
            "index": 0,
            "function": {
                "name": "call_tool",
                "arguments": '{"tool_name": "normal_cdf"',
            },
        }
    ]

    # Chunk 2: More args
    chunk2 = [
        {
            "id": "call_123",
            "index": 0,
            "function": {"name": "", "arguments": ', "x": 11.03}'},
        }
    ]

    print("\n1. CHUNK 1:")
    print(f"   {json.dumps(chunk1, indent=2)}")
    acc.process_chunk_tool_calls(chunk1)

    print("\n2. CHUNK 2:")
    print(f"   {json.dumps(chunk2, indent=2)}")
    acc.process_chunk_tool_calls(chunk2)

    print("\n3. ACCUMULATED RESULT:")
    result = acc.finalize()
    for tc in result:
        print(f"   Tool: {tc['function']['name']}")
        print(f"   Args: {tc['function']['arguments']}")
        # Parse to show final values
        try:
            parsed = json.loads(tc["function"]["arguments"])
            print(f"   Parsed: {json.dumps(parsed, indent=6)}")
        except json.JSONDecodeError as e:
            print(f"   Parse error: {e}")


def demo_complete_then_partial():
    """Demo what happens when model sends complete JSON then partial update."""
    print("\n" + "=" * 60)
    print("COMPLETE THEN PARTIAL DEMO - Same ID scenario")
    print("=" * 60)
    print("Same call_id means chunks should merge (preserving first values)")

    acc = ToolCallAccumulator()

    # Same ID - these should merge
    chunk1 = [
        {
            "id": "call_456",
            "index": 0,
            "function": {
                "name": "call_tool",
                "arguments": '{"tool_name": "normal_cdf", "x": 9.067}',
            },
        }
    ]

    chunk2 = [
        {
            "id": "call_456",  # SAME ID - will merge
            "index": 0,
            "function": {
                "name": "",
                "arguments": '{"x": 4.617}',  # Different x value - should be ignored
            },
        }
    ]

    print("\n1. CHUNK 1 (id=call_456, x=9.067):")
    acc.process_chunk_tool_calls(chunk1)

    print("\n2. CHUNK 2 (id=call_456, x=4.617) - same ID, should merge:")
    acc.process_chunk_tool_calls(chunk2)

    print("\n3. RESULT (should have x=9.067 - first value preserved):")
    result = acc.finalize()
    for tc in result:
        parsed = json.loads(tc["function"]["arguments"])
        print(f"   x = {parsed.get('x')}")
        if parsed.get("x") == 9.067:
            print("   ✓ CORRECT: First value preserved")
        else:
            print("   ✗ WRONG: Expected 9.067")


def demo_different_ids():
    """Demo that different IDs create separate tool calls."""
    print("\n" + "=" * 60)
    print("DIFFERENT IDs DEMO - Should NOT merge")
    print("=" * 60)
    print("Different call_ids mean separate tool calls")

    acc = ToolCallAccumulator()

    # Different IDs - should NOT merge (this was the bug!)
    chunk1 = [
        {
            "id": "call_AAA",
            "index": 0,
            "function": {
                "name": "call_tool",
                "arguments": '{"tool_name": "normal_cdf", "x": 9.067}',
            },
        }
    ]

    chunk2 = [
        {
            "id": "call_BBB",  # DIFFERENT ID - should NOT merge!
            "index": 0,  # Same index - old bug would merge these!
            "function": {
                "name": "call_tool",
                "arguments": '{"tool_name": "normal_cdf", "x": 4.617}',
            },
        }
    ]

    print("\n1. CHUNK 1 (id=call_AAA, x=9.067):")
    acc.process_chunk_tool_calls(chunk1)

    print("\n2. CHUNK 2 (id=call_BBB, x=4.617) - DIFFERENT ID:")
    acc.process_chunk_tool_calls(chunk2)

    print("\n3. RESULT (should be TWO separate tool calls):")
    result = acc.finalize()
    print(f"   Number of tool calls: {len(result)}")
    for i, tc in enumerate(result):
        parsed = json.loads(tc["function"]["arguments"])
        print(f"   Tool call {i + 1}: id={tc['id']}, x={parsed.get('x')}")

    if len(result) == 2:
        print("   ✓ CORRECT: Two separate tool calls")
    else:
        print("   ✗ WRONG: Should have 2 tool calls, got", len(result))


def demo_json_merge():
    """Demo JSON string merging behavior."""
    print("\n" + "=" * 60)
    print("JSON MERGE BEHAVIOR DEMO")
    print("=" * 60)

    acc = ToolCallAccumulator()

    test_cases = [
        # Case 1: Simple concatenation
        ('{"x": 11', ".03}", "Simple concat"),
        # Case 2: Two complete objects - FIRST should win now
        ('{"x": 1}', '{"x": 11.03}', "Two complete objects (1st wins)"),
        # Case 3: Overlapping keys - FIRST should win now
        ('{"x": 1, "mean": 0}', '{"x": 11.03}', "Overlapping keys (1st wins)"),
        # Case 4: First correct value preserved
        ('{"x": 11.03}', '{"x": 1}', "First value preserved"),
        # Case 5: First has correct value, second corrupted - first wins
        ('{"tool_name": "cdf", "x": 11.03}', '{"x": 2.546}', "Correct preserved"),
    ]

    for i, (str1, str2, desc) in enumerate(test_cases, 1):
        print(f"\n{i}. {desc}")
        print(f"   String 1: {str1}")
        print(f"   String 2: {str2}")
        result = acc._merge_json_strings(str1, str2)
        print(f"   Merged:   {result}")
        try:
            parsed = json.loads(result)
            print(f"   Parsed:   {parsed}")
        except json.JSONDecodeError as e:
            print(f"   Parse error: {e}")


def demo_argument_parsing():
    """Demo how arguments are parsed in tool_processor."""
    print("\n" + "=" * 60)
    print("ARGUMENT PARSING DEMO")
    print("=" * 60)

    # Simulate different argument formats the model might send
    test_args = [
        '{"tool_name": "normal_cdf", "x": 11.03}',
        '{"tool_name": "normal_cdf", "x": 11.03, "mean": 0, "std": 1}',
        '{"tool_name": "normal_cdf", "x": null}',
        '{"tool_name": "normal_cdf"}',  # Missing x
        "invalid json",
    ]

    for i, raw in enumerate(test_args, 1):
        print(f"\n{i}. Raw: {raw}")
        try:
            if isinstance(raw, str):
                parsed = json.loads(raw) if raw.strip() else {}
            else:
                parsed = raw or {}
            print(f"   Parsed: {parsed}")

            # Filter for display (like tool_processor does)
            display = {k: v for k, v in parsed.items() if k != "tool_name"}
            print(f"   Display args: {display}")

            # Check for None
            none_args = [k for k, v in parsed.items() if v is None]
            if none_args:
                print(f"   WARNING: None values for: {none_args}")

        except json.JSONDecodeError as e:
            print(f"   Parse error: {e}")


def demo_precondition_gate():
    """Demo the precondition gate that blocks premature tool calls."""
    print("\n" + "=" * 60)
    print("PRECONDITION GATE DEMO")
    print("=" * 60)
    print("Parameterized tools are blocked when no values exist in state")

    from mcp_cli.chat.tool_state import get_tool_state, reset_tool_state

    # Reset state for clean test
    reset_tool_state()
    state = get_tool_state()

    print("\n--- Scenario 1: Premature call (no values in state) ---")
    # Model tries to call normal_cdf before computing any values
    args_premature = {"x": -0.751501502}  # Garbage value from GPT-5.2
    allowed, error = state.check_tool_preconditions("normal_cdf", args_premature)
    print("   Tool: normal_cdf")
    print(f"   Args: {args_premature}")
    print(f"   Values in state: {len(state.bindings.bindings)}")
    print(f"   Allowed: {allowed}")
    if error:
        print(f"   Error: {error[:100]}...")
    if not allowed:
        print("   ✓ CORRECT: Premature call blocked")
    else:
        print("   ✗ WRONG: Should have been blocked")

    print("\n--- Scenario 2: After computing a value ---")
    # Simulate computing Z and storing it
    state.bind_value("sqrt", {"x": 666}, 25.807)  # sqrt(666) for sigma_LT
    state.bind_value("divide", {"a": 234, "b": 25.807}, 9.067)  # Z = 234/25.807

    print(f"   Computed values in state: {len(state.bindings.bindings)}")
    for vid, binding in state.bindings.bindings.items():
        print(f"     ${vid} = {binding.typed_value} (from {binding.tool_name})")

    # Now try calling normal_cdf with the computed Z
    args_valid = {"x": 9.067}
    allowed, error = state.check_tool_preconditions("normal_cdf", args_valid)
    print("\n   Tool: normal_cdf")
    print(f"   Args: {args_valid}")
    print(f"   Allowed: {allowed}")
    if allowed:
        print("   ✓ CORRECT: Call allowed after values computed")
    else:
        print(f"   ✗ WRONG: Should have been allowed. Error: {error}")

    print("\n--- Scenario 3: Discovery tools are NOT gated ---")
    reset_tool_state()
    state = get_tool_state()

    # Discovery tools like search_tools should always be allowed
    args_search = {"query": "cdf"}
    allowed, error = state.check_tool_preconditions("search_tools", args_search)
    print("   Tool: search_tools")
    print(f"   Args: {args_search}")
    print(f"   Allowed: {allowed}")
    if allowed:
        print("   ✓ CORRECT: Discovery tool not gated")
    else:
        print("   ✗ WRONG: Discovery tools should not be gated")


def demo_model_sends_garbage():
    """Demo proving the model sends garbage x values."""
    print("\n" + "=" * 60)
    print("MODEL GARBAGE DETECTION DEMO")
    print("=" * 60)
    print("Shows how we detect when model sends wrong x values")

    from mcp_cli.chat.tool_state import reset_tool_state, get_tool_state

    reset_tool_state()
    state = get_tool_state()

    # These are actual values GPT-5.2 sent (from debug logs)
    garbage_calls = [
        {
            "x": -0.751501502,
            "expected_z": 9.067,
            "context": "First prompt, Poisson assumption",
        },
        {"x": 1.051146509, "expected_z": 11.03, "context": "Second prompt, σ=5"},
        {"x": 3.177643716, "expected_z": 9.067, "context": "Another run"},
    ]

    print("\nGPT-5.2 sent these x values to normal_cdf:")
    for i, call in enumerate(garbage_calls, 1):
        print(f"\n{i}. {call['context']}")
        print(f"   Model computed Z ≈ {call['expected_z']} in text")
        print(f"   Model sent x = {call['x']} to tool")
        print(f"   Difference: {abs(call['x'] - call['expected_z']):.2f}")

        # Check if precondition would block it
        allowed, error = state.check_tool_preconditions("normal_cdf", {"x": call["x"]})
        if not allowed:
            print("   ✓ BLOCKED by precondition gate")
        else:
            print("   Would be allowed (values exist in state)")

    print("\n" + "-" * 60)
    print("CONCLUSION: The model computes Z correctly in text but sends")
    print("garbage values in the tool call. The precondition gate blocks")
    print("these premature calls until actual values are computed.")


def main():
    print("TOOL ARGUMENT FLOW INVESTIGATION")
    print("=" * 60)
    print("This demo helps identify where tool arguments might")
    print("get corrupted between model output and execution.")
    print("=" * 60)

    demo_accumulator()
    demo_complete_then_partial()
    demo_different_ids()
    demo_json_merge()
    demo_argument_parsing()
    demo_precondition_gate()
    demo_model_sends_garbage()

    print("\n" + "=" * 60)
    print("INVESTIGATION COMPLETE")
    print("=" * 60)
    print("\nKey findings:")
    print("1. JSON merge preserves first values (fixed)")
    print("2. Different call IDs create separate tool calls (fixed)")
    print("3. Model sends garbage x values before computing Z")
    print("4. Precondition gate blocks premature parameterized tool calls")


if __name__ == "__main__":
    main()
