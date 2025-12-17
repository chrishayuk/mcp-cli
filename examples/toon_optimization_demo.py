#!/usr/bin/env python3
"""
Demo script showcasing TOON optimization feature.

This script demonstrates how TOON (Token-Optimized Object Notation) reduces
token costs by comparing JSON and TOON formats for various conversation scenarios.
"""

import json
from mcp_cli.llm.toon_optimizer import (
    ToonOptimizer,
    format_token_comparison,
    get_format_decision_message,
)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_simple_conversation():
    """Demonstrate TOON optimization for a simple conversation."""
    print_section("Simple Conversation")

    optimizer = ToonOptimizer(enabled=True)
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"},
        {"role": "user", "content": "Can you explain what TOON optimization is?"},
    ]

    print("\nOriginal Messages (JSON):")
    print(json.dumps({"messages": messages}, indent=2))

    toon_str = optimizer.convert_to_toon(messages)
    print("\nTOON Format:")
    print(toon_str)

    comparison = optimizer.compare_formats(messages)
    print("\n" + format_token_comparison(comparison))
    print(get_format_decision_message(comparison))


def demo_conversation_with_tools():
    """Demonstrate TOON optimization with tool definitions."""
    print_section("Conversation with Tool Definitions")

    optimizer = ToonOptimizer(enabled=True)
    messages = [
        {"role": "user", "content": "What's the weather like in San Francisco?"},
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    print("\nOriginal Format (JSON):")
    print(json.dumps({"messages": messages, "tools": tools}, indent=2))

    toon_str = optimizer.convert_to_toon(messages, tools)
    print("\nTOON Format:")
    print(toon_str)

    comparison = optimizer.compare_formats(messages, tools)
    print("\n" + format_token_comparison(comparison))
    print(get_format_decision_message(comparison))


def demo_conversation_with_tool_calls():
    """Demonstrate TOON optimization with tool calls."""
    print_section("Conversation with Tool Calls")

    optimizer = ToonOptimizer(enabled=True)
    messages = [
        {"role": "user", "content": "What's the weather in New York?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "New York, NY", "unit": "fahrenheit"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "name": "get_weather",
            "content": '{"temperature": 72, "condition": "sunny", "humidity": 45}',
        },
        {
            "role": "assistant",
            "content": "The weather in New York is currently 72°F and sunny with 45% humidity.",
        },
    ]

    print("\nOriginal Messages (JSON):")
    print(json.dumps({"messages": messages}, indent=2))

    toon_str = optimizer.convert_to_toon(messages)
    print("\nTOON Format:")
    print(toon_str)

    comparison = optimizer.compare_formats(messages)
    print("\n" + format_token_comparison(comparison))
    print(get_format_decision_message(comparison))


def demo_long_conversation():
    """Demonstrate TOON optimization benefits for long conversations."""
    print_section("Long Conversation (Maximum Token Savings)")

    optimizer = ToonOptimizer(enabled=True)

    # Simulate a long conversation
    messages = []
    for i in range(10):
        messages.append({
            "role": "user",
            "content": f"This is user message number {i+1} in a longer conversation.",
        })
        messages.append({
            "role": "assistant",
            "content": f"This is assistant response number {i+1} providing helpful information.",
        })

    json_str = json.dumps({"messages": messages}, indent=2)
    toon_str = optimizer.convert_to_toon(messages)

    print(f"\nConversation with {len(messages)} messages")
    print(f"JSON format length: {len(json_str):,} characters")
    print(f"TOON format length: {len(toon_str):,} characters")

    comparison = optimizer.compare_formats(messages)
    print("\n" + format_token_comparison(comparison))
    print(get_format_decision_message(comparison))

    savings_ratio = (comparison["saved_tokens"] / comparison["json_tokens"]) * 100
    print(f"\nWith longer conversations, TOON saves {savings_ratio:.1f}% of tokens!")


def demo_optimization_decision():
    """Demonstrate how TOON optimizer makes decisions."""
    print_section("Optimization Decision Process")

    optimizer = ToonOptimizer(enabled=True)

    test_cases = [
        ("Very short message", [{"role": "user", "content": "Hi"}]),
        (
            "Medium conversation",
            [
                {"role": "user", "content": "Hello, how are you today?"},
                {"role": "assistant", "content": "I'm doing great, thanks!"},
            ],
        ),
        (
            "Long conversation with multiple turns",
            [
                {"role": "user", "content": f"Message {i}"}
                for i in range(5)
            ]
            + [
                {"role": "assistant", "content": f"Response {i}"}
                for i in range(5)
            ],
        ),
    ]

    for name, messages in test_cases:
        print(f"\n{name}:")
        comparison = optimizer.compare_formats(messages)
        print("  " + format_token_comparison(comparison))
        print("  " + get_format_decision_message(comparison))


def main():
    """Run all demonstrations."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "TOON OPTIMIZATION DEMO" + " " * 31 + "║")
    print("╚" + "═" * 68 + "╝")

    demo_simple_conversation()
    demo_conversation_with_tools()
    demo_conversation_with_tool_calls()
    demo_long_conversation()
    demo_optimization_decision()

    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("=" * 70)
    print("\nTo enable TOON optimization in your application:")
    print('  1. Add "enableToonOptimization": true to server_config.json')
    print("  2. Restart the application")
    print("  3. Token savings will be displayed automatically\n")


if __name__ == "__main__":
    main()
