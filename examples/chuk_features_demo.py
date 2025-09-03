#!/usr/bin/env python
"""
Core chuk-llm features demonstration: conversation memory, streaming, tool calling, sessions.

Run with: uv run examples/chuk_features_demo.py
"""

import asyncio
import json
from datetime import datetime

from chuk_llm import conversation, ask, tools_from_functions


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


async def demo_conversation_memory():
    """Conversation with memory."""
    print_section("Conversation Memory")

    async with conversation("ollama", "gpt-oss") as conv:
        print("\n👤: My favorite color is blue")
        response = await conv.ask("My favorite color is blue")
        print(
            f"🤖: {response if isinstance(response, str) else response.get('response', '')}"
        )

        print("\n👤: What's my favorite color?")
        response = await conv.ask("What's my favorite color?")
        print(
            f"🤖: {response if isinstance(response, str) else response.get('response', '')}"
        )


async def demo_streaming():
    """Streaming responses."""
    print_section("Streaming")

    async with conversation("ollama", "gpt-oss") as conv:
        print("\n👤: Count to 5")
        print("🤖: ", end="", flush=True)

        async for chunk in conv.stream("Count to 5"):
            if isinstance(chunk, dict) and "response" in chunk:
                content = chunk["response"]
            elif isinstance(chunk, str):
                content = chunk
            else:
                continue

            if content:
                print(content, end="", flush=True)
        print()


async def demo_tool_calling():
    """Tool calling example using chuk-llm's tools_from_functions."""
    print_section("Tool Calling")

    # Define tools as simple functions
    def calculate(operation: str, a: float, b: float) -> dict:
        """Perform a calculation."""
        ops = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b if b != 0 else "Cannot divide by zero",
        }
        result = ops.get(operation, "Unknown operation")
        return {"result": result, "operation": operation, "a": a, "b": b}

    def get_weather(location: str) -> dict:
        """Get weather for a location."""
        return {"location": location, "temperature": 72, "condition": "Sunny"}

    # Create toolkit using chuk-llm's tools_from_functions
    toolkit = tools_from_functions(calculate, get_weather)
    tools = toolkit.to_openai_format()

    # Math example
    print("\n👤: What's 15 times 8?")
    result = await ask(
        "What's 15 times 8? Use the calculate function with operation='multiply'.",
        provider="ollama",
        model="gpt-oss",
        tools=tools,
    )

    if isinstance(result, dict):
        print(f"🤖: {result.get('response', '')}")

        if "tool_calls" in result and result["tool_calls"]:
            print("   ✅ Tool called!")
            for tc in result["tool_calls"]:
                func = tc.get("function", {})
                name = func.get("name")
                args_str = func.get("arguments", "{}")

                if name == "calculate":
                    args = json.loads(args_str)
                    calc_result = calculate(**args)
                    print(f"   → Result: {calc_result}")
    else:
        print(f"🤖: {result}")

    # Weather example
    print("\n👤: What's the weather in Paris?")
    result = await ask(
        "What's the weather in Paris? Use the get_weather function.",
        provider="ollama",
        model="gpt-oss",
        tools=tools,
    )

    if isinstance(result, dict):
        print(f"🤖: {result.get('response', '')}")

        if "tool_calls" in result and result["tool_calls"]:
            print("   ✅ Tool called!")
            for tc in result["tool_calls"]:
                func = tc.get("function", {})
                name = func.get("name")
                args_str = func.get("arguments", "{}")

                if name == "get_weather":
                    args = json.loads(args_str)
                    weather_result = get_weather(**args)
                    print(f"   → Result: {weather_result}")
    else:
        print(f"🤖: {result}")


async def demo_session_info():
    """Session tracking."""
    print_section("Session Tracking")

    async with conversation("ollama", "gpt-oss") as conv:
        await conv.ask("Hello")

        print(f"\nSession ID: {conv.session_id if conv.has_session else 'None'}")
        print(f"Messages: {len(conv.messages)}")
        print(f"Has tracking: {conv.has_session}")


async def demo_streaming_with_tools():
    """Streaming with tool support - shows tool call then streaming."""
    print_section("Streaming + Tools")

    # Define a simple tool
    def get_time() -> dict:
        """Get current time."""

        return {"time": datetime.now().strftime("%I:%M %p")}

    # Create toolkit
    toolkit = tools_from_functions(get_time)
    tools = toolkit.to_openai_format()

    print("\n👤: What time is it?")

    # Non-streaming tool call first (more reliable)
    result = await ask(
        "What time is it? Use the get_time function.",
        provider="ollama",
        model="gpt-oss",
        tools=tools,
    )

    if isinstance(result, dict) and "tool_calls" in result:
        print("🤖: Let me check the time...")
        for tc in result["tool_calls"]:
            func = tc.get("function", {})
            if func.get("name") == "get_time":
                time_result = get_time()
                print("   ✅ Tool called: get_time()")
                print(f"   → Result: {time_result['time']}")
    else:
        print(
            f"🤖: {result if isinstance(result, str) else result.get('response', '')}"
        )

    # Now demonstrate streaming separately
    print("\n👤: Now count to 3")
    print("🤖: ", end="", flush=True)

    async with conversation("ollama", "gpt-oss") as conv:
        async for chunk in conv.stream("Count to 3"):
            if isinstance(chunk, dict) and "response" in chunk:
                content = chunk["response"]
                if content:
                    print(content, end="", flush=True)
            elif isinstance(chunk, str):
                print(chunk, end="", flush=True)
    print()


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  CHUK-LLM CHAT FEATURES")
    print("=" * 60)

    try:
        await demo_conversation_memory()
        await demo_streaming()
        await demo_tool_calling()
        await demo_session_info()
        await demo_streaming_with_tools()

        print("\n✅ All features demonstrated")

    except KeyboardInterrupt:
        print("\n\n[Interrupted]")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
