#!/usr/bin/env python3
"""
MCP-CLI Streaming with Tools Demo
==================================
Shows how streaming works when tools are invoked.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.chat.streaming_handler import StreamingResponseHandler
from mcp_cli.ui.streaming_display import StreamingContext
from rich.console import Console
from chuk_term.ui.output import get_output
from chuk_term.ui.theme import set_theme


async def demo_tool_streaming():
    """Demonstrate streaming with tool calls."""
    output = get_output()
    set_theme("default")
    console = Console()
    
    output.print("\n" + "="*70)
    output.print("MCP-CLI TOOL STREAMING DEMO", style="bold cyan")
    output.print("Shows how streaming works with tool invocations", style="dim")
    output.print("="*70)
    
    # Create the streaming handler
    handler = StreamingResponseHandler(console)
    
    # Mock a client that returns tool calls in streaming
    class MockToolStreamingClient:
        async def create_completion(self, messages, tools=None, stream=True, **kwargs):
            """Simulate streaming that includes tool calls."""
            
            user_msg = messages[-1]["content"] if messages else ""
            
            if "echo" in user_msg.lower() or "database" in user_msg.lower():
                # Response that triggers a tool call
                # First stream the thinking
                thinking_chunks = [
                    {"response": "I'll", "tool_calls": None},
                    {"response": "I'll use", "tool_calls": None},
                    {"response": "I'll use the", "tool_calls": None},
                    {"response": "I'll use the echo", "tool_calls": None},
                    {"response": "I'll use the echo tool", "tool_calls": None},
                    {"response": "I'll use the echo tool to", "tool_calls": None},
                    {"response": "I'll use the echo tool to process", "tool_calls": None},
                    {"response": "I'll use the echo tool to process your", "tool_calls": None},
                    {"response": "I'll use the echo tool to process your message.", "tool_calls": None},
                ]
                
                for chunk in thinking_chunks:
                    yield chunk
                    await asyncio.sleep(0.05)
                
                # Then include the tool call
                tool_call_chunk = {
                    "response": "I'll use the echo tool to process your message.",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "echo_text",
                                "arguments": json.dumps({"message": "Hello from the tool!"})
                            }
                        }
                    ]
                }
                yield tool_call_chunk
                
            else:
                # Regular response without tools
                chunks = [
                    {"response": "This", "tool_calls": None},
                    {"response": "This is", "tool_calls": None},
                    {"response": "This is a", "tool_calls": None},
                    {"response": "This is a normal", "tool_calls": None},
                    {"response": "This is a normal response", "tool_calls": None},
                    {"response": "This is a normal response without", "tool_calls": None},
                    {"response": "This is a normal response without tools.", "tool_calls": None},
                ]
                
                for chunk in chunks:
                    yield chunk
                    await asyncio.sleep(0.05)
    
    client = MockToolStreamingClient()
    
    # Test 1: Regular message (no tools)
    output.print("\n" + "-"*40)
    output.print("Test 1: Regular Response (No Tools)", style="bold yellow")
    output.print("-"*40)
    
    messages = [{"role": "user", "content": "Tell me something"}]
    output.user_message("Tell me something")
    
    result = await handler.stream_response(
        client=client,
        messages=messages,
        tools=[]
    )
    
    output.info(f"Result: {result['chunks_received']} chunks, {len(result.get('tool_calls', []))} tool calls")
    
    await asyncio.sleep(1)
    
    # Test 2: Message that triggers tool
    output.print("\n" + "-"*40)
    output.print("Test 2: Response with Tool Call", style="bold yellow")
    output.print("-"*40)
    
    messages = [{"role": "user", "content": "Echo this message"}]
    output.user_message("Echo this message")
    
    # Define available tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo_text",
                "description": "Echo a text message",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "The message to echo"}
                    },
                    "required": ["message"]
                }
            }
        }
    ]
    
    result = await handler.stream_response(
        client=client,
        messages=messages,
        tools=tools
    )
    
    output.info(f"Result: {result['chunks_received']} chunks")
    
    if result.get("tool_calls"):
        output.success(f"Tool calls detected: {len(result['tool_calls'])}")
        for i, tc in enumerate(result["tool_calls"]):
            func = tc.get("function", {})
            output.print(f"  Tool {i+1}: {func.get('name')} with args: {func.get('arguments')}")
    
    await asyncio.sleep(1)
    
    # Now demonstrate what happens after tool execution
    output.print("\n" + "-"*40)
    output.print("Test 3: Tool Execution Flow", style="bold yellow")
    output.print("-"*40)
    
    # Simulate the full flow
    output.user_message("Use the echo tool to say 'Hello World'")
    
    # 1. Stream the assistant's decision to use a tool
    with StreamingContext(
        console=console,
        title="🤖 Assistant",
        mode="thinking",
        refresh_per_second=8,
        transient=True
    ) as ctx:
        thinking = "I'll use the echo_text tool to display your message."
        for i in range(0, len(thinking), 5):
            ctx.update(thinking[i:i+5])
            await asyncio.sleep(0.05)
    
    # 2. Show tool invocation
    output.tool_call("echo_text", {"message": "Hello World"})
    
    # 3. Show tool execution with streaming
    with StreamingContext(
        console=console,
        title="⚙️ Tool: echo_text",
        mode="tool",
        refresh_per_second=8,
        transient=True
    ) as ctx:
        tool_output = "Executing echo_text...\nMessage: 'Hello World'\nEchoing: Hello World\nTool completed successfully."
        for i in range(0, len(tool_output), 3):
            ctx.update(tool_output[i:i+3])
            await asyncio.sleep(0.02)
    
    # 4. Show tool result
    output.success("Tool Result: Hello World")
    
    # 5. Stream the final assistant response
    with StreamingContext(
        console=console,
        title="🤖 Assistant",
        mode="response",
        refresh_per_second=8,
        transient=True
    ) as ctx:
        response = "I've successfully echoed your message 'Hello World' using the echo_text tool."
        for i in range(0, len(response), 4):
            ctx.update(response[i:i+4])
            await asyncio.sleep(0.03)
    
    output.print("\n" + "="*70)
    output.success("✅ Tool Streaming Demo Complete!")
    output.print("="*70)
    
    output.info("\n📊 Key Points:")
    output.print("• Streaming continues while deciding to use tools")
    output.print("• Tool calls are extracted from streaming chunks")
    output.print("• Tool execution can also use streaming display")
    output.print("• Final response streams after tool completion")


async def demo_multiple_tools():
    """Demonstrate streaming with multiple tool calls."""
    output = get_output()
    console = Console()
    
    output.print("\n" + "-"*40)
    output.print("Test 4: Multiple Tool Calls", style="bold yellow")
    output.print("-"*40)
    
    output.user_message("Query the database and format the results")
    
    # Stream the planning phase
    with StreamingContext(
        console=console,
        title="💭 Planning",
        mode="thinking",
        refresh_per_second=8,
        transient=True
    ) as ctx:
        planning = """Let me break this down:
1. First, I'll query the database to get the data
2. Then I'll format the results in a readable way
3. Finally, I'll present the formatted output"""
        
        for i in range(0, len(planning), 6):
            ctx.update(planning[i:i+6])
            await asyncio.sleep(0.03)
    
    # Execute multiple tools
    tools = [
        ("sqlite_query", {"query": "SELECT * FROM users LIMIT 5"}),
        ("format_table", {"data": "[results]", "format": "markdown"})
    ]
    
    for tool_name, tool_args in tools:
        output.tool_call(tool_name, tool_args)
        
        with StreamingContext(
            console=console,
            title=f"⚙️ Tool: {tool_name}",
            mode="tool",
            refresh_per_second=8,
            transient=True
        ) as ctx:
            if "query" in tool_name:
                result = """Connecting to database...
Executing query: SELECT * FROM users LIMIT 5
Fetching results...
Retrieved 5 rows successfully."""
            else:
                result = """Formatting data as markdown table...
Processing 5 rows...
Adding headers and alignment...
Table formatted successfully."""
            
            for i in range(0, len(result), 4):
                ctx.update(result[i:i+4])
                await asyncio.sleep(0.02)
        
        output.success(f"✓ {tool_name} completed")
        await asyncio.sleep(0.5)
    
    # Final response with formatted results
    with StreamingContext(
        console=console,
        title="🤖 Assistant",
        mode="response",
        refresh_per_second=8,
        transient=True
    ) as ctx:
        response = """Here are the query results formatted as a table:

| ID | Name     | Email              | Status |
|----|----------|--------------------|--------|
| 1  | Alice    | alice@example.com  | Active |
| 2  | Bob      | bob@example.com    | Active |
| 3  | Charlie  | charlie@example.com| Inactive |
| 4  | Diana    | diana@example.com  | Active |
| 5  | Eve      | eve@example.com    | Pending |

The query returned 5 users from the database, showing their basic information and current status."""
        
        for i in range(0, len(response), 8):
            ctx.update(response[i:i+8])
            await asyncio.sleep(0.02)


async def main():
    """Run all demos."""
    await demo_tool_streaming()
    await demo_multiple_tools()


if __name__ == "__main__":
    asyncio.run(main())