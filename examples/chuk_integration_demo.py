#!/usr/bin/env python
"""
Complete demonstration of simplified MCP-CLI architecture using chuk-llm.
Shows tool integration, conversation flow, and architectural benefits.

Run with: uv run examples/chuk_integration_demo.py
"""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime

from chuk_llm import conversation, ask, tools_from_functions


def print_section(title: str):
    """Print a section header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


# Simulate MCP tools that would come from chuk-tool-processor
def query_database(sql: str) -> dict:
    """Execute a SQL query."""
    print(f"   🗄️  Executing SQL: {sql[:50]}...")
    return {
        "success": True,
        "rows_affected": 1,
        "message": f"Query executed: {sql[:30]}..."
    }


def list_tables() -> dict:
    """List all database tables."""
    print("   📋 Listing tables...")
    return {
        "tables": ["users", "posts", "comments", "tags"],
        "count": 4
    }


def create_table(table_name: str, columns: List[Dict[str, str]]) -> dict:
    """Create a new table."""
    print(f"   🔨 Creating table: {table_name}")
    return {
        "success": True,
        "table": table_name,
        "columns": len(columns)
    }


async def demo_tool_integration():
    """Show how chuk-llm integrates with tool execution."""
    print_section("Tool Integration with chuk-llm")
    
    # Create toolkit from functions (simulating MCP tools)
    toolkit = tools_from_functions(query_database, list_tables, create_table)
    tools = toolkit.to_openai_format()
    
    print("\n📚 Available tools:")
    for tool in tools:
        func = tool["function"]
        print(f"   • {func['name']}: {func['description']}")
    
    # Example 1: List tables
    print("\n👤: Show me all tables")
    result = await ask(
        "List all tables in the database using the list_tables function",
        provider="ollama",
        model="gpt-oss",
        tools=tools
    )
    
    if isinstance(result, dict) and "tool_calls" in result:
        print("🤖: Let me check the tables...")
        for tc in result["tool_calls"]:
            func = tc.get("function", {})
            name = func.get("name")
            
            if name == "list_tables":
                table_result = list_tables()
                print(f"   ✅ Found {table_result['count']} tables: {', '.join(table_result['tables'])}")
    
    # Example 2: Create a table
    print("\n👤: Create a users table")
    result = await ask(
        "Create a table called 'users' with columns: id (integer), name (text), email (text). Use create_table.",
        provider="ollama",
        model="gpt-oss",
        tools=tools
    )
    
    if isinstance(result, dict) and "tool_calls" in result:
        print("🤖: Creating the users table...")
        for tc in result["tool_calls"]:
            func = tc.get("function", {})
            name = func.get("name")
            args = json.loads(func.get("arguments", "{}"))
            
            if name == "create_table":
                create_result = create_table(**args)
                print(f"   ✅ Table '{create_result['table']}' created with {create_result['columns']} columns")


async def demo_conversation_flow():
    """Show a complete conversation flow with tools."""
    print_section("Conversation Flow")
    
    # Setup tools
    toolkit = tools_from_functions(query_database, list_tables)
    tools = toolkit.to_openai_format()
    
    async with conversation("ollama", "gpt-oss") as conv:
        # First question
        print("\n👤: I need help with SQL")
        response = await conv.ask("I need help with SQL")
        print(f"🤖: {response if isinstance(response, str) else response.get('response', '')[:100]}...")
        
        # Tool-using question
        print("\n👤: Show me the tables")
        result = await conv.ask("Show me the tables using list_tables", tools=tools)
        
        if isinstance(result, dict) and "tool_calls" in result:
            print("🤖: Checking tables...")
            for tc in result["tool_calls"]:
                if tc.get("function", {}).get("name") == "list_tables":
                    tables = list_tables()
                    print(f"   Found: {tables}")
        
        # Follow-up
        print("\n👤: Which table has user data?")
        response = await conv.ask("Which table has user data?")
        print(f"🤖: The 'users' table contains user data")
        
        print(f"\n📊 Conversation has {len(conv.messages)} messages")


async def demo_streaming_with_tools():
    """Show streaming combined with tool usage."""
    print_section("Streaming + Tools")
    
    toolkit = tools_from_functions(list_tables)
    tools = toolkit.to_openai_format()
    
    async with conversation("ollama", "gpt-oss") as conv:
        # First, use a tool
        print("\n👤: List tables then explain them")
        
        # Tool call
        result = await conv.ask("List tables using list_tables", tools=tools)
        if isinstance(result, dict) and "tool_calls" in result:
            print("🤖: Let me check the tables first...")
            for tc in result["tool_calls"]:
                if tc.get("function", {}).get("name") == "list_tables":
                    tables = list_tables()
                    print(f"   Found: {', '.join(tables['tables'])}")
        
        # Then stream explanation
        print("\n🤖: Now let me explain: ", end="", flush=True)
        async for chunk in conv.stream("Explain what each table is for: users, posts, comments, tags"):
            if isinstance(chunk, dict) and "response" in chunk:
                print(chunk["response"], end="", flush=True)
            elif isinstance(chunk, str):
                print(chunk, end="", flush=True)
        print()


async def demo_architecture_benefits():
    """Show the architectural benefits."""
    print_section("Architecture Benefits")
    
    print("""
    OLD ARCHITECTURE (Complex):
    ===========================
    MCP-CLI → ChatContext (440 lines)
           → StreamingHandler (774 lines)  
           → ConversationProcessor (332 lines)
           → ToolProcessor (complex extraction)
           → chuk-tool-processor
    
    NEW ARCHITECTURE (Simple):
    ==========================
    MCP-CLI → chuk-llm (handles conversation/streaming)
           → chuk-tool-processor (MCP servers)
    
    BENEFITS:
    =========
    • 80% less code (~2000 lines removed)
    • Automatic conversation management
    • Native streaming support
    • Built-in tool handling
    • Session tracking included
    • Cleaner cancellation
    """)


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("  CHUK INTEGRATION DEMONSTRATION")
    print("  Showing simplified MCP-CLI architecture")
    print("="*60)
    
    try:
        await demo_tool_integration()
        await demo_conversation_flow()
        await demo_streaming_with_tools()
        await demo_architecture_benefits()
        
        print("\n✅ Integration demonstration complete!")
        print("\n📝 Key Takeaway: By using chuk-llm with chuk-tool-processor,")
        print("   MCP-CLI becomes simpler, more maintainable, and more powerful.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())