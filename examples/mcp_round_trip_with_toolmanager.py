#!/usr/bin/env python
# examples/mcp_round_trip_with_toolmanager.py
"""
MCP "round-trip" example with SQLite stdio transport using ToolManager.

This script demonstrates:
1. Connecting to an MCP SQLite server
2. Using the enhanced ToolManager to handle tool execution
3. Handling OpenAI's function naming restrictions automatically
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv

# MCP CLI imports
from mcp_cli.tools.manager import ToolManager
from mcp_cli.llm.llm_client import get_llm_client
from mcp_cli.llm.system_prompt_generator import SystemPromptGenerator

# Initialize colorama for colored output
colorama_init(autoreset=True)

# ------------------------------------------------------------------ #
# Helper functions for displaying registry tools and results
# ------------------------------------------------------------------ #
async def display_registry_tools(tool_manager: ToolManager, namespace_filter: Optional[str] = None) -> Any:
    """Display all tools from the tool manager, optionally filtered by namespace."""
    # Get all tools
    tools = await tool_manager.get_all_tools()
    
    # Filter by namespace if requested
    if namespace_filter:
        tools = [t for t in tools if t.namespace == namespace_filter]
    
    # Print tools
    print(Fore.CYAN + f"🔧  Registered MCP tools ({len(tools)}):")
    
    for tool in tools:
        print(f"  • {Fore.GREEN}{tool.namespace}.{tool.name:<20}{Style.RESET_ALL} – {tool.description or '<no description>'}")
    print()
    
    return tools

def display_tool_result(result: Any, duration: float = None) -> None:
    """Display a single tool result with nice formatting."""
    # Determine if this is a ToolCallResult or a ToolResult
    is_error = False
    tool_name = getattr(result, 'tool_name', getattr(result, 'tool', 'unknown'))
    error = getattr(result, 'error', None)
    
    if error:
        is_error = True
    
    # Get the result content
    result_data = getattr(result, 'result', None)
    
    # Format output
    status = Fore.GREEN if not is_error else Fore.RED
    
    # Print header with tool name and duration
    if duration is not None:
        print(f"{status}{tool_name} ({duration:.3f}s){Style.RESET_ALL}")
    else:
        print(f"{status}{tool_name}{Style.RESET_ALL}")
    
    # Print error if any
    if is_error:
        print(f"  {Fore.RED}Error:{Style.RESET_ALL} {error}")
    else:
        # Format result based on type
        if isinstance(result_data, (dict, list)):
            print(f"  {Fore.CYAN}Result:{Style.RESET_ALL} {json.dumps(result_data, indent=2)}")
        else:
            print(f"  {Fore.CYAN}Result:{Style.RESET_ALL} {result_data}")

# ------------------------------------------------------------------ #
# Main function
# ------------------------------------------------------------------ #
async def main() -> None:
    load_dotenv()

    p = argparse.ArgumentParser(description="MCP round-trip with ToolManager")
    p.add_argument("--provider", default="openai", help="LLM provider (openai|ollama)")
    p.add_argument("--model", default="gpt-4o-mini", help="Model name")
    p.add_argument("--prompt", required=True, help="User prompt for the LLM")
    args = p.parse_args()

    # 1) Create and initialize the ToolManager
    # The key difference: use "stdio" namespace to match the direct ToolProcessor script
    tool_manager = ToolManager(
        config_file="server_config.json",
        servers=["sqlite"],
        server_names={0: "sqlite"}
    )
    
    # Initialize with the stdio namespace
    if not await tool_manager.initialize(namespace="stdio"):
        print(f"{Fore.RED}Failed to initialize ToolManager{Style.RESET_ALL}")
        return

    try:
        # 2) Display and filter stdio tools
        tools = await display_registry_tools(tool_manager, namespace_filter="stdio")
        
        if not tools:
            print(f"{Fore.YELLOW}No tools found with 'stdio' namespace. Showing all tools:{Style.RESET_ALL}")
            tools = await display_registry_tools(tool_manager)
        
        # 3) Get LLM-compatible tools with name mapping
        llm_tools, name_mapping = await tool_manager.get_adapted_tools_for_llm(provider=args.provider)
        
        if not llm_tools:
            print(f"{Fore.RED}No tools available for LLM. Check your configuration.{Style.RESET_ALL}")
            return
        
        # 4) Send prompt to LLM
        client = get_llm_client(provider=args.provider, model=args.model)
        sys_prompt = SystemPromptGenerator().generate_prompt({"tools": llm_tools})
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": args.prompt},
        ]
        
        completion = await client.create_completion(messages=messages, tools=llm_tools)
        
        # Print the assistant's response
        reply = completion.get("response", "")
        tool_calls = completion.get("tool_calls", [])
        
        if reply:
            print(Fore.CYAN + "=== Assistant Reply ===")
            print(reply, end="\n\n")
        
        # 5) Process tool calls using the ToolManager
        if tool_calls:
            print(Fore.CYAN + "=== Tool Calls ===")
            
            # Display each tool call before execution
            for tc in tool_calls:
                if tc.get("function") and "name" in tc.get("function", {}):
                    openai_name = tc["function"]["name"]
                    
                    # Convert back to original name for display
                    original_name = name_mapping.get(openai_name, openai_name)
                    args_str = tc["function"].get("arguments", "{}")
                    args_dict = json.loads(args_str) if isinstance(args_str, str) else args_str
                    
                    print(f"{Fore.GREEN}Tool: {openai_name} → {original_name}{Style.RESET_ALL}")
                    print(f"  {Fore.YELLOW}Arguments:{Style.RESET_ALL} {json.dumps(args_dict, indent=2)}")
            
            # Process all tool calls
            results = await tool_manager.process_llm_tool_calls(
                tool_calls=tool_calls,
                name_mapping=name_mapping
            )
            
            # Display results
            if results:
                print(Fore.CYAN + "=== Tool Results ===")
                for result in results:
                    duration = (result.end_time - result.start_time).total_seconds()
                    display_tool_result(result, duration)
            else:
                print(Fore.YELLOW + "[no results returned]")
                
        elif not reply:
            print(Fore.YELLOW + "[no response or tool calls]")
    
    finally:
        # 6) Clean up
        await tool_manager.close()

if __name__ == "__main__":
    asyncio.run(main())