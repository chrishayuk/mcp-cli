#!/usr/bin/env python
# examples/mcp_round_trip_with_toolmanager.py
"""
MCP round-trip demo using ToolManager + SQLite (stdio transport).

• Lists all stdio-namespace tools.
• Lets an OpenAI (or Ollama) model call them via function-calling.
• Executes the calls through ToolManager and feeds results back to the model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv

# ── MCP & LLM helpers ───────────────────────────────────────────────────
from chuk_llm.llm.llm_client import get_llm_client
from mcp_cli.tools.manager import ToolManager
from mcp_cli.llm.system_prompt_generator import SystemPromptGenerator

colorama_init(autoreset=True)


# ╭── pretty-printing helpers ────────────────────────────────────────────╮
async def display_registry_tools(
    tm: ToolManager, namespace_filter: Optional[str] = None
) -> List[Any]:
    """Return & pretty-print all tools (optionally filtered by namespace)."""
    tools = await tm.get_all_tools()
    if namespace_filter:
        tools = [t for t in tools if t.namespace == namespace_filter]

    ns_info = f" – namespace='{namespace_filter}'" if namespace_filter else ""
    print(Fore.CYAN + f"🔧  Registered MCP tools ({len(tools)}){ns_info}" + Style.RESET_ALL)
    for t in tools:
        desc = t.description or "<no description>"
        print(f"  • {Fore.GREEN}{t.namespace}.{t.name:<20}{Style.RESET_ALL} – {desc}")
    print()
    return tools


def display_tool_result(res: Any) -> None:
    """Pretty-print a ToolResult (or ToolCallResult)."""
    name = getattr(res, "tool_name", getattr(res, "tool", "unknown"))
    is_err = bool(getattr(res, "error", None))
    data = getattr(res, "result", None)

    head_color = Fore.RED if is_err else Fore.GREEN
    head = head_color + name

    # duration info if timestamps are present
    if hasattr(res, "start_time") and hasattr(res, "end_time"):
        dur = (res.end_time - res.start_time).total_seconds()
        head += f" ({dur:.3f}s)"
    print(head + Style.RESET_ALL)

    if is_err:
        print(f"  {Fore.RED}Error:{Style.RESET_ALL} {res.error}")
    else:
        body = json.dumps(data, indent=2) if isinstance(data, (dict, list)) else data
        print(f"  {Fore.CYAN}Result:{Style.RESET_ALL} {body}")


# ╰───────────────────────────────────────────────────────────────────────╯


async def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(description="MCP ↔ LLM round-trip with ToolManager")
    ap.add_argument("--provider", default="openai", help="LLM provider (openai|ollama)")
    ap.add_argument("--model", default="gpt-4o-mini", help="Model name")
    ap.add_argument("--prompt", required=True, help="User prompt for the LLM")
    args = ap.parse_args()

    # 1️⃣  create ToolManager & connect to stdio/SQLite
    tm = ToolManager(
        config_file="server_config.json",
        servers=["sqlite"],
        server_names={0: "sqlite"},
    )
    if not await tm.initialize(namespace="stdio"):
        print(Fore.RED + "❌  ToolManager initialisation failed" + Style.RESET_ALL)
        return

    try:
        # 2️⃣  list stdio tools (or all if none under that namespace)
        stdio_tools = await display_registry_tools(tm, "stdio")
        if not stdio_tools:
            stdio_tools = await display_registry_tools(tm)

        # 3️⃣  convert registry tools → tools-v2 schema + OpenAI-safe names
        llm_tools, name_map = await tm.get_adapted_tools_for_llm(provider=args.provider)
        if not llm_tools:
            print(Fore.RED + "❌  No LLM-compatible tools found" + Style.RESET_ALL)
            return

        # 4️⃣  ask the model, allowing tool calls
        client = get_llm_client(provider=args.provider, model=args.model)
        sys_prompt = SystemPromptGenerator().generate_prompt({"tools": llm_tools})
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": args.prompt},
        ]

        completion = await client.create_completion(
            messages=messages, tools=llm_tools, tool_choice="auto"
        )

        reply = completion.get("response", "")
        tool_calls = completion.get("tool_calls", [])

        if reply:
            print(Fore.CYAN + "\n=== Assistant reply ===" + Style.RESET_ALL)
            print(reply, "\n")

        # 5️⃣  run any tool calls via ToolManager
        if tool_calls:
            print(Fore.CYAN + "=== Tool calls ===" + Style.RESET_ALL)
            for tc in tool_calls:
                fn = tc["function"]
                openai_name = fn["name"]
                orig_name = name_map.get(openai_name, openai_name)
                args_dict = json.loads(fn.get("arguments", "{}"))
                print(f"{Fore.GREEN}{openai_name} → {orig_name}{Style.RESET_ALL}")
                print(f"  {Fore.YELLOW}Args:{Style.RESET_ALL} {json.dumps(args_dict, indent=2)}")

            # process calls (old API → returns plain ToolResult objects)
            results = await tm.process_tool_calls(tool_calls=tool_calls, name_mapping=name_map)

            print(Fore.CYAN + "\n=== Tool results ===" + Style.RESET_ALL)
            for res in results:
                display_tool_result(res)

            # 6️⃣  hand tool outputs back to the LLM for a final answer
            messages.append(
                {"role": "assistant", "content": None, "tool_calls": tool_calls}
            )
            for tc, res in zip(tool_calls, results):
                messages.append(
                    {
                        "role": "tool",
                        "name": tc["function"]["name"],
                        "content": json.dumps(getattr(res, "result", None)),
                        "tool_call_id": tc["id"],
                    }
                )

            follow_up = await client.create_completion(messages=messages)
            final = follow_up.get("response", "")
            if final:
                print(Fore.CYAN + "\n=== Final answer ===" + Style.RESET_ALL)
                print(final)

        elif not reply:
            print(Fore.YELLOW + "[no reply and no tool calls]" + Style.RESET_ALL)

    finally:
        await tm.close()


if __name__ == "__main__":
    asyncio.run(main())
