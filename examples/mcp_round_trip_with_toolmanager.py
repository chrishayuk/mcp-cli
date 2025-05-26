#!/usr/bin/env python
# examples/mcp_round_trip_with_toolmanager.py
"""
MCP round-trip demo using ToolManager + SQLite (stdio).

• Lists all stdio tools.
• Lets an OpenAI (or Ollama) model call them via function-calling.
• Executes the calls one-by-one via ToolManager.run_tool().
• Feeds results back to the LLM for a polished final answer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv

# ── MCP & LLM helpers ───────────────────────────────────────────────────
from chuk_llm.llm.llm_client import get_llm_client
from mcp_cli.tools.manager import ToolManager
from mcp_cli.llm.system_prompt_generator import SystemPromptGenerator

colorama_init(autoreset=True)


# ╭── printing helpers ──────────────────────────────────────────────────╮
async def display_registry_tools(
    tm: ToolManager, namespace: Optional[str] = None
) -> Dict[str, Any]:
    tools = await tm.get_all_tools()
    if namespace:
        tools = [t for t in tools if t.namespace == namespace]

    ns_note = f" – namespace='{namespace}'" if namespace else ""
    print(
        Fore.CYAN + f"🔧  Registered MCP tools ({len(tools)}){ns_note}" + Style.RESET_ALL
    )
    for t in tools:
        print(
            f"  • {Fore.GREEN}{t.namespace}.{t.name:<20}{Style.RESET_ALL} – "
            f"{t.description or '<no description>'}"
        )
    print()
    return {f"{t.namespace}.{t.name}": t for t in tools}


def pretty_result(name: str, result: Any, started: datetime, ended: datetime) -> None:
    head = f"{Fore.GREEN}{name} ({(ended - started).total_seconds():.3f}s){Style.RESET_ALL}"
    print(head)
    body = (
        json.dumps(result, indent=2)
        if isinstance(result, (dict, list))
        else str(result)
    )
    print(f"  {Fore.CYAN}Result:{Style.RESET_ALL} {body}")


# ╰───────────────────────────────────────────────────────────────────────╯


async def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(description="MCP ↔ LLM round-trip demo")
    ap.add_argument("--provider", default="openai", help="LLM provider (openai|ollama)")
    ap.add_argument("--model", default="gpt-4o-mini", help="Model name")
    ap.add_argument("--prompt", required=True, help="User prompt for the LLM")
    args = ap.parse_args()

    # 1️⃣  ToolManager initialisation
    tm = ToolManager(
        config_file="server_config.json",
        servers=["sqlite"],
        server_names={0: "sqlite"},
    )
    if not await tm.initialize(namespace="stdio"):
        print(Fore.RED + "❌  ToolManager initialisation failed" + Style.RESET_ALL)
        return

    try:
        await display_registry_tools(tm, "stdio")

        # 2️⃣  Build tools-v2 schema for the LLM
        llm_tools, name_map = await tm.get_adapted_tools_for_llm(provider=args.provider)
        if not llm_tools:
            print(Fore.RED + "❌  No LLM-compatible tools found" + Style.RESET_ALL)
            return

        # 3️⃣  Initial LLM call (allow tool usage)
        client = get_llm_client(provider=args.provider, model=args.model)
        sys_prompt = SystemPromptGenerator().generate_prompt({"tools": llm_tools})
        messages: List[Dict[str, str | None]] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": args.prompt},
        ]

        completion = await client.create_completion(
            messages=messages, tools=llm_tools, tool_choice="auto"
        )
        reply, tool_calls = completion.get("response", ""), completion.get(
            "tool_calls", []
        )

        if reply:
            print(Fore.CYAN + "\n=== Assistant reply ===" + Style.RESET_ALL)
            print(reply, "\n")

        # 4️⃣  Execute tool calls one-by-one through ToolManager.run_tool()
        if tool_calls:
            print(Fore.CYAN + "=== Tool calls ===" + Style.RESET_ALL)

            results: List[Any] = []
            for tc in tool_calls:
                oai_name = tc["function"]["name"]
                orig_name = name_map.get(oai_name, oai_name)
                args_dict = json.loads(tc["function"].get("arguments", "{}"))
                print(f"{Fore.GREEN}{oai_name} → {orig_name}{Style.RESET_ALL}")
                print(
                    f"  {Fore.YELLOW}Args:{Style.RESET_ALL} {json.dumps(args_dict, indent=2)}"
                )

                start = datetime.utcnow()
                res = await tm.run_tool(orig_name, args_dict)  # ← key change
                end = datetime.utcnow()
                results.append(res)

                pretty_result(orig_name, res, start, end)

            # 5️⃣  Feed tool outputs back for a final model answer
            messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
            for tc, res in zip(tool_calls, results):
                messages.append(
                    {
                        "role": "tool",
                        "name": tc["function"]["name"],
                        "content": json.dumps(res),
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
