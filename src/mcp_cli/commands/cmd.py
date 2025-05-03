"""
Command-mode (“non-interactive”) entry-point for MCP-CLI.

Two workflows are supported:

1. Direct tool invocation
     $ mcp-cli cmd run --tool echo --tool-args '{"text":"hi"}'

2. LLM inference (optionally followed by one round-trip of tool-calls)
     $ mcp-cli cmd run --input prompt.txt
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import typer
from rich import print  # noqa: F401 – imported for side-effects (tests capture)

from mcp_cli.chat.system_prompt import generate_system_prompt
from mcp_cli.llm.llm_client import get_llm_client
from mcp_cli.llm.tools_handler import convert_to_openai_tools, handle_tool_call

logger = logging.getLogger("mcp_cli.cmd")
app = typer.Typer(help="Command mode for non-interactive usage")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
async def _run_single_tool(
    tool_name: str,
    tool_args_json: Optional[str],
    stream_manager: Any,
) -> str:
    """Execute **one** tool through the given `stream_manager`."""
    try:
        tool_args = json.loads(tool_args_json) if tool_args_json else {}
    except json.JSONDecodeError as exc:  # pragma: no cover
        logger.error("Invalid JSON for --tool-args: %s", exc)
        raise typer.Exit(code=1) from exc

    logger.debug("Calling tool %s with %s", tool_name, tool_args)
    result = await stream_manager.call_tool(
        tool_name=tool_name,
        arguments=tool_args,
    )

    if result.get("isError"):
        logger.error("Tool reported error: %s", result.get("error"))
        raise typer.Exit(code=1)

    return json.dumps(result.get("content", ""), indent=2)


async def _process_tool_calls(
    tool_calls: List[dict],
    conversation: List[dict],
    stream_manager: Any,
) -> None:
    """Iterate over tool-calls requested by the LLM and execute them (once)."""
    for tc in tool_calls:
        await handle_tool_call(tc, conversation, stream_manager=stream_manager)


async def _run_llm_with_tools(
    *,
    provider: str,
    model: str,
    user_input: str,
    prompt_template: Optional[str],
    custom_system_prompt: Optional[str],
    stream_manager: Any,
) -> str:
    """LLM inference with (at most one) tool round-trip."""
    tools = stream_manager.get_internal_tools()
    openai_tools = convert_to_openai_tools(tools)

    system_prompt = custom_system_prompt or generate_system_prompt(tools)
    conversation = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": prompt_template.replace("{{input}}", user_input)
            if prompt_template
            else user_input,
        },
    ]

    llm = get_llm_client(provider=provider, model=model)
    logger.debug("LLM client ready (%s / %s)", provider, model)

    completion = llm.create_completion(messages=conversation, tools=openai_tools)

    # First pass: if the model wants tools → execute once
    if completion.get("tool_calls"):
        await _process_tool_calls(
            completion["tool_calls"],
            conversation,
            stream_manager,
        )
        completion = llm.create_completion(messages=conversation)

    # Extract final answer (no further tool-calls are attempted)
    if isinstance(completion, dict):
        return (
            completion.get("response")
            or completion.get("content")
            or json.dumps(completion)
        )
    if isinstance(completion, str):
        return completion
    return json.dumps(completion)


def _write_output(data: str, path: Optional[str], raw: bool) -> None:
    """Print to stdout or write to *path* (or “–” for stdout)."""
    if path and path != "-":
        try:
            with open(path, "w", encoding="utf-8") as fp:
                fp.write(data if raw else data.strip() + "\n")
        except OSError as exc:
            logger.error("Could not write output: %s", exc)
            raise typer.Exit(code=1) from exc
    else:
        print(data if raw else data.strip())


# ──────────────────────────────────────────────────────────────────────────────
# Typer coroutine (private) – registered below as `cmd_run`
# ──────────────────────────────────────────────────────────────────────────────
async def _cmd_run(  # noqa: C901 – CLI entry-points are naturally long
    *,
    # legacy “server_streams” kept for argv compatibility (ignored)
    server_streams: Optional[str] = None,  # noqa: ARG001
    # user / tool options
    input: Optional[str] = None,  # noqa: A002
    prompt: Optional[str] = None,
    output: Optional[str] = None,
    raw: bool = False,
    tool: Optional[str] = None,
    tool_args: Optional[str] = None,
    system_prompt: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    verbose: bool = False,
    server_names: Optional[Dict[int, str]] = None,  # noqa: ARG001 – reserved
    stream_manager: Any = None,
) -> str | None:  # returning result simplifies testing
    """Execute one *scriptable* command – tool-call or LLM query."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(name)s | %(message)s",
        stream=sys.stderr,
    )

    if stream_manager is None:
        logger.error("`--stream-manager` is required in command mode.")
        raise typer.Exit(code=1)

    provider = provider or os.getenv("LLM_PROVIDER", "openai")
    model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

    # ────────────── 1) direct tool invocation ──────────────
    if tool:
        result = await _run_single_tool(tool, tool_args, stream_manager)
        _write_output(result, output, raw)
        return result  # useful in tests

    # ────────────── 2) LLM workflow ──────────────
    user_input: str = ""
    if input:
        if input == "-":  # convention used in tests: read nothing
            user_input = ""
        else:
            try:
                with open(input, encoding="utf-8") as fp:
                    user_input = fp.read()
            except OSError as exc:
                logger.error("Could not read --input file: %s", exc)
                raise typer.Exit(code=1) from exc

    result = await _run_llm_with_tools(
        provider=provider,
        model=model,
        user_input=user_input,
        prompt_template=prompt,
        custom_system_prompt=system_prompt,
        stream_manager=stream_manager,
    )
    _write_output(result, output, raw)
    return result


# Register the coroutine with Typer – the actual CLI entry-point
cmd_run = app.command("run")(_cmd_run)

from types import SimpleNamespace as _SN  # placed here to avoid top-level import
cmd_run = _SN(callback=_cmd_run)          # gives tests .callback attribute