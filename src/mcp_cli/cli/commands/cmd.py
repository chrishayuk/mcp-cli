# src/mcp_cli/cli/commands/cmd.py
"""Command mode implementation (fixed for ToolManager v0.9+).

This version works with either the modern *ToolManager* or the legacy
stream‑manager test stub.  Key points:
  • No longer relies on the removed ``get_internal_tools()`` method.
  • Defaults provider → *openai* and model → *gpt‑4o‑mini* (or
    *qwen2.5-coder* for Ollama) when the user omits them, preventing the
    “you must provide a model parameter” 400‑error from OpenAI.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional

import typer
from rich import print

from mcp_cli.cli.commands.base import BaseCommand
from mcp_cli.cli_options import process_options
from mcp_cli.tools.manager import ToolManager

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# helper – robust tool list extractor
# --------------------------------------------------------------------

def _extract_tools_list(manager: Any) -> List[Dict[str, Any]]:
    """Return tools as list[dict] irrespective of manager flavour."""
    if manager is None:
        return []
    if hasattr(manager, "get_unique_tools"):
        tools: List[Dict[str, Any]] = []
        for t in manager.get_unique_tools():  # type: ignore[attr-defined]
            tools.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "namespace": t.namespace,
                }
            )
        return tools
    if hasattr(manager, "get_internal_tools"):
        return list(manager.get_internal_tools())  # type: ignore[attr-defined]
    if hasattr(manager, "get_all_tools"):
        return list(manager.get_all_tools())  # type: ignore[attr-defined]
    return []

# --------------------------------------------------------------------
# CmdCommand
# --------------------------------------------------------------------

class CmdCommand(BaseCommand):
    """Non‑interactive one‑shot command runner."""

    def __init__(self) -> None:
        super().__init__("cmd", "Execute commands non‑interactively.")

    # ..................................................................
    # public API (invoked by run_command)
    # ..................................................................
    async def execute(self, tool_manager: ToolManager, **params) -> Optional[str]:
        logger.debug("Executing cmd with params: %s", params)

        # unwrap params -------------------------------------------------
        manager: Any = tool_manager
        provider: str = params.get("provider") or "openai"
        model: Optional[str] = params.get("model")
        if not model:
            model = os.getenv("LLM_MODEL") or (
                "gpt-4o-mini" if provider.lower() == "openai" else "qwen2.5-coder"
            )
            params["model"] = model  # keep downstream in sync

        input_file: Optional[str] = params.get("input")
        prompt_template: Optional[str] = params.get("prompt")
        output_file: Optional[str] = params.get("output")
        raw: bool = params.get("raw", False)
        tool: Optional[str] = params.get("tool")
        tool_args: Optional[str] = params.get("tool_args")
        system_prompt: Optional[str] = params.get("system_prompt")
        verbose: bool = params.get("verbose", False)

        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.WARNING,
            format="%(asctime)s  %(levelname)-8s  %(name)s | %(message)s",
            stream=sys.stderr,
        )

        # ---- direct tool path ----------------------------------------
        if tool:
            result = await self._run_single_tool(tool, tool_args, manager)
            self._write_output(result, output_file, raw)
            return result

        # ---- LLM round‑trip -----------------------------------------
        user_input = ""
        if input_file:
            if input_file == "-":
                user_input = ""
            else:
                try:
                    with open(input_file, encoding="utf-8") as fh:
                        user_input = fh.read()
                except OSError as exc:
                    logger.error("Could not read --input file: %s", exc)
                    raise typer.Exit(code=1) from exc

        maybe = self._run_llm_with_tools(
            provider=provider,
            model=model,
            user_input=user_input,
            prompt_template=prompt_template,
            custom_system_prompt=system_prompt,
            manager=manager,
        )
        result = await maybe if inspect.isawaitable(maybe) else maybe
        self._write_output(result, output_file, raw)
        return result

    # ..................................................................
    # single‑tool execution helper
    # ..................................................................
    async def _run_single_tool(
        self,
        tool_name: str,
        tool_args_json: Optional[str],
        manager: Any,
    ) -> str:
        try:
            args = json.loads(tool_args_json) if tool_args_json else {}
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON for --tool-args: %s", exc)
            raise typer.Exit(code=1) from exc

        logger.debug("Calling tool %s with %s", tool_name, args)

        if hasattr(manager, "execute_tool"):
            tcr = await manager.execute_tool(tool_name, args)  # type: ignore[attr-defined]
            if not tcr.success:
                logger.error("Tool reported error: %s", tcr.error)
                raise typer.Exit(code=1)
            return json.dumps(tcr.result, indent=2)
        if hasattr(manager, "call_tool"):
            result = await manager.call_tool(tool_name=tool_name, arguments=args)  # type: ignore[attr-defined]
            if result.get("isError"):
                logger.error("Tool reported error: %s", result.get("error"))
                raise typer.Exit(code=1)
            return json.dumps(result.get("content", ""), indent=2)
        logger.error("No compatible tool execution API on manager")
        raise typer.Exit(code=1)

    # ..................................................................
    # tool‑call processing
    # ..................................................................
    async def _process_tool_calls(self, tool_calls: List[dict], conversation: List[dict], manager: Any) -> None:
        from mcp_cli.llm.tools_handler import handle_tool_call
        for tc in tool_calls:
            await handle_tool_call(tc, conversation, tool_manager=manager)

    # ..................................................................
    # LLM work‑flow
    # ..................................................................
    async def _run_llm_with_tools(
        self,
        *,
        provider: str,
        model: str,
        user_input: str,
        prompt_template: Optional[str],
        custom_system_prompt: Optional[str],
        manager: Any,
    ) -> str:
        from mcp_cli.chat.system_prompt import generate_system_prompt
        from mcp_cli.llm.llm_client import get_llm_client
        from mcp_cli.llm.tools_handler import convert_to_openai_tools

        tools = _extract_tools_list(manager)
        openai_tools = convert_to_openai_tools(tools)

        system_prompt = custom_system_prompt or generate_system_prompt(tools)
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (prompt_template.replace("{{input}}", user_input) if prompt_template else user_input)},
        ]

        llm = get_llm_client(provider=provider, model=model)
        logger.debug("LLM client ready (%s / %s)", provider, model)

        completion = await llm.create_completion(messages=conversation, tools=openai_tools)
        if completion.get("tool_calls"):
            await self._process_tool_calls(completion["tool_calls"], conversation, manager)
            completion = await llm.create_completion(messages=conversation)

        if isinstance(completion, dict):
            return completion.get("response") or completion.get("content") or json.dumps(completion)
        if isinstance(completion, str):
            return completion
        return json.dumps(completion)

    # ..................................................................
    # output helper
    # ..................................................................
    def _write_output(self, data: str, path: Optional[str], raw: bool) -> None:
        if path and path != "-":
            try:
                with open(path, "w", encoding="utf-8") as fp:
                    fp.write(data if raw else data.strip() + "\n")
            except OSError as exc:
                logger.error("Could not write output: %s", exc)
                raise typer.Exit(code=1) from exc
        else:
            print(data if raw else data.strip())

    # ..................................................................
    # Typer wrapper
    # ..................................................................
    def register(self, app: typer.Typer, run_command_func: Callable) -> None:
        @app.command(self.name)
        def _cmd(
            config_file: str = "server_config.json",
            server: Optional[str] = None,
            provider: str = "openai",
            model: Optional[str] = None,
            disable_filesystem: bool = False,
            input: Optional[str] = None,  # noqa: A002
            prompt: Optional[str] = None,
            output: Optional[str] = None,
            raw: bool = False,
            tool: Optional[str] = None,
            tool_args: Optional[str] = None,
            system_prompt: Optional[str] = None,
            verbose: bool = False,
        ) -> None:
            # default model if omitted --------------------------------------------------
            if not model:
                model_local = os.getenv("LLM_MODEL") or ("gpt-4o-mini" if provider.lower() == "openai" else "qwen2.5-coder")
            else:
                model_local = model

            servers, _, server_names = process_options(server, disable_filesystem, provider, model_local, config_file)

            extra_params: Dict[str, Any] = {
                "provider": provider,
                "model": model_local,
                "server_names": server_names,
                "input": input,
                "prompt": prompt,
                "output": output,
                "raw": raw,
                "tool": tool,
                "tool_args": tool_args,
                "system_prompt": system_prompt,
                "verbose": verbose,
            }
            run_command_func(self.wrapped_execute, config_file, servers, extra_params=extra_params)
