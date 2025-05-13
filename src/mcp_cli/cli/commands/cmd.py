"""
CMD - non-interactive one-shot command for MCP-CLI
──────────────────────────────────────────────────
• Sends a single prompt (or tool call) to the LLM, optionally with
  multi-turn reasoning & tool use, then exits.
• `--plain  /  -P`  ➜  no Rich colours / ANSI codes.
• `--raw`           ➜  text is forwarded exactly as received.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional

import typer
from rich import print as rich_print
from rich.console import Console

from mcp_cli.cli.commands.base import BaseCommand
from mcp_cli.cli_options import process_options
from mcp_cli.provider_config import ProviderConfig
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
        return list(manager.get_internal_tools())                       # type: ignore[attr-defined]

    if hasattr(manager, "get_all_tools"):
        return list(manager.get_all_tools())                            # type: ignore[attr-defined]

    return []


def _extract_response_text(completion: Any) -> str:
    """Handle dict-style completions or plain strings from llm_client."""
    if isinstance(completion, dict):
        return str(completion.get("response", ""))
    return str(completion)


# ════════════════════════════════════════════════════════════════════════
# Command
# ════════════════════════════════════════════════════════════════════════
class CmdCommand(BaseCommand):
    """`mcp-cli cmd` - run a prompt (multi-turn by default) then exit."""

    def __init__(self) -> None:
        help_text = (
            "Execute commands non-interactively (with multi-turn by default)."
        )
        super().__init__(name="cmd", help_text=help_text)  # BaseCommand stores this as `.help`
        self.help_text = help_text                         # Typer wrapper expects `.help_text`


    # ---------------------------------------------------------------------
    # Core logic
    # ---------------------------------------------------------------------
    async def execute(self, tool_manager: ToolManager, **params) -> Optional[str]:  # noqa: C901
        manager: Any = tool_manager
        provider: str = params.get("provider") or "openai"
        model: str = params.get("model") or "gpt-4o-mini"

        plain: bool = params.get("plain", False)
        raw: bool = params.get("raw", False)

        input_file: Optional[str] = params.get("input")
        prompt: Optional[str] = params.get("prompt")
        output_file: Optional[str] = params.get("output")

        tool: Optional[str] = params.get("tool")
        tool_args: Optional[str] = params.get("tool_args")

        system_prompt: Optional[str] = params.get("system_prompt")
        verbose: bool = params.get("verbose", False)
        single_turn: bool = params.get("single_turn", False)
        max_turns: int = params.get("max_turns", 5)

        # ---- logging --------------------
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.WARNING,
            format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
            stream=sys.stderr,
        )

        # ------------------------------------------------------------------
        # direct tool execution
        # ------------------------------------------------------------------
        if tool:
            result = await self._run_single_tool(manager, tool, tool_args)
            self._write_output(result, output_file, raw, plain)
            return result

        # ------------------------------------------------------------------
        # prompt / stdin
        # ------------------------------------------------------------------
        if input_file:
            input_content = sys.stdin.read() if input_file == "-" else open(input_file).read()
        else:
            input_content = ""

        if not prompt and not input_file:
            raise typer.BadParameter("Either --prompt or --input must be supplied")

        user_message = prompt or input_content
        if prompt and input_file:
            user_message = prompt.replace("{{input}}", input_content)

        # ------------------------------------------------------------------
        # LLM plumbing
        # ------------------------------------------------------------------
        from mcp_cli.llm.llm_client import get_llm_client
        from mcp_cli.llm.tools_handler import convert_to_openai_tools
        from mcp_cli.chat.system_prompt import generate_system_prompt

        tools = await _extract_tools_list(manager)
        openai_tools = convert_to_openai_tools(tools)
        system_message = system_prompt or generate_system_prompt(tools)

        conversation: List[Dict[str, str]] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        llm = get_llm_client(provider=provider, model=model)
        progress = Console(stderr=True, no_color=plain) if verbose else None

        # ------------------------------------------------------------------
        # single-turn
        # ------------------------------------------------------------------
        if single_turn:
            completion = await llm.create_completion(messages=conversation, tools=openai_tools)
            response_text = _extract_response_text(completion)
            self._write_output(response_text, output_file, raw, plain)
            return response_text

        # ------------------------------------------------------------------
        # multi-turn loop
        # ------------------------------------------------------------------
        for turn in range(max_turns):
            if progress:
                progress.print(f"[dim]Turn {turn+1}/{max_turns}…[/dim]", end="\r")

            completion = await llm.create_completion(messages=conversation, tools=openai_tools)
            tool_calls = completion.get("tool_calls", []) if isinstance(completion, dict) else []

            if tool_calls:
                await self._process_tool_calls(tool_calls, conversation, manager)
                continue  # another turn

            response_text = _extract_response_text(completion)
            conversation.append({"role": "assistant", "content": response_text})
            self._write_output(response_text, output_file, raw, plain)
            return response_text

        fallback = "Failed to complete within max_turns"
        self._write_output(fallback, output_file, raw, plain)
        return fallback

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    async def _run_single_tool(self, manager: Any, name: str, args_json: Optional[str]) -> str:
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as exc:
            raise typer.BadParameter("--tool-args must be valid JSON") from exc

        if hasattr(manager, "execute_tool"):
            res = await manager.execute_tool(name, args)                # type: ignore[attr-defined]
            if not res.success:
                raise RuntimeError(res.error or "Tool execution failed")
            return json.dumps(res.result, indent=2)

        if hasattr(manager, "call_tool"):
            res = await manager.call_tool(tool_name=name, arguments=args)  # type: ignore[attr-defined]
            if res.get("isError"):
                raise RuntimeError(res.get("error", "Tool execution failed"))
            return json.dumps(res.get("content", ""), indent=2)

        raise RuntimeError("ToolManager does not expose a compatible API")

    async def _process_tool_calls(self, calls: List[dict], convo: List[dict], manager: Any) -> None:
        from mcp_cli.llm.tools_handler import handle_tool_call
        for call in calls:
            await handle_tool_call(call, convo, tool_manager=manager)

    def _write_output(self, data: str, path: Optional[str], raw: bool, plain: bool) -> None:
        """Write to a file or stdout, respecting --plain / --raw flags."""
        text = str(data)

        # file or stdout?
        if path and path != "-":
            with open(path, "w", encoding="utf-8") as fp:
                fp.write(text)
                if not text.endswith("\n"):
                    fp.write("\n")
            return

        if plain or raw:
            print(text, end="" if text.endswith("\n") else "\n")
        else:
            rich_print(text)

    # ---------------------------------------------------------------------
    # Typer wiring
    # ---------------------------------------------------------------------
    def register(self, app: typer.Typer, run_command=None) -> None:  # noqa: D401
        if run_command is None:
            from mcp_cli.run_command import run_command_sync as run_command

        @app.command(self.name, help=self.help_text)
        def cmd_standalone(  # noqa: PLR0913
            prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Prompt to send to the LLM"),
            config_file: str = typer.Option("server_config.json", "--config-file", help="Configuration file path"),
            server: Optional[str] = typer.Option(None, "--server", help="Server to connect to"),
            provider: str = typer.Option("openai", "--provider", help="LLM provider name"),
            model: Optional[str] = typer.Option(None, "--model", help="Model name (defaults to provider's default)"),
            disable_filesystem: bool = typer.Option(False, "--disable-filesystem/--no-disable-filesystem", help="Disable filesystem access"),
            input: Optional[str] = typer.Option(None, "--input", help="Input file path (- for stdin)"),
            output: Optional[str] = typer.Option(None, "--output", help="Output file path (- for stdout)"),
            raw: bool = typer.Option(False, "--raw/--no-raw", help="Output raw response without formatting"),
            plain: bool = typer.Option(False, "--plain", "-P", help="Disable colour / Rich markup in output"),
            tool: Optional[str] = typer.Option(None, "--tool", help="Execute a specific tool directly"),
            tool_args: Optional[str] = typer.Option(None, "--tool-args", help="JSON string of tool arguments"),
            system_prompt: Optional[str] = typer.Option(None, "--system-prompt", help="Custom system prompt"),
            verbose: bool = typer.Option(False, "--verbose/--no-verbose", help="Enable verbose output"),
            single_turn: bool = typer.Option(False, "--single-turn", "-s", help="Disable multi-turn mode"),
            max_turns: int = typer.Option(5, "--max-turns", help="Maximum number of turns in multi-turn mode"),
            api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL for the provider"),
            api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for the provider"),
        ) -> None:
            prov_cfg = ProviderConfig()
            if api_base or api_key:
                overrides: Dict[str, str] = {}
                if api_base:
                    overrides["api_base"] = api_base
                if api_key:
                    overrides["api_key"] = api_key
                prov_cfg.set_provider_config(provider, overrides)

            resolved_model = model or prov_cfg.get_default_model(provider)

            servers, _, server_names = process_options(
                server, disable_filesystem, provider, resolved_model, config_file
            )

            extra: Dict[str, Any] = {
                "provider": provider,
                "model": resolved_model,
                "server_names": server_names,
                "input": input,
                "prompt": prompt,
                "output": output,
                "raw": raw,
                "plain": plain,
                "tool": tool,
                "tool_args": tool_args,
                "system_prompt": system_prompt,
                "verbose": verbose,
                "single_turn": single_turn,
                "max_turns": max_turns,
                "api_base": api_base,
                "api_key": api_key,
            }

            run_command(self.wrapped_execute, config_file, servers, extra_params=extra)
