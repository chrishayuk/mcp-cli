# mcp_cli/cli/commands/cmd.py
"""
Clean CMD command implementation using ModelManager.

Non-interactive one-shot command for MCP-CLI that sends a single prompt
(or tool call) to the LLM, optionally with multi-turn reasoning & tool use.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, List, Optional

import typer
from rich import print as rich_print

from mcp_cli.cli.commands.base import BaseCommand
from mcp_cli.cli_options import process_options
from mcp_cli.model_manager import ModelManager
from mcp_cli.tools.manager import ToolManager

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════
async def _extract_tools_list(tool_manager: ToolManager) -> List[Dict[str, Any]]:
    """Extract tool metadata from ToolManager."""
    if not tool_manager:
        return []

    try:
        tool_infos = await tool_manager.get_unique_tools()
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "namespace": t.namespace,
            }
            for t in tool_infos
        ]
    except Exception as exc:
        logger.warning(f"Error extracting tools: {exc}")
        return []


def _extract_response_text(completion: Any) -> str:
    """Extract text from LLM completion response."""
    if isinstance(completion, dict):
        return str(completion.get("response", ""))
    return str(completion)


# ════════════════════════════════════════════════════════════════════════
# Command
# ════════════════════════════════════════════════════════════════════════
class CmdCommand(BaseCommand):
    """Non-interactive command execution with LLM and tool support."""

    def __init__(self) -> None:
        help_text = "Execute commands non-interactively (with multi-turn by default)."
        super().__init__(name="cmd", help_text=help_text)
        # self.help is already set by super().__init__

    async def execute(self, tool_manager: ToolManager, **params) -> Optional[str]:
        """Execute the command with given parameters."""
        # Extract parameters
        provider = params.get("provider")
        model = params.get("model")
        api_base = params.get("api_base")
        api_key = params.get("api_key")

        plain = params.get("plain", False)
        raw = params.get("raw", False)
        input_file = params.get("input")
        prompt = params.get("prompt")
        output_file = params.get("output")
        tool = params.get("tool")
        tool_args = params.get("tool_args")
        system_prompt = params.get("system_prompt")
        verbose = params.get("verbose", False)
        single_turn = params.get("single_turn", False)
        max_turns = params.get("max_turns", 5)

        # Set up logging
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.WARNING,
            format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
            stream=sys.stderr,
        )

        # Create ModelManager and configure if needed
        model_manager = ModelManager()

        if api_base or api_key:
            target_provider = provider or model_manager.get_active_provider()
            model_manager.configure_provider(
                target_provider, api_key=api_key, api_base=api_base
            )

        # Switch model if requested
        if provider and model:
            model_manager.switch_model(provider, model)
        elif provider:
            model_manager.switch_provider(provider)
        elif model:
            model_manager.switch_to_model(model)

        # Direct tool execution
        if tool:
            result = await self._run_single_tool(tool_manager, tool, tool_args)
            self._write_output(result, output_file, raw, plain)
            return result

        # Prompt/input handling
        if input_file:
            input_content = (
                sys.stdin.read() if input_file == "-" else open(input_file).read()
            )
        else:
            input_content = ""

        if not prompt and not input_file:
            raise typer.BadParameter("Either --prompt or --input must be supplied")

        user_message = prompt or input_content
        if prompt and input_file:
            user_message = prompt.replace("{{input}}", input_content)

        # Set up LLM interaction
        tools = await _extract_tools_list(tool_manager)
        openai_tools = self._convert_tools_for_llm(tools, tool_manager)

        from mcp_cli.chat.system_prompt import generate_system_prompt

        system_message = system_prompt or generate_system_prompt(tools)

        conversation: List[Dict[str, str]] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        # Get LLM client from ModelManager
        client = model_manager.get_client()
        progress = Console(stderr=True, no_color=plain) if verbose else None

        # Single-turn mode
        if single_turn:
            completion = await client.create_completion(
                messages=conversation, tools=openai_tools
            )
            response_text = _extract_response_text(completion)
            self._write_output(response_text, output_file, raw, plain)
            return response_text

        # Multi-turn loop
        for turn in range(max_turns):
            if progress:
                progress.print(f"[dim]Turn {turn + 1}/{max_turns}…[/dim]", end="\r")

            completion = await client.create_completion(
                messages=conversation, tools=openai_tools
            )
            tool_calls = (
                completion.get("tool_calls", []) if isinstance(completion, dict) else []
            )

            if tool_calls:
                await self._process_tool_calls(tool_calls, conversation, tool_manager)
                continue  # another turn

            response_text = _extract_response_text(completion)
            conversation.append({"role": "assistant", "content": response_text})
            self._write_output(response_text, output_file, raw, plain)
            return response_text

        # Max turns reached
        fallback = "Failed to complete within max_turns"
        self._write_output(fallback, output_file, raw, plain)
        return fallback

    def _convert_tools_for_llm(
        self, tools: List[Dict[str, Any]], tool_manager: ToolManager
    ) -> List[Dict[str, Any]]:
        """Convert tools to LLM format."""
        try:
            # Try to use tool manager's conversion if available
            if hasattr(tool_manager, "convert_to_openai_tools"):
                return tool_manager.convert_to_openai_tools(tools)

            # Fallback to basic conversion
            return [
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", "unknown"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    },
                }
                for tool in tools
            ]
        except Exception as exc:
            logger.warning(f"Error converting tools: {exc}")
            return []

    async def _run_single_tool(
        self, tool_manager: ToolManager, name: str, args_json: Optional[str]
    ) -> str:
        """Execute a single tool directly."""
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError as exc:
            raise typer.BadParameter("--tool-args must be valid JSON") from exc

        try:
            result = await tool_manager.execute_tool(name, args)
            if not result.success:
                raise RuntimeError(result.error or "Tool execution failed")
            return json.dumps(result.result, indent=2)
        except Exception as exc:
            raise RuntimeError(f"Tool execution failed: {exc}") from exc

    async def _process_tool_calls(
        self, calls: List[dict], conversation: List[dict], tool_manager: ToolManager
    ) -> None:
        """Process tool calls and add results to conversation."""
        from mcp_cli.llm.tools_handler import handle_tool_call

        for call in calls:
            await handle_tool_call(call, conversation, tool_manager=tool_manager)

    def _write_output(
        self, data: str, path: Optional[str], raw: bool, plain: bool
    ) -> None:
        """Write output to file or stdout."""
        text = str(data)

        # Write to file
        if path and path != "-":
            with open(path, "w", encoding="utf-8") as fp:
                fp.write(text)
                if not text.endswith("\n"):
                    fp.write("\n")
            return

        # Write to stdout
        if plain or raw:
            print(text, end="" if text.endswith("\n") else "\n")
        else:
            rich_print(text)

    def register(self, app: typer.Typer, run_command=None) -> None:
        """Register the command with Typer."""
        if run_command is None:
            from mcp_cli.run_command import run_command_sync as run_command

        @app.command(self.name, help=self.help)
        def cmd_standalone(
            prompt: Optional[str] = typer.Option(
                None, "--prompt", "-p", help="Prompt to send to the LLM"
            ),
            config_file: str = typer.Option(
                "server_config.json", "--config-file", help="Configuration file path"
            ),
            server: Optional[str] = typer.Option(
                None, "--server", help="Server to connect to"
            ),
            provider: Optional[str] = typer.Option(
                None, "--provider", help="LLM provider name"
            ),
            model: Optional[str] = typer.Option(None, "--model", help="Model name"),
            disable_filesystem: bool = typer.Option(
                False, "--disable-filesystem", help="Disable filesystem access"
            ),
            input: Optional[str] = typer.Option(
                None, "--input", help="Input file path (- for stdin)"
            ),
            output: Optional[str] = typer.Option(
                None, "--output", help="Output file path (- for stdout)"
            ),
            raw: bool = typer.Option(
                False, "--raw", help="Output raw response without formatting"
            ),
            plain: bool = typer.Option(
                False, "--plain", "-P", help="Disable colour/Rich markup in output"
            ),
            tool: Optional[str] = typer.Option(
                None, "--tool", help="Execute a specific tool directly"
            ),
            tool_args: Optional[str] = typer.Option(
                None, "--tool-args", help="JSON string of tool arguments"
            ),
            system_prompt: Optional[str] = typer.Option(
                None, "--system-prompt", help="Custom system prompt"
            ),
            verbose: bool = typer.Option(
                False, "--verbose", help="Enable verbose output"
            ),
            single_turn: bool = typer.Option(
                False, "--single-turn", "-s", help="Disable multi-turn mode"
            ),
            max_turns: int = typer.Option(
                5, "--max-turns", help="Maximum number of turns in multi-turn mode"
            ),
            api_base: Optional[str] = typer.Option(
                None, "--api-base", help="API base URL for the provider"
            ),
            api_key: Optional[str] = typer.Option(
                None, "--api-key", help="API key for the provider"
            ),
        ) -> None:
            """Execute non-interactive commands with LLM and tool support."""

            # Use ModelManager to determine effective provider/model
            model_manager = ModelManager()
            effective_provider = provider or model_manager.get_active_provider()
            effective_model = model or model_manager.get_active_model()

            servers, _, server_names = process_options(
                server,
                disable_filesystem,
                effective_provider,
                effective_model,
                config_file,
            )

            extra: Dict[str, Any] = {
                "provider": provider,
                "model": model,
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


# ════════════════════════════════════════════════════════════════════════
# Usage Examples
# ════════════════════════════════════════════════════════════════════════

"""
# Execute a prompt with default model
mcp-cli cmd --prompt "List database tables"

# Execute with specific provider and model
mcp-cli cmd --provider anthropic --model claude-3-sonnet --prompt "Analyze this data"

# Execute a tool directly
mcp-cli cmd --tool list_tables --tool-args '{}'

# Multi-turn conversation with output to file
mcp-cli cmd --prompt "Create a report" --output report.txt --max-turns 10

# Single-turn with custom system prompt
mcp-cli cmd --prompt "Hello" --single-turn --system-prompt "You are a helpful assistant"

# With API overrides
mcp-cli cmd --provider openai --api-key your-key --prompt "Hello world"
"""
