# mcp_cli/cli/commands/cmd.py
"""Command mode implementation with multi-turn support by default."""
from __future__ import annotations

import inspect
import json
import logging
import os
import sys
from typing import Any, Callable, Dict, List, Optional

import typer
from rich import print
from rich.console import Console

from mcp_cli.cli.commands.base import BaseCommand
from mcp_cli.cli_options import process_options
from mcp_cli.tools.manager import ToolManager
from mcp_cli.provider_config import ProviderConfig

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
    """Non‑interactive one‑shot command runner with multi-turn support."""

    def __init__(self) -> None:
        super().__init__("cmd", "Execute commands non‑interactively (with multi-turn by default).")

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
        prompt: Optional[str] = params.get("prompt")
        output_file: Optional[str] = params.get("output")
        raw: bool = params.get("raw", False)
        tool: Optional[str] = params.get("tool")
        tool_args: Optional[str] = params.get("tool_args")
        system_prompt: Optional[str] = params.get("system_prompt")
        verbose: bool = params.get("verbose", False)
        single_turn: bool = params.get("single_turn", False)  # Now default is multi-turn
        max_turns: int = params.get("max_turns", 5)

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
                user_input = sys.stdin.read()
            else:
                try:
                    with open(input_file, encoding="utf-8") as fh:
                        user_input = fh.read()
                except OSError as exc:
                    logger.error("Could not read --input file: %s", exc)
                    raise typer.Exit(code=1) from exc

        # Error if no input source is provided
        if not prompt and not input_file:
            logger.error("Either --prompt or --input must be specified")
            raise typer.Exit(code=1)

        # Apply prompt formatting to input if needed
        formatted_input = prompt or user_input
        if prompt and input_file:
            formatted_input = prompt.replace("{{input}}", user_input)

        # We'll build the conversation as we go
        conversation = []
        
        # Generate or get the system prompt
        from mcp_cli.chat.system_prompt import generate_system_prompt
        from mcp_cli.llm.llm_client import get_llm_client
        from mcp_cli.llm.tools_handler import convert_to_openai_tools

        tools = _extract_tools_list(manager)
        openai_tools = convert_to_openai_tools(tools)
        
        system_prompt_text = system_prompt or generate_system_prompt(tools)
        conversation.append({"role": "system", "content": system_prompt_text})
        
        # Add the initial user message
        conversation.append({"role": "user", "content": formatted_input})

        # Get LLM client
        llm = get_llm_client(provider=provider, model=model)
        logger.debug("LLM client ready (%s / %s)", provider, model)

        console = Console(stderr=True) if verbose else None
        
        # For single-turn mode (now the exception)
        if single_turn:
            completion = await llm.create_completion(messages=conversation, tools=openai_tools)
            if completion.get("tool_calls"):
                await self._process_tool_calls(completion["tool_calls"], conversation, manager)
                completion = await llm.create_completion(messages=conversation)

            final_response = None
            if isinstance(completion, dict):
                final_response = completion.get("response") or completion.get("content") or json.dumps(completion)
            elif isinstance(completion, str):
                final_response = completion
            else:
                final_response = json.dumps(completion)
                
            self._write_output(final_response, output_file, raw)
            return final_response
        
        # Multi-turn mode (now the default)
        final_results = []
        tool_call_count = 0
        
        for turn in range(max_turns):
            if verbose and console:
                console.print(f"[dim]Turn {turn+1}/{max_turns}...[/dim]", end="\r")
                
            completion = await llm.create_completion(messages=conversation, tools=openai_tools)
            
            # If there are tool calls, process them
            if completion.get("tool_calls"):
                tool_call_count += len(completion.get("tool_calls", []))
                if verbose and console:
                    console.print(f"[dim]Processing {len(completion.get('tool_calls', []))} tool calls...[/dim]")
                    
                await self._process_tool_calls(completion["tool_calls"], conversation, manager)
                # Continue to next turn after tool calls
                continue
                
            # If we get a response and no tool calls, we're done
            if isinstance(completion, dict) and completion.get("response"):
                response = completion["response"]
                final_results.append(response)
                conversation.append({"role": "assistant", "content": response})
                
                # Display intermediate results if verbose
                if verbose and console:
                    console.print(f"[dim]Response: {response[:50]}...[/dim]")
                
                # End if this looks like a final response
                if "I have completed" in response or "Here's the final" in response:
                    break
            else:
                # Unknown completion format
                if verbose and console:
                    console.print("[yellow]Unexpected completion format - ending multi-turn.[/yellow]")
                break
                
            # We got a response but might need another turn
            # Add a generic "continue" prompt
            conversation.append({
                "role": "user", 
                "content": "Continue solving the problem. If you've completed the task, please summarize the solution."
            })
        
        if verbose and console:
            if tool_call_count > 0:
                console.print(f"[dim]Multi-turn completed with {tool_call_count} total tool calls.[/dim]")
            else:
                console.print("[dim]Multi-turn completed with no tool calls.[/dim]")
        
        # Combine final results
        final_response = "\n\n".join(final_results)
        self._write_output(final_response, output_file, raw)
        return final_response

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
    # Typer wrapper with improved help text - OPTION ONLY APPROACH
    # ..................................................................
    def register(self, app: typer.Typer, run_command_func: Callable) -> None:
        """Register the command with option-only approach."""
        
        @app.command(name=self.name, help=self.help)
        def cmd_standalone(
            prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Prompt to send to the LLM"),
            config_file: str = typer.Option("server_config.json", "--config-file", help="Configuration file path"),
            server: Optional[str] = typer.Option(None, "--server", help="Server to connect to"),
            provider: str = typer.Option("openai", "--provider", help="LLM provider name"),
            model: Optional[str] = typer.Option(None, "--model", help="Model name (defaults to provider's default)"),
            disable_filesystem: bool = typer.Option(False, "--disable-filesystem/--no-disable-filesystem", help="Disable filesystem access"),
            input: Optional[str] = typer.Option(None, "--input", help="Input file path (- for stdin)"),
            output: Optional[str] = typer.Option(None, "--output", help="Output file path (- for stdout)"),
            raw: bool = typer.Option(False, "--raw/--no-raw", help="Output raw response without formatting"),
            tool: Optional[str] = typer.Option(None, "--tool", help="Execute a specific tool directly"),
            tool_args: Optional[str] = typer.Option(None, "--tool-args", help="JSON string of tool arguments"),
            system_prompt: Optional[str] = typer.Option(None, "--system-prompt", help="Custom system prompt"),
            verbose: bool = typer.Option(False, "--verbose/--no-verbose", help="Enable verbose output"),
            single_turn: bool = typer.Option(False, "--single-turn", "-s", help="Disable multi-turn mode (multi-turn is default)"),
            max_turns: int = typer.Option(5, "--max-turns", help="Maximum number of turns in multi-turn mode"),
            api_base: Optional[str] = typer.Option(None, "--api-base", help="API base URL for the provider"),
            api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for the provider"),
        ):
            """
            Execute commands non-interactively with multi-turn support by default.
            
            Either --prompt or --input must be specified.
            """
            # Get provider config for defaults
            provider_cfg = ProviderConfig()
            
            # Handle API settings overrides
            if api_base or api_key:
                config_updates = {}
                if api_base:
                    config_updates["api_base"] = api_base
                if api_key:
                    config_updates["api_key"] = api_key
                    
                provider_cfg.set_provider_config(provider, config_updates)
            
            # Default model if omitted
            if not model:
                model_local = os.getenv("LLM_MODEL") or provider_cfg.get_default_model(provider) or "gpt-4o-mini"
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
                "single_turn": single_turn,
                "max_turns": max_turns,
                "api_base": api_base,
                "api_key": api_key,
            }
            run_command_func(self.wrapped_execute, config_file, servers, extra_params=extra_params)