# src/mcp_cli/cli/commands/cmd.py
"""Command mode implementation."""
from __future__ import annotations
import asyncio
import json
import logging
import os
import sys
import inspect
from typing import Any, Callable, Dict, List, Optional

import typer
from rich import print

# mcp cli imports
from mcp_cli.cli.commands.base import BaseCommand
from mcp_cli.cli_options import process_options
from mcp_cli.tools.manager import ToolManager

# logger
logger = logging.getLogger(__name__)


class CmdCommand(BaseCommand):
    """Command for non-interactive command execution."""
    
    def __init__(self):
        super().__init__("cmd", "Execute commands non-interactively.")
    
    async def execute(self, tool_manager: ToolManager, **params) -> Optional[str]:
        """Execute non-interactive command mode."""
        logger.debug(f"Executing cmd with params: {params}")
        
        # Extract command-specific parameters
        stream_manager = tool_manager
        server_names = params.get("server_names")
        input_file = params.get("input")
        prompt_template = params.get("prompt")
        output_file = params.get("output")
        raw = params.get("raw", False)
        tool = params.get("tool")
        tool_args = params.get("tool_args")
        system_prompt = params.get("system_prompt")
        provider = params.get("provider")
        model = params.get("model")
        verbose = params.get("verbose", False)
        
        # Configure logging for this command
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.WARNING,
            format="%(asctime)s  %(levelname)-8s  %(name)s | %(message)s",
            stream=sys.stderr,
        )
        
        # Direct tool execution
        if tool:
            result = await self._run_single_tool(tool, tool_args, stream_manager)
            self._write_output(result, output_file, raw)
            return result
        
        # LLM workflow
        user_input = ""
        if input_file:
            if input_file == "-":  # Convention used in tests: read nothing
                user_input = ""
            else:
                try:
                    with open(input_file, encoding="utf-8") as fp:
                        user_input = fp.read()
                except OSError as exc:
                    logger.error(f"Could not read --input file: {exc}")
                    raise typer.Exit(code=1) from exc
        
        # _run_llm_with_tools may return awaitable or plain str
        maybe = self._run_llm_with_tools(
            provider=provider,
            model=model,
            user_input=user_input,
            prompt_template=prompt_template,
            custom_system_prompt=system_prompt,
            stream_manager=stream_manager,
        )
        if inspect.isawaitable(maybe):
            result = await maybe
        else:
            result = maybe
        
        self._write_output(result, output_file, raw)
        return result
    
    async def _run_single_tool(
        self,
        tool_name: str,
        tool_args_json: Optional[str],
        stream_manager: Any,
    ) -> str:
        """Execute a single tool through the ToolManager or legacy stream_manager."""
        # Parse JSON args
        try:
            args = json.loads(tool_args_json) if tool_args_json else {}
        except json.JSONDecodeError as exc:
            logger.error(f"Invalid JSON for --tool-args: {exc}")
            raise typer.Exit(code=1) from exc
        
        logger.debug(f"Calling tool {tool_name} with {args}")
        
        # Prefer new ToolManager.execute_tool API
        if hasattr(stream_manager, "execute_tool"):
            tcr = await stream_manager.execute_tool(tool_name, args)
            if not tcr.success:
                logger.error(f"Tool reported error: {tcr.error}")
                raise typer.Exit(code=1)
            return json.dumps(tcr.result, indent=2)
        else:
            # Fallback to legacy call_tool if available
            try:
                result = await stream_manager.call_tool(
                    tool_name=tool_name,
                    arguments=args,
                )
            except AttributeError:
                logger.error("ToolManager does not support legacy call_tool")
                raise typer.Exit(code=1)
            
            if result.get("isError"):
                logger.error(f"Tool reported error: {result.get('error')}")
                raise typer.Exit(code=1)
            return json.dumps(result.get("content", ""), indent=2)
    
    async def _process_tool_calls(
        self,
        tool_calls: List[dict],
        conversation: List[dict],
        stream_manager: Any,
    ) -> None:
        """Process tool calls requested by the LLM."""
        from mcp_cli.llm.tools_handler import handle_tool_call
        
        for tc in tool_calls:
            await handle_tool_call(tc, conversation, stream_manager=stream_manager)
    
    async def _run_llm_with_tools(
        self,
        *,
        provider: str,
        model: str,
        user_input: str,
        prompt_template: Optional[str],
        custom_system_prompt: Optional[str],
        stream_manager: Any,
    ) -> Any:
        """Run LLM inference with optional tool round-trip."""
        from mcp_cli.chat.system_prompt import generate_system_prompt
        from mcp_cli.llm.llm_client import get_llm_client
        from mcp_cli.llm.tools_handler import convert_to_openai_tools

        # Get tools and convert to OpenAI format
        tools = stream_manager.get_internal_tools()
        openai_tools = convert_to_openai_tools(tools)

        # Create conversation with system prompt
        system_prompt = custom_system_prompt or generate_system_prompt(tools)
        conversation = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": prompt_template.replace("{{input}}", user_input)
                        if prompt_template else user_input,
            },
        ]

        # Initialize LLM client
        llm = get_llm_client(provider=provider, model=model)
        logger.debug(f"LLM client ready ({provider} / {model})")

        # First pass LLM call
        completion = await llm.create_completion(messages=conversation, tools=openai_tools)

        # Execute tool calls if requested
        if completion.get("tool_calls"):
            await self._process_tool_calls(
                completion["tool_calls"],
                conversation,
                stream_manager,
            )
            # Second pass LLM call after tool execution
            completion = await llm.create_completion(messages=conversation)

        # Extract final answer
        if isinstance(completion, dict):
            return (
                completion.get("response")
                or completion.get("content")
                or json.dumps(completion)
            )
        if isinstance(completion, str):
            return completion
        return json.dumps(completion)

    def _write_output(self, data: str, path: Optional[str], raw: bool) -> None:
        """Write output to file or stdout."""
        if path and path != "-":
            try:
                with open(path, "w", encoding="utf-8") as fp:
                    fp.write(data if raw else data.strip() + "\n")
            except OSError as exc:
                logger.error(f"Could not write output: {exc}")
                raise typer.Exit(code=1) from exc
        else:
            print(data if raw else data.strip())

    def register(self, app: typer.Typer, run_command_func: Callable) -> None:
        """Register CMD command with extended parameters."""
        @app.command(self.name)
        def _cmd(
            config_file: str = "server_config.json",
            server: Optional[str] = None,
            provider: str = "openai",
            model: Optional[str] = None,
            disable_filesystem: bool = False,
            input: Optional[str] = None,
            prompt: Optional[str] = None,
            output: Optional[str] = None,
            raw: bool = False,
            tool: Optional[str] = None,
            tool_args: Optional[str] = None,
            system_prompt: Optional[str] = None,
            verbose: bool = False,
        ) -> None:
            # Process options
            servers, _, server_names = process_options(
                server, disable_filesystem, provider, model, config_file
            )
            
            # Prepare extra parameters
            extra_params: Dict[str, Any] = {
                "provider": provider,
                "model": model,
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
            
            # Execute the command
            run_command_func(
                self.wrapped_execute,
                config_file,
                servers,
                extra_params=extra_params,
            )
