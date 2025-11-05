"""Command mode actions for Unix-friendly automation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from chuk_term.ui import output


async def cmd_action_async(
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    prompt: Optional[str] = None,
    tool: Optional[str] = None,
    tool_args: Optional[str] = None,
    system_prompt: Optional[str] = None,
    raw: bool = False,
    single_turn: bool = False,
    max_turns: int = 30,
) -> None:
    """
    Execute command mode operations for automation and scripting.

    Args:
        input_file: Input file path (use "-" for stdin)
        output_file: Output file path (use "-" for stdout)
        prompt: Prompt text to use
        tool: Tool name to execute
        tool_args: Tool arguments as JSON string
        system_prompt: Custom system prompt
        raw: Output raw response without formatting
        single_turn: Disable multi-turn conversation
        max_turns: Maximum conversation turns
    """
    from mcp_cli.context import get_context

    try:
        # Get the initialized context
        context = get_context()
        if not context or not context.tool_manager:
            output.error(
                "Context not initialized. This command requires a tool manager."
            )
            return

        # Handle tool execution mode
        if tool:
            await _execute_tool_direct(
                tool_name=tool,
                tool_args_json=tool_args,
                output_file=output_file,
                raw=raw,
            )
            return

        # Handle prompt mode with LLM
        if prompt or input_file:
            await _execute_prompt_mode(
                input_file=input_file,
                output_file=output_file,
                prompt=prompt,
                system_prompt=system_prompt,
                raw=raw,
                single_turn=single_turn,
                max_turns=max_turns,
            )
            return

        # No mode specified
        output.error("No operation specified. Use --tool or --prompt/--input")
        output.hint("Examples:")
        output.info("  mcp-cli cmd --tool list_tables")
        output.info(
            '  mcp-cli cmd --tool read_query --tool-args \'{"query": "SELECT * FROM users"}\''
        )
        output.info("  echo 'Analyze this' | mcp-cli cmd --input - --output result.txt")
        output.info("  mcp-cli cmd --prompt 'Summarize the data' --input data.txt")

    except Exception as e:
        output.error(f"Command execution failed: {e}")
        raise


async def _execute_tool_direct(
    tool_name: str,
    tool_args_json: Optional[str],
    output_file: Optional[str],
    raw: bool,
) -> None:
    """Execute a tool directly without LLM interaction."""
    from mcp_cli.context import get_context

    context = get_context()
    tool_manager = context.tool_manager

    if not tool_manager:
        output.error("Tool manager not initialized")
        return

    # Parse tool arguments
    tool_args = {}
    if tool_args_json:
        try:
            tool_args = json.loads(tool_args_json)
        except json.JSONDecodeError as e:
            output.error(f"Invalid JSON in tool arguments: {e}")
            return

    # Execute the tool
    try:
        if not raw:
            output.info(f"Executing tool: {tool_name}")

        tool_call_result = await tool_manager.execute_tool(tool_name, tool_args)

        # Check for errors
        if not tool_call_result.success or tool_call_result.error:
            output.error(f"Tool execution failed: {tool_call_result.error}")
            return

        # Extract the actual result
        result_data = tool_call_result.result

        # Format output
        if raw:
            result_str = (
                json.dumps(result_data)
                if not isinstance(result_data, str)
                else result_data
            )
        else:
            result_str = (
                json.dumps(result_data, indent=2)
                if not isinstance(result_data, str)
                else result_data
            )

        # Write output
        if output_file and output_file != "-":
            Path(output_file).write_text(result_str)
            if not raw:
                output.success(f"Output written to: {output_file}")
        else:
            # Write to stdout
            print(result_str)

    except Exception as e:
        output.error(f"Tool execution failed: {e}")
        raise


async def _execute_prompt_mode(
    input_file: Optional[str],
    output_file: Optional[str],
    prompt: Optional[str],
    system_prompt: Optional[str],
    raw: bool,
    single_turn: bool,
    max_turns: int,
) -> None:
    """Execute prompt mode with LLM interaction."""
    from mcp_cli.context import get_context

    context = get_context()

    # Read input
    input_text = ""
    if input_file:
        if input_file == "-":
            # Read from stdin
            input_text = sys.stdin.read()
        else:
            input_text = Path(input_file).read_text()

    # Build the full prompt
    if prompt and input_text:
        full_prompt = f"{prompt}\n\nInput:\n{input_text}"
    elif prompt:
        full_prompt = prompt
    elif input_text:
        full_prompt = input_text
    else:
        output.error("No prompt or input provided")
        return

    # Get the LLM client - use the model_manager from context
    try:
        # Use the model manager from context which has the correct provider/model
        model_manager = context.model_manager
        if not model_manager:
            # Fallback: create new one if context doesn't have it
            from mcp_cli.model_manager import ModelManager
            model_manager = ModelManager()
            # Set it to the correct provider/model from context
            model_manager.switch_model(context.provider, context.model)

        client = model_manager.get_client(provider=context.provider, model=context.model)

        if not client:
            output.error(
                f"Failed to get LLM client for {context.provider}/{context.model}"
            )
            return
    except Exception as e:
        output.error(f"Failed to initialize LLM client: {e}")
        return

    # Build messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": full_prompt})

    # Execute the conversation
    try:
        if not raw:
            output.info(f"Processing with {context.provider}/{context.model}...")

        # Get available tools
        tools = None
        if context.tool_manager and not single_turn:
            tools = await context.tool_manager.get_tools_for_llm()

        # Make the LLM call using chuk-llm interface
        response = await client.create_completion(
            model=context.model,
            messages=messages,
            tools=tools,
            max_tokens=4096,
        )

        # Extract the response - chuk-llm returns a dict
        result_text = response.get("response", "")
        tool_calls = response.get("tool_calls", [])

        # Handle tool calls if present
        if tool_calls and not single_turn:
            # Execute tools and continue conversation
            result_text = await _handle_tool_calls(
                client=client,
                messages=messages,
                tool_calls=tool_calls,
                response_text=result_text,
                max_turns=max_turns,
                raw=raw,
            )

        # Write output
        if output_file and output_file != "-":
            Path(output_file).write_text(result_text)
            if not raw:
                output.success(f"Output written to: {output_file}")
        else:
            # Write to stdout
            print(result_text)

    except Exception as e:
        output.error(f"LLM execution failed: {e}")
        raise


async def _handle_tool_calls(
    client,
    messages: list,
    tool_calls: list,
    response_text: str,
    max_turns: int,
    raw: bool,
) -> str:
    """Handle tool calls in multi-turn conversation."""
    from mcp_cli.context import get_context

    context = get_context()
    tool_manager = context.tool_manager

    if not tool_manager:
        output.error("Tool manager not initialized")
        return response_text

    # Add assistant message with tool calls
    messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "tool_calls": tool_calls,
        }
    )

    # Execute each tool call
    for tool_call in tool_calls:
        # Handle dict format from chuk-llm
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("function", {}).get("name", "")
            tool_args_str = tool_call.get("function", {}).get("arguments", "{}")
            tool_call_id = tool_call.get("id", "")
        else:
            # Handle object format
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments
            tool_call_id = tool_call.id

        # Parse arguments
        if isinstance(tool_args_str, str):
            tool_args = json.loads(tool_args_str)
        else:
            tool_args = tool_args_str

        if not raw:
            output.info(f"Executing tool: {tool_name}")

        try:
            tool_call_result = await tool_manager.execute_tool(tool_name, tool_args)
            # Extract result data and format as string
            result_data = (
                tool_call_result.result
                if tool_call_result.success
                else f"Error: {tool_call_result.error}"
            )
            result_str = (
                json.dumps(result_data)
                if not isinstance(result_data, str)
                else result_data
            )

            # Add tool result to messages
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result_str,
                }
            )
        except Exception as e:
            error_msg = f"Tool execution failed: {e}"
            output.error(error_msg)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": f"Error: {error_msg}",
                }
            )

    # Continue conversation
    turns = 1
    while turns < max_turns:
        tools = await tool_manager.get_tools_for_llm() if tool_manager else None
        response = await client.create_completion(
            model=context.model,
            messages=messages,
            tools=tools,
            max_tokens=4096,
        )

        # Extract response from dict
        response_text = response.get("response", "")
        response_tool_calls = response.get("tool_calls", [])

        # If no more tool calls, we're done
        if not response_tool_calls:
            return response_text

        # Add assistant message and execute tools
        messages.append(
            {
                "role": "assistant",
                "content": response_text,
                "tool_calls": response_tool_calls,
            }
        )

        # Execute tool calls
        for tool_call in response_tool_calls:
            # Handle dict format
            if isinstance(tool_call, dict):
                tool_name = tool_call.get("function", {}).get("name", "")
                tool_args_str = tool_call.get("function", {}).get("arguments", "{}")
                tool_call_id = tool_call.get("id", "")
            else:
                tool_name = tool_call.function.name
                tool_args_str = tool_call.function.arguments
                tool_call_id = tool_call.id

            # Parse arguments
            if isinstance(tool_args_str, str):
                tool_args = json.loads(tool_args_str)
            else:
                tool_args = tool_args_str

            if not raw:
                output.info(f"Executing tool: {tool_name}")

            try:
                tool_call_result = await tool_manager.execute_tool(tool_name, tool_args)
                # Extract result data and format as string
                result_data = (
                    tool_call_result.result
                    if tool_call_result.success
                    else f"Error: {tool_call_result.error}"
                )
                result_str = (
                    json.dumps(result_data)
                    if not isinstance(result_data, str)
                    else result_data
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": result_str,
                    }
                )
            except Exception as e:
                error_msg = f"Tool execution failed: {e}"
                output.error(error_msg)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": f"Error: {error_msg}",
                    }
                )

        turns += 1

    # Max turns reached
    if not raw:
        output.warning(f"Max turns ({max_turns}) reached")

    return response_text
