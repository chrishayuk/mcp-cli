"""
Command mode module for non-interactive, scriptable usage of MCP CLI.
"""
import typer
import os
import sys
import json
import logging
import asyncio
from typing import Optional
from rich import print

# llm imports
from mcp_cli.llm.llm_client import get_llm_client
from mcp_cli.llm.tools_handler import fetch_tools, convert_to_openai_tools

# Chat context for system prompt generation
from mcp_cli.chat.system_prompt import generate_system_prompt

# Configure logging
logger = logging.getLogger("mcp_cli.cmd")

app = typer.Typer(help="Command mode for non-interactive usage")

@app.command("run")
async def cmd_run(
    server_streams,
    input: Optional[str] = None,
    prompt: Optional[str] = None, 
    output: Optional[str] = None,
    raw: bool = False,
    tool: Optional[str] = None,
    tool_args: Optional[str] = None,
    system_prompt: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    verbose: bool = False,
):
    """Run a command in non-interactive mode for automation and scripting."""
    
    # Configure logging based on verbosity
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            stream=sys.stderr
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(levelname)s: %(message)s",
            stream=sys.stderr
        )
    
    try:
        # Get provider and model from options or environment
        provider_name = provider or os.getenv("LLM_PROVIDER", "openai")
        model_name = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        
        # Handle input from file or stdin
        input_text = ""
        if input:
            if input == "-":
                input_text = sys.stdin.read().strip()
            else:
                try:
                    with open(input, "r") as f:
                        input_text = f.read().strip()
                except Exception as e:
                    logger.error(f"Error reading input file: {e}")
                    sys.exit(1)
        
        # If tool is specified, execute tool directly
        if tool:
            result = await run_single_tool(server_streams, tool, tool_args)
            write_output(result, output, raw)
            return
            
        # Otherwise, run LLM inference with tools
        result = await run_llm_with_tools(
            server_streams, 
            provider_name, 
            model_name, 
            input_text,
            prompt, 
            system_prompt
        )
        
        # Output result
        write_output(result, output, raw)
            
    except Exception as e:
        logger.error(f"Error in command mode: {e}")
        sys.exit(1)

async def run_single_tool(server_streams, tool_name, tool_args_json):
    """Run a single tool directly."""
    from mcp_cli.llm.tools_handler import send_tools_call
    
    # Parse tool arguments
    tool_args = {}
    if tool_args_json:
        try:
            tool_args = json.loads(tool_args_json)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in tool arguments")
            sys.exit(1)
    
    # Try each server until we find one that has the tool
    for read_stream, write_stream in server_streams:
        try:
            # First check if the server has the requested tool
            tools_response = await fetch_tools(read_stream, write_stream)
            tools_list = []
            
            if isinstance(tools_response, dict):
                tools_list = tools_response.get("tools", [])
            elif isinstance(tools_response, list):
                # Some implementations might return a list directly
                tools_list = tools_response
                
            tool_names = [tool["name"] for tool in tools_list if isinstance(tool, dict) and "name" in tool]
            
            if tool_name in tool_names:
                logger.debug(f"Found tool '{tool_name}' on server")
                # Call the tool
                result = await send_tools_call(
                    read_stream=read_stream, 
                    write_stream=write_stream,
                    name=tool_name,
                    arguments=tool_args
                )
                
                # Check for errors
                if result.get("isError"):
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"Error calling tool {tool_name}: {error_msg}")
                    sys.exit(1)
                    
                # Return the tool result
                return json.dumps(result.get("content", "No content"), indent=2)
        except Exception as e:
            logger.debug(f"Error with server: {e}")
            continue  # Try next server
    
    # If we get here, no server had the requested tool
    logger.error(f"Tool '{tool_name}' not found on any server")
    sys.exit(1)

async def run_llm_with_tools(server_streams, provider, model, input_text, prompt_template, custom_system_prompt):
    """Run LLM inference with tool support."""
    # Collect tools from all servers
    all_tools = []
    for read_stream, write_stream in server_streams:
        try:
            # Get tools from this server
            tools_response = await fetch_tools(read_stream, write_stream)
            if tools_response and isinstance(tools_response, dict):
                all_tools.extend(tools_response.get("tools", []))
            elif tools_response and isinstance(tools_response, list):
                # Some implementations might return a list directly
                all_tools.extend(tools_response)
        except Exception as e:
            # Just log the error and continue
            logger.warning(f"Failed to fetch tools from a server: {e}")
    
    # Convert tools to OpenAI format
    openai_tools = convert_to_openai_tools(all_tools)
    
    # Generate system prompt
    system_prompt = custom_system_prompt or generate_system_prompt(all_tools)
    
    # Create LLM client
    try:
        client = get_llm_client(provider=provider, model=model)
        logger.debug(f"Using LLM provider: {provider}, model: {model}")
    except Exception as e:
        logger.error(f"Error creating LLM client: {e}")
        return f"Error: Could not initialize LLM client with provider={provider}, model={model}. {str(e)}"
    
    # Build the user prompt
    user_prompt = input_text
    if prompt_template:
        # Replace {{input}} in the template with the actual input
        user_prompt = prompt_template.replace("{{input}}", input_text)
    
    # Create conversation
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Get completion
    try:
        logger.debug(f"Sending request to LLM...")
        completion = client.create_completion(
            messages=conversation,
            tools=openai_tools
        )
        
        if completion is None:
            logger.warning(f"LLM returned None completion")
            return "Error: LLM returned no response. Please check your API key and connection."
        
        # Handle tool calls if necessary
        if completion.get("tool_calls"):
            logger.debug(f"LLM requested tool calls - processing...")
            # Process tool calls
            await process_tool_calls(completion.get("tool_calls"), conversation, server_streams)
            
            # Get final response after tool calls
            logger.debug(f"Getting final response after tool calls...")
            try:
                max_iterations = 3  # Maximum number of additional tool call iterations
                iterations = 0
                
                final_completion = client.create_completion(messages=conversation)
                logger.debug(f"Final completion keys: {list(final_completion.keys() if final_completion else [])}")
                
                if final_completion is None:
                    logger.warning(f"LLM returned None for final completion")
                    return "Error: LLM returned no response after tool calls."
                
                # If there are more tool calls, process them too
                while "tool_calls" in final_completion and final_completion["tool_calls"] and iterations < max_iterations:
                    logger.debug(f"LLM requested more tool calls (iteration {iterations+1}/{max_iterations})")
                    await process_tool_calls(final_completion.get("tool_calls"), conversation, server_streams)
                    
                    # Try one more time with another completion
                    logger.debug(f"Getting final response after additional tool calls...")
                    final_completion = client.create_completion(messages=conversation)
                    iterations += 1
                
                # If we max out on iterations but still have tool calls, consider it a success but mention it
                if iterations >= max_iterations and "tool_calls" in final_completion and final_completion["tool_calls"]:
                    logger.warning(f"Reached maximum tool call iterations ({max_iterations})")
                    # Create a summary of the conversation as a fallback response
                    # Use the last few user and tool messages to create a summary
                    tool_messages = [msg for msg in conversation if msg.get("role") == "tool"]
                    if tool_messages:
                        last_tools = tool_messages[-min(3, len(tool_messages)):]
                        summary = "Based on the tools executed, here's what I found:\n\n"
                        for msg in last_tools:
                            summary += f"- From {msg.get('name', 'tool')}: {msg.get('content', 'No content')[:150]}...\n"
                        return summary
                
                # Now extract the response
                response = None
                if "response" in final_completion and final_completion["response"] is not None:
                    response = final_completion.get("response")
                elif "content" in final_completion:
                    response = final_completion.get("content")
                elif isinstance(final_completion, str):
                    # Some implementations might return the string directly
                    response = final_completion
                else:
                    # If we can't find a response field, try to convert the entire object to string
                    try:
                        response = json.dumps(final_completion)
                    except:
                        response = str(final_completion)
                
                if response is None:
                    logger.warning(f"Could not extract response from final completion")
                    return "Error: Could not extract a valid response from LLM output."
                    
                return response
            except Exception as e:
                logger.error(f"Error getting final response: {e}")
                return f"Error: Failed to get final response after tool calls: {str(e)}"
        else:
            # Return direct response
            response = completion.get("response")
            if response is None:
                logger.warning(f"'response' field missing in completion: {completion}")
                return "Error: LLM response format invalid (missing 'response' field)."
                
            return response
    except Exception as e:
        logger.error(f"Error during LLM completion: {e}")
        return f"Error: An exception occurred while processing your request: {str(e)}"
    
async def process_tool_calls(tool_calls, conversation, server_streams):
    """Process tool calls and update conversation."""
    from mcp_cli.llm.tools_handler import handle_tool_call
    
    for i, tool_call in enumerate(tool_calls):
        logger.debug(f"Processing tool call {i+1}/{len(tool_calls)}")
        await handle_tool_call(tool_call, conversation, server_streams)

def write_output(content, output_path, raw=False):
    """Write output to file or stdout."""
    # Handle None content
    if content is None:
        formatted_content = "No content returned from command"
        logger.warning("Command returned None")
    # Format the content if not raw
    elif not raw and isinstance(content, str):
        # Keep markdown formatting but avoid adding panels or other decoration
        formatted_content = content
    else:
        # Raw output - as is
        formatted_content = str(content)
    
    # Write to file or stdout
    if output_path:
        if output_path == "-":
            print(formatted_content)
        else:
            try:
                with open(output_path, "w") as f:
                    f.write(formatted_content)
            except Exception as e:
                logger.error(f"Error writing to output file: {e}")
                sys.exit(1)
    else:
        # Default to stdout
        print(formatted_content)