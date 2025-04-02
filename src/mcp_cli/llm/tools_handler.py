# mcp_cli/llm/tools_handler.py
import json
import logging
import re
import uuid
from typing import Any, Dict, Optional, List, Tuple, Union

# Imports from your application for sending tool messages
from chuk_mcp.mcp_client.messages.tools.send_messages import send_tools_call, send_tools_list

def parse_tool_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse tool call from Llama's XML-style format.

    Expected format:
      <function=FUNCTION_NAME>{"arg1": "value1", ...}</function>

    Returns a dictionary with 'function' and 'arguments' keys if found.
    """
    function_regex = r"<function=(\w+)>(.*?)</function>"
    match = re.search(function_regex, response)
    if match:
        function_name, args_string = match.groups()
        try:
            args = json.loads(args_string)
            return {"function": function_name, "arguments": args}
        except json.JSONDecodeError as error:
            logging.debug(f"Error parsing function arguments: {error}")
    return None


def format_tool_response(response_content: Union[List[Dict[str, Any]], Any]) -> str:
    """Format the response content from a tool.
    
    Preserves structured data in a readable format, ensuring that all data is
    available for the model in future conversation turns.
    """
    # Handle list of dictionaries (likely structured data like SQL results)
    if isinstance(response_content, list) and response_content and isinstance(response_content[0], dict):
        # Check if this looks like text records with type field
        if all(item.get("type") == "text" for item in response_content if "type" in item):
            # Text records - extract just the text
            return "\n".join(
                item.get("text", "No content")
                for item in response_content
                if item.get("type") == "text"
            )
        else:
            # This could be data records (like SQL results)
            # Return a JSON representation that preserves all data
            try:
                return json.dumps(response_content, indent=2)
            except:
                # Fallback if JSON serialization fails
                return str(response_content)
    elif isinstance(response_content, dict):
        # Single dictionary - return as JSON
        try:
            return json.dumps(response_content, indent=2)
        except:
            return str(response_content)
    else:
        # Default case - convert to string
        return str(response_content)


"""
Updated handle_tool_call function to preserve the original tool call IDs.
"""

async def handle_tool_call(
    tool_call: Union[Dict[str, Any], Any],
    conversation_history: List[Dict[str, Any]],
    server_streams: List[Tuple[Any, Any]]
) -> None:
    """
    Handle a single tool call for both OpenAI and Llama formats.

    This function updates the conversation history with both the tool call and its response.
    """
    tool_name: str = "unknown_tool"
    raw_arguments: Any = {}
    tool_call_id: str = None

    try:
        # Support for object-style tool calls from both OpenAI and the new Ollama function tools.
        if hasattr(tool_call, "function") or (isinstance(tool_call, dict) and "function" in tool_call):
            if hasattr(tool_call, "function"):
                tool_name = tool_call.function.name
                raw_arguments = tool_call.function.arguments
                # Extract ID if available (OpenAI format)
                tool_call_id = getattr(tool_call, "id", None)
            else:
                tool_name = tool_call["function"]["name"]
                raw_arguments = tool_call["function"]["arguments"]
                # Extract ID if available (OpenAI format)
                tool_call_id = tool_call.get("id")
        else:
            # Fallback: attempt to parse Llama's XML format from the last message in history.
            last_message = conversation_history[-1]["content"]
            parsed_tool = parse_tool_response(last_message)
            if not parsed_tool:
                logging.debug("Unable to parse tool call from message")
                return
            tool_name = parsed_tool["function"]
            raw_arguments = parsed_tool["arguments"]

        # Ensure tool arguments are in dictionary form.
        tool_args: Dict[str, Any] = (
            json.loads(raw_arguments)
            if isinstance(raw_arguments, str)
            else raw_arguments
        )

        # Generate a unique tool call ID if one wasn't provided
        if not tool_call_id:
            tool_call_id = f"call_{tool_name}_{str(uuid.uuid4())[:8]}"

        # Execute the tool via server streams.
        tool_response: Dict[str, Any] = {}
        for read_stream, write_stream in server_streams:
            # Correct parameter order: streams first, then tool name and arguments.
            tool_response = await send_tools_call(
                read_stream=read_stream,
                write_stream=write_stream,
                name=tool_name,
                arguments=tool_args
            )
            if not tool_response.get("isError"):
                break

        if tool_response.get("isError"):
            logging.debug(f"Error calling tool '{tool_name}': {tool_response.get('error')}")
            return

        # Get the raw content from the response
        raw_content = tool_response.get("content", [])
        
        # Format the tool response.
        formatted_response: str = format_tool_response(raw_content)
        logging.debug(f"Tool '{tool_name}' Response: {formatted_response}")

        # Append the tool call (for tracking purposes).
        conversation_history.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args)
                        if isinstance(tool_args, dict)
                        else tool_args,
                    },
                }
            ],
        })

        # Append the tool's response to the conversation history.
        conversation_history.append({
            "role": "tool",
            "name": tool_name,
            "content": formatted_response,
            "tool_call_id": tool_call_id,
        })

    except json.JSONDecodeError:
        logging.debug(f"Error decoding arguments for tool '{tool_name}': {raw_arguments}")
    except Exception as e:
        logging.debug(f"Error handling tool call '{tool_name}': {str(e)}")
        
async def fetch_tools(read_stream: Any, write_stream: Any) -> Optional[List[Dict[str, Any]]]:
    """Fetch tools from the server."""
    logging.debug("\nFetching tools for chat mode...")

    # Get the tools list by passing read_stream and write_stream as named parameters.
    tools_response = await send_tools_list(read_stream=read_stream, write_stream=write_stream)
    tools = tools_response.get("tools", [])

    # Validate that the tools are in the expected format.
    if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
        logging.debug("Invalid tools format received.")
        return None

    return tools


def convert_to_openai_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert tools into OpenAI-compatible function definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "parameters": tool.get("inputSchema", {}),
            },
        }
        for tool in tools
    ]