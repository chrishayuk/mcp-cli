# mcp_cli/llm/tools_handler.py
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

# Import CHUK tool registry for tool conversions

from mcp_cli.tools.manager import ToolManager
from mcp_cli.tools.models import ToolCallResult


def format_tool_response(response_content: list[dict[str, Any]] | Any) -> str:
    """Format the response content from a tool.

    Preserves structured data in a readable format, ensuring that all data is
    available for the model in future conversation turns.
    """
    # Handle list of dictionaries (likely structured data like SQL results)
    if (
        isinstance(response_content, list)
        and response_content
        and isinstance(response_content[0], dict)
    ):
        # Check if this looks like text records with type field
        if all(
            item.get("type") == "text" for item in response_content if "type" in item
        ):
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
            except (TypeError, ValueError):
                # Fallback if JSON serialization fails
                return str(response_content)
    elif isinstance(response_content, dict):
        # Single dictionary - return as JSON
        try:
            return json.dumps(response_content, indent=2)
        except (TypeError, ValueError):
            return str(response_content)
    else:
        # Default case - convert to string
        return str(response_content)


async def handle_tool_call(
    tool_call: dict[str, Any] | Any,
    conversation_history: list[dict[str, Any]],
    tool_manager: ToolManager,
) -> None:
    """
    Handle a single tool call using the centralized ToolManager.

    This function updates the conversation history with both the tool call and its response.

    Args:
        tool_call: The tool call object
        conversation_history: The conversation history to update
        tool_manager: ToolManager instance for executing tools
    """
    tool_name: str = "unknown_tool"
    tool_args: dict[str, Any] = {}
    tool_call_id: str | None = None

    try:
        # Extract tool call information
        if hasattr(tool_call, "function"):
            tool_name = tool_call.function.name
            raw_arguments = tool_call.function.arguments
            tool_call_id = getattr(tool_call, "id", None)
        elif isinstance(tool_call, dict) and "function" in tool_call:
            tool_name = tool_call["function"]["name"]
            raw_arguments = tool_call["function"]["arguments"]
            tool_call_id = tool_call.get("id")
        else:
            logging.error("Invalid tool call format")
            return

        # Ensure tool arguments are in dictionary form
        if isinstance(raw_arguments, str):
            try:
                tool_args = json.loads(raw_arguments)
            except json.JSONDecodeError:
                logging.error(f"Failed to parse tool arguments: {raw_arguments}")
                tool_args = {}
        else:
            tool_args = raw_arguments

        # Generate a unique tool call ID if not provided
        if not tool_call_id:
            tool_call_id = f"call_{tool_name}_{str(uuid.uuid4())[:8]}"

        # Log which tool we're calling
        if hasattr(tool_manager, "get_server_for_tool"):
            server_name = tool_manager.get_server_for_tool(tool_name)
            logging.debug(f"Calling tool '{tool_name}' on server '{server_name}'")

        # Call the tool using ToolManager
        result: ToolCallResult = await tool_manager.execute_tool(tool_name, tool_args)

        if not result.success:
            error_msg = result.error or "Unknown error"
            logging.debug(f"Error calling tool '{tool_name}': {error_msg}")

            # Add failed tool call to conversation history
            conversation_history.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args),
                            },
                        }
                    ],
                }
            )

            # Add error response
            conversation_history.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": f"Error: {error_msg}",
                    "tool_call_id": tool_call_id,
                }
            )
            return

        raw_content = result.result

        # Format the tool response
        formatted_response: str = format_tool_response(raw_content)
        logging.debug(f"Tool '{tool_name}' Response: {formatted_response}")

        # Append the tool call (for tracking purposes)
        conversation_history.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args),
                        },
                    }
                ],
            }
        )

        # Append the tool's response to the conversation history
        conversation_history.append(
            {
                "role": "tool",
                "name": tool_name,
                "content": formatted_response,
                "tool_call_id": tool_call_id,
            }
        )

    except Exception as e:
        logging.error(f"Error handling tool call '{tool_name}': {str(e)}")
