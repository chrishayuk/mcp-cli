# mcp_cli/llm/tools_handler.py
import json
import logging
import uuid
from typing import Any, Dict, Optional, List, Union

# Import CHUK tool registry for tool conversions
from chuk_tool_processor.registry.tool_export import openai_functions

from mcp_cli.tools.manager import ToolManager
from mcp_cli.tools.models import ToolCallResult


def format_tool_response(response_content: Union[List[Dict[str, Any]], Any]) -> str:
    """Format the response content from a tool.
    
    Preserves structured data in a readable format, ensuring that all data is
    available for the model in future conversation turns.
    """
    # Handle string input (could be JSON)
def format_tool_response(response_content: Union[List[Dict[str, Any]], Any]) -> str:
    """Format the response content from a tool."""
    # Handle dictionary with 'result' and 'content'
    if isinstance(response_content, dict) and 'result' in response_content:
        result = response_content['result']
        if isinstance(result, dict) and 'content' in result:
            content = result['content']
            if isinstance(content, list) and content and isinstance(content[0], dict):
                # Handle each content item
                formatted_contents = []
                for item in content:
                    if 'text' in item:
                        try:
                            # Try to parse text as JSON
                            parsed_text = json.loads(item['text'])
                            if isinstance(parsed_text, (dict, list)):
                                formatted_contents.append(json.dumps(parsed_text, indent=2, ensure_ascii=False))
                            else:
                                formatted_contents.append(str(parsed_text))
                        except (json.JSONDecodeError, TypeError):
                            formatted_contents.append(item['text'])
                    elif 'type' in item and item.get('type') == 'text' and 'text' in item:
                        formatted_contents.append(item['text'])
                return '\n'.join(formatted_contents) if formatted_contents else 'No content'
    
    # Handle string input (could be JSON)
    if isinstance(response_content, str):
        try:
            parsed = json.loads(response_content)
            if isinstance(parsed, str):
                return parsed
            return format_tool_response(parsed)
        except json.JSONDecodeError:
            return response_content
    
    # Handle list of dictionaries
    if isinstance(response_content, list) and response_content:
        try:
            return json.dumps(response_content, indent=2, ensure_ascii=False)
        except:
            pass
    
    # Handle dictionary
    if isinstance(response_content, dict):
        try:
            return json.dumps(response_content, indent=2, ensure_ascii=False)
        except:
            pass
    
    # Default case
    return str(response_content)

async def handle_tool_call(
    tool_call: Union[Dict[str, Any], Any],
    conversation_history: List[Dict[str, Any]],
    server_streams = None,  # Kept for backward compatibility but ignored
    stream_manager = None,  # Kept for backward compatibility but recommended to use tool_manager
    tool_manager: Optional[ToolManager] = None  # Preferred parameter
) -> None:
    """
    Handle a single tool call using the centralized ToolManager.

    This function updates the conversation history with both the tool call and its response.
    
    Args:
        tool_call: The tool call object
        conversation_history: The conversation history to update
        server_streams: Legacy parameter (ignored)
        stream_manager: Legacy StreamManager instance (optional)
        tool_manager: Preferred ToolManager instance
    """
    # Use tool_manager if provided, otherwise fall back to stream_manager
    manager = tool_manager or stream_manager
    
    if manager is None:
        logging.error("Either tool_manager or stream_manager is required for handle_tool_call")
        return
        
    tool_name: str = "unknown_tool"
    tool_args: Dict[str, Any] = {}
    tool_call_id: Optional[str] = None

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
        if hasattr(manager, 'get_server_for_tool'):
            server_name = manager.get_server_for_tool(tool_name)
            logging.debug(f"Calling tool '{tool_name}' on server '{server_name}'")
        
        # Call the tool using either manager
        if isinstance(manager, ToolManager):
            # Use ToolManager (preferred)
            result: ToolCallResult = await manager.execute_tool(tool_name, tool_args)
            
            if not result.success:
                error_msg = result.error or "Unknown error"
                logging.debug(f"Error calling tool '{tool_name}': {error_msg}")
                
                # Add failed tool call to conversation history
                conversation_history.append({
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
                })
                
                # Add error response
                conversation_history.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": f"Error: {error_msg}",
                    "tool_call_id": tool_call_id,
                })
                return
            
            raw_content = result.result
        else:
            # Use StreamManager (backward compatibility)
            tool_response = await manager.call_tool(tool_name, tool_args)
            
            if tool_response.get("isError"):
                error_msg = tool_response.get("error", "Unknown error")
                logging.debug(f"Error calling tool '{tool_name}': {error_msg}")
                
                # Handle error similar to above
                conversation_history.append({
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
                })
                
                conversation_history.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": f"Error: {error_msg}",
                    "tool_call_id": tool_call_id,
                })
                return
            
            raw_content = tool_response.get("content", [])
        
        # Format the tool response
        formatted_response: str = format_tool_response(raw_content)
        logging.debug(f"Tool '{tool_name}' Response: {formatted_response}")

        # Append the tool call (for tracking purposes)
        conversation_history.append({
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
        })

        # Append the tool's response to the conversation history
        conversation_history.append({
            "role": "tool",
            "name": tool_name,
            "content": formatted_response,
            "tool_call_id": tool_call_id,
        })

    except Exception as e:
        logging.error(f"Error handling tool call '{tool_name}': {str(e)}")

def convert_to_openai_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert a list of MCP-style tool metadata dictionaries into the
    OpenAI “function call” JSON schema.

    ⚠️  **Deprecated** – new code should call `ToolManager.get_tools_for_llm()`.
    This helper remains for older tests / scripts.
    """
    # Already-converted? → return unchanged
    if tools and isinstance(tools[0], dict) and tools[0].get("type") == "function":
        return tools

    openai_tools: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):  # skip un-recognised entries
            continue

        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", "unknown"),
                    # NEW: carry over the human-readable description
                    "description": tool.get("description", ""),
                    # Accept either `parameters` (already OpenAI-style) or
                    # legacy `inputSchema`
                    "parameters": tool.get("parameters", tool.get("inputSchema", {})),
                },
            }
        )

    return openai_tools
