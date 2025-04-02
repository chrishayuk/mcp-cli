"""
Tests that verify compatibility with OpenAI's function calling pattern.
"""
import pytest
import json
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

# Enable asyncio tests
pytest_plugins = ['pytest_asyncio']

from mcp_cli.llm.tools_handler import (
    handle_tool_call,
    format_tool_response,
    fetch_tools,
    convert_to_openai_tools
)

class TestOpenAIFunctionCalling:
    """Tests specifically for OpenAI function calling compatibility."""
    
    @pytest.fixture
    def mock_server_streams(self):
        """Create a fixture for mock server streams."""
        read_stream = AsyncMock()
        write_stream = AsyncMock()
        return [(read_stream, write_stream)]
    
    @pytest.fixture
    def mock_conversation_history(self):
        """Create a fixture for conversation history."""
        return [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "What's the weather like in Paris today?"}
        ]
    
    @pytest.fixture
    def mock_weather_tool_response(self):
        """Create a fixture for a weather tool response."""
        return {
            "isError": False,
            "content": {"temperature": 14, "unit": "celsius"}
        }
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_call")
    async def test_openai_style_tool_call(self, mock_send_tools_call, 
                                         mock_server_streams, 
                                         mock_conversation_history,
                                         mock_weather_tool_response):
        """Test handling a tool call in OpenAI format."""
        # Setup mock
        mock_send_tools_call.return_value = mock_weather_tool_response
        
        # Create tool call exactly as it would come from OpenAI API
        tool_call = {
            "id": "call_12345xyz",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\":\"Paris, France\"}"
            }
        }
        
        # Call the function
        await handle_tool_call(tool_call, mock_conversation_history, mock_server_streams)
        
        # Verify tool call was sent with correct arguments
        mock_send_tools_call.assert_called_once()
        args = mock_send_tools_call.call_args.kwargs
        assert args["name"] == "get_weather"
        assert args["arguments"] == {"location": "Paris, France"}
        
        # Verify conversation history was updated with correct format
        assert len(mock_conversation_history) == 4  # Original 2 + tool call + tool response
        
        # Check the tool call entry
        tool_call_entry = mock_conversation_history[2]
        assert tool_call_entry["role"] == "assistant"
        assert tool_call_entry["content"] is None
        assert "tool_calls" in tool_call_entry
        
        # Instead of checking exact ID (which is generated), check the format
        tool_call_id = tool_call_entry["tool_calls"][0]["id"]
        assert tool_call_id.startswith("call_")
        
        # Check the tool response entry 
        tool_response_entry = mock_conversation_history[3]
        assert tool_response_entry["role"] == "tool"
        assert tool_response_entry["name"] == "get_weather"
        # The tool_call_id should match what was generated in the tool call entry
        assert tool_response_entry["tool_call_id"] == tool_call_id
        assert isinstance(tool_response_entry["content"], str)
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_call")
    async def test_multiple_tool_calls(self, mock_send_tools_call, 
                                      mock_server_streams, 
                                      mock_conversation_history):
        """Test handling multiple tool calls in a single response."""
        # Setup mocks to return different responses for different tools
        mock_send_tools_call.side_effect = [
            {"isError": False, "content": {"temperature": 14, "unit": "celsius"}},
            {"isError": False, "content": {"temperature": 18, "unit": "celsius"}},
            {"isError": False, "content": "Email sent successfully"}
        ]
        
        # Create multiple tool calls as would come from OpenAI
        tool_calls = [
            {
                "id": "call_12345xyz",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\":\"Paris, France\"}"
                }
            },
            {
                "id": "call_67890abc",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"location\":\"Bogotá, Colombia\"}"
                }
            },
            {
                "id": "call_99999def",
                "type": "function",
                "function": {
                    "name": "send_email",
                    "arguments": "{\"to\":\"bob@email.com\",\"body\":\"Hi bob\"}"
                }
            }
        ]
        
        # Process each tool call sequentially
        for tool_call in tool_calls:
            await handle_tool_call(tool_call, mock_conversation_history, mock_server_streams)
        
        # Verify all tool calls were executed
        assert mock_send_tools_call.call_count == 3
        
        # Verify conversation history contains all tool calls and responses
        assert len(mock_conversation_history) == 8  # Original 2 + (3 tool calls + 3 responses)
        
        # Check each tool call and response pair
        tool_call_ids = []
        for i in range(3):
            # Tool call entry
            tool_call_index = 2 + i*2
            tool_call_entry = mock_conversation_history[tool_call_index]
            assert tool_call_entry["role"] == "assistant"
            assert tool_call_entry["content"] is None
            
            # Store the tool call ID generated by the implementation
            tool_call_id = tool_call_entry["tool_calls"][0]["id"]
            tool_call_ids.append(tool_call_id)
            assert tool_call_id.startswith("call_")
            
            # Verify function name matches input
            assert tool_call_entry["tool_calls"][0]["function"]["name"] == tool_calls[i]["function"]["name"]
            
            # Tool response entry
            tool_response_index = 3 + i*2
            tool_response_entry = mock_conversation_history[tool_response_index]
            assert tool_response_entry["role"] == "tool"
            assert tool_response_entry["name"] == tool_calls[i]["function"]["name"]
            assert tool_response_entry["tool_call_id"] == tool_call_id
    
    @pytest.mark.asyncio
    async def test_convert_to_openai_tools_format(self):
        """Test conversion from MCP tools format to OpenAI tools format."""
        # Sample MCP tools
        mcp_tools = [
            {
                "name": "get_weather",
                "description": "Retrieves current weather for the given location.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogotá, Colombia"
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Units the temperature will be returned in."
                        }
                    },
                    "required": ["location", "units"]
                }
            }
        ]
        
        # Convert to OpenAI format
        openai_tools = convert_to_openai_tools(mcp_tools)
        
        # Verify the conversion is correct
        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "get_weather"
        
        # Check that parameters were preserved
        parameters = openai_tools[0]["function"]["parameters"]
        assert parameters["type"] == "object"
        assert "location" in parameters["properties"]
        assert "units" in parameters["properties"]
        assert parameters["properties"]["units"]["enum"] == ["celsius", "fahrenheit"]
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_call")
    async def test_handling_complex_arguments(self, mock_send_tools_call, 
                                            mock_server_streams, 
                                            mock_conversation_history):
        """Test handling complex nested arguments in function calls."""
        # Setup mock
        mock_send_tools_call.return_value = {
            "isError": False,
            "content": "Action completed successfully"
        }
        
        # Create tool call with complex nested JSON arguments
        tool_call = {
            "id": "call_complex_args",
            "type": "function",
            "function": {
                "name": "complex_action",
                "arguments": json.dumps({
                    "user": {
                        "id": 123,
                        "name": "John Doe",
                        "preferences": {
                            "theme": "dark",
                            "notifications": True
                        }
                    },
                    "items": [
                        {"id": 1, "name": "Item 1", "quantity": 2},
                        {"id": 2, "name": "Item 2", "quantity": 1}
                    ],
                    "options": {
                        "priority": "high",
                        "shipping": "express"
                    }
                })
            }
        }
        
        # Call the function
        await handle_tool_call(tool_call, mock_conversation_history, mock_server_streams)
        
        # Verify tool call was sent with correctly parsed complex arguments
        mock_send_tools_call.assert_called_once()
        args = mock_send_tools_call.call_args.kwargs
        assert args["name"] == "complex_action"
        
        # Check nested structures were preserved
        complex_args = args["arguments"]
        assert complex_args["user"]["id"] == 123
        assert complex_args["user"]["preferences"]["theme"] == "dark"
        assert len(complex_args["items"]) == 2
        assert complex_args["items"][1]["name"] == "Item 2"
        assert complex_args["options"]["shipping"] == "express"
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_list")
    async def test_fetch_tools_with_strict_parameters(self, mock_send_tools_list,
                                                    mock_server_streams):
        """Test fetching tools with strict mode schema."""
        # Create a tool schema that matches OpenAI's strict mode requirements
        strict_tool_schema = {
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Retrieves current weather for the given location.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and country e.g. Bogotá, Colombia"
                            },
                            "units": {
                                "type": ["string", "null"],
                                "enum": ["celsius", "fahrenheit"],
                                "description": "Units the temperature will be returned in."
                            }
                        },
                        "required": ["location", "units"],
                        "additionalProperties": False
                    }
                }
            ]
        }
        
        # Setup mock
        mock_send_tools_list.return_value = strict_tool_schema
        
        # Fetch tools
        read_stream, write_stream = mock_server_streams[0]
        tools = await fetch_tools(read_stream, write_stream)
        
        # Verify tools were fetched correctly
        assert len(tools) == 1
        assert tools[0]["name"] == "get_weather"
        
        # Convert to OpenAI format
        openai_tools = convert_to_openai_tools(tools)
        
        # Verify the conversion preserves strict mode properties
        assert "additionalProperties" in openai_tools[0]["function"]["parameters"]
        assert openai_tools[0]["function"]["parameters"]["additionalProperties"] is False
        assert "required" in openai_tools[0]["function"]["parameters"]
        
        # Verify the type array for optional nullable fields is preserved
        units_type = openai_tools[0]["function"]["parameters"]["properties"]["units"]["type"]
        assert isinstance(units_type, list)
        assert "null" in units_type