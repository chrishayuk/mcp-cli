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
    convert_to_openai_tools
)

class TestOpenAIFunctionCalling:
    """Tests specifically for OpenAI function calling compatibility."""
    
    @pytest.fixture
    def mock_stream_manager(self):
        """Create a fixture for mock StreamManager."""
        stream_manager = MagicMock()
        stream_manager.call_tool = AsyncMock()
        stream_manager.get_server_for_tool = MagicMock(return_value="test_server")
        return stream_manager
    
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
    async def test_openai_style_tool_call(self, 
                                         mock_stream_manager, 
                                         mock_conversation_history,
                                         mock_weather_tool_response):
        """Test handling a tool call in OpenAI format using StreamManager."""
        # Setup mock
        mock_stream_manager.call_tool.return_value = mock_weather_tool_response
        
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
        await handle_tool_call(tool_call, mock_conversation_history, stream_manager=mock_stream_manager)
        
        # Verify tool call was sent with correct arguments
        mock_stream_manager.call_tool.assert_called_once()
        args = mock_stream_manager.call_tool.call_args.kwargs
        assert args["tool_name"] == "get_weather"
        assert args["arguments"] == {"location": "Paris, France"}
        
        # Verify conversation history was updated with correct format
        assert len(mock_conversation_history) == 4  # Original 2 + tool call + tool response
        
        # Check the tool call entry
        tool_call_entry = mock_conversation_history[2]
        assert tool_call_entry["role"] == "assistant"
        assert tool_call_entry["content"] is None
        assert "tool_calls" in tool_call_entry
        
        # Tool call ID should match what was provided in the input
        tool_call_id = tool_call_entry["tool_calls"][0]["id"]
        assert tool_call_id == "call_12345xyz"
        
        # Check the tool response entry 
        tool_response_entry = mock_conversation_history[3]
        assert tool_response_entry["role"] == "tool"
        assert tool_response_entry["name"] == "get_weather"
        assert tool_response_entry["tool_call_id"] == tool_call_id
        assert isinstance(tool_response_entry["content"], str)
    
    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self, 
                                      mock_stream_manager, 
                                      mock_conversation_history):
        """Test handling multiple tool calls in a single response."""
        # Setup mocks to return different responses for different tools
        mock_stream_manager.call_tool.side_effect = [
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
            await handle_tool_call(tool_call, mock_conversation_history, stream_manager=mock_stream_manager)
        
        # Verify all tool calls were executed
        assert mock_stream_manager.call_tool.call_count == 3
        
        # Verify conversation history contains all tool calls and responses
        assert len(mock_conversation_history) == 8  # Original 2 + (3 tool calls + 3 responses)
        
        # Check each tool call and response pair
        for i in range(3):
            # Tool call entry
            tool_call_index = 2 + i*2
            tool_call_entry = mock_conversation_history[tool_call_index]
            assert tool_call_entry["role"] == "assistant"
            assert tool_call_entry["content"] is None
            
            # Tool call ID should match the input
            tool_call_id = tool_call_entry["tool_calls"][0]["id"]
            assert tool_call_id == tool_calls[i]["id"]
            
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
    async def test_handling_complex_arguments(self, 
                                            mock_stream_manager, 
                                            mock_conversation_history):
        """Test handling complex nested arguments in function calls."""
        # Setup mock
        mock_stream_manager.call_tool.return_value = {
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
        await handle_tool_call(tool_call, mock_conversation_history, stream_manager=mock_stream_manager)
        
        # Verify tool call was sent with correctly parsed complex arguments
        mock_stream_manager.call_tool.assert_called_once()
        args = mock_stream_manager.call_tool.call_args.kwargs
        assert args["tool_name"] == "complex_action"
        
        # Check nested structures were preserved
        complex_args = args["arguments"]
        assert complex_args["user"]["id"] == 123
        assert complex_args["user"]["preferences"]["theme"] == "dark"
        assert len(complex_args["items"]) == 2
        assert complex_args["items"][1]["name"] == "Item 2"
        assert complex_args["options"]["shipping"] == "express"
    
    @pytest.mark.asyncio
    async def test_namespaced_tools_compatibility(self):
        """Test that namespaced tools work with OpenAI's function calling pattern."""
        # Create tools with namespaced names
        namespaced_tools = [
            {
                "name": "Server1_get_weather",
                "description": "Retrieves current weather for the given location.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "Server2_get_weather",
                "description": "Retrieves weather from second server.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        ]
        
        # Convert to OpenAI format
        openai_tools = convert_to_openai_tools(namespaced_tools)
        
        # Verify namespaced names are preserved
        assert len(openai_tools) == 2
        assert openai_tools[0]["function"]["name"] == "Server1_get_weather"
        assert openai_tools[1]["function"]["name"] == "Server2_get_weather"
        
        # Create a mock stream manager
        stream_manager = MagicMock()
        stream_manager.call_tool = AsyncMock(return_value={
            "isError": False,
            "content": {"temperature": 14, "unit": "celsius"}
        })
        stream_manager.get_server_for_tool = MagicMock(return_value="Server1")
        
        # Create conversation history
        conversation_history = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "What's the weather like in Paris?"}
        ]
        
        # Create tool call with namespaced tool name
        tool_call = {
            "id": "call_12345xyz",
            "type": "function",
            "function": {
                "name": "Server1_get_weather",
                "arguments": "{\"location\":\"Paris, France\"}"
            }
        }
        
        # Call the function
        await handle_tool_call(tool_call, conversation_history, stream_manager=stream_manager)
        
        # Verify tool call was sent with correct tool name
        stream_manager.call_tool.assert_called_once()
        args = stream_manager.call_tool.call_args.kwargs
        assert args["tool_name"] == "Server1_get_weather"
        assert args["arguments"] == {"location": "Paris, France"}
        
        # Verify conversation history was updated correctly
        assert len(conversation_history) == 4
        
        # Check tool name is preserved in conversation history
        tool_call_entry = conversation_history[2]
        assert tool_call_entry["tool_calls"][0]["function"]["name"] == "Server1_get_weather"
        
        tool_response_entry = conversation_history[3]
        assert tool_response_entry["name"] == "Server1_get_weather"
    
    @pytest.mark.asyncio
    async def test_openai_error_handling(self, 
                                        mock_stream_manager, 
                                        mock_conversation_history):
        """Test error handling with OpenAI's function calling pattern."""
        # Setup mock to return an error
        mock_stream_manager.call_tool.return_value = {
            "isError": True,
            "error": "Tool execution failed",
            "content": "Error: Tool execution failed"
        }
        
        # Create tool call in OpenAI format
        tool_call = {
            "id": "call_12345xyz",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\":\"Invalid Location\"}"
            }
        }
        
        # Call the function
        await handle_tool_call(tool_call, mock_conversation_history, stream_manager=mock_stream_manager)
        
        # Verify tool call was sent
        mock_stream_manager.call_tool.assert_called_once()
        
        # Verify conversation history includes error response
        assert len(mock_conversation_history) == 4
        
        # Check tool call entry
        tool_call_entry = mock_conversation_history[2]
        assert tool_call_entry["role"] == "assistant"
        
        # Check error response
        tool_response_entry = mock_conversation_history[3]
        assert tool_response_entry["role"] == "tool"
        assert tool_response_entry["name"] == "get_weather"
        assert "Error" in tool_response_entry["content"]