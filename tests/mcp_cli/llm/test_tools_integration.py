"""
Integration tests for the tools handler components.
"""
import pytest
import json
import asyncio
from unittest.mock import AsyncMock, patch

# Enable asyncio tests
pytest_plugins = ['pytest_asyncio']

from mcp_cli.llm.tools_handler import (
    fetch_tools,
    convert_to_openai_tools,
    handle_tool_call,
    format_tool_response
)

class TestToolsIntegration:
    """Integration tests for the tools handler components."""
    
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
            {"role": "user", "content": "User message"}
        ]
    
    @pytest.fixture
    def complex_tool_response(self):
        """Create a fixture for a complex tool response with different data types."""
        return {
            "isError": False,
            "content": [
                {
                    "type": "result",
                    "rows": [
                        {"id": 1, "name": "Item 1", "value": 100},
                        {"id": 2, "name": "Item 2", "value": 200}
                    ],
                    "metadata": {
                        "count": 2,
                        "schema": ["id", "name", "value"]
                    }
                },
                {
                    "type": "error",
                    "message": "Warning: partial results",
                    "code": "PARTIAL_RESULTS"
                }
            ]
        }
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_call")
    @patch("mcp_cli.llm.tools_handler.format_tool_response")
    async def test_integration_complex_response(self, mock_format_tool_response, 
                                              mock_send_tools_call, 
                                              mock_server_streams, 
                                              mock_conversation_history,
                                              complex_tool_response):
        """Test full integration with a complex response and formatting."""
        # Setup mocks
        mock_send_tools_call.return_value = complex_tool_response
        # We want to test the actual formatting flow
        mock_format_tool_response.side_effect = format_tool_response
        
        # Create tool call
        tool_call = {
            "function": {
                "name": "complexTool",
                "arguments": {"param": "value"}
            }
        }
        
        # Call the function
        await handle_tool_call(tool_call, mock_conversation_history, mock_server_streams)
        
        # Verify tool call was sent
        mock_send_tools_call.assert_called_once()
        
        # Verify the format_tool_response function was called with the complex response
        mock_format_tool_response.assert_called_once_with(complex_tool_response["content"])
        
        # Verify conversation history was updated
        assert len(mock_conversation_history) == 4
        
        # Check the tool call entry
        tool_call_entry = mock_conversation_history[2]
        assert tool_call_entry["role"] == "assistant"
        assert tool_call_entry["content"] is None
        assert "tool_calls" in tool_call_entry
        
        # Check the tool response entry
        tool_response_entry = mock_conversation_history[3]
        assert tool_response_entry["role"] == "tool"
        assert tool_response_entry["name"] == "complexTool"
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_list")
    @patch("mcp_cli.llm.tools_handler.send_tools_call")
    async def test_full_tools_workflow(self, mock_send_tools_call, mock_send_tools_list,
                                     mock_server_streams, mock_conversation_history):
        """Test the complete workflow from fetching tools to executing them."""
        # Setup mocks for tools list and tool call
        mock_send_tools_list.return_value = {
            "tools": [
                {
                    "name": "testTool",
                    "description": "Test tool",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"param": {"type": "string"}}
                    }
                }
            ]
        }
        
        mock_send_tools_call.return_value = {
            "isError": False,
            "content": {"result": "Success"}
        }
        
        # First fetch tools
        tools = await fetch_tools(mock_server_streams[0][0], mock_server_streams[0][1])
        
        # Verify tools were fetched
        assert len(tools) == 1
        assert tools[0]["name"] == "testTool"
        
        # Convert to OpenAI format
        openai_tools = convert_to_openai_tools(tools)
        
        # Verify conversion
        assert len(openai_tools) == 1
        assert openai_tools[0]["function"]["name"] == "testTool"
        
        # Create a tool call using the fetched tool
        tool_call = {
            "function": {
                "name": "testTool",
                "arguments": json.dumps({"param": "test value"})
            }
        }
        
        # Execute the tool call
        await handle_tool_call(tool_call, mock_conversation_history, mock_server_streams)
        
        # Verify tool call was executed
        mock_send_tools_call.assert_called_once_with(
            read_stream=mock_server_streams[0][0],
            write_stream=mock_server_streams[0][1],
            name="testTool",
            arguments={"param": "test value"}
        )
        
        # Verify conversation history was updated
        assert len(mock_conversation_history) == 4
        
        # Verify last message has correct format and content
        last_message = mock_conversation_history[-1]
        assert last_message["role"] == "tool"
        assert last_message["name"] == "testTool"
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_list")
    @patch("mcp_cli.llm.tools_handler.send_tools_call")
    async def test_multiple_tool_calls(self, mock_send_tools_call, mock_send_tools_list,
                                     mock_server_streams, mock_conversation_history):
        """Test multiple sequential tool calls in a conversation."""
        # Setup mocks
        mock_send_tools_list.return_value = {
            "tools": [
                {"name": "tool1", "description": "First tool"},
                {"name": "tool2", "description": "Second tool"}
            ]
        }
        
        # Different responses for each tool call
        mock_send_tools_call.side_effect = [
            {"isError": False, "content": {"result": "First result"}},
            {"isError": False, "content": {"result": "Second result"}}
        ]
        
        # Fetch tools
        tools = await fetch_tools(mock_server_streams[0][0], mock_server_streams[0][1])
        
        # Create first tool call
        tool_call1 = {
            "function": {
                "name": "tool1",
                "arguments": {"param": "value1"}
            }
        }
        
        # Execute first tool call
        await handle_tool_call(tool_call1, mock_conversation_history, mock_server_streams)
        
        # Verify first tool call was executed
        assert mock_send_tools_call.call_count == 1
        
        # Create second tool call
        tool_call2 = {
            "function": {
                "name": "tool2",
                "arguments": {"param": "value2"}
            }
        }
        
        # Execute second tool call
        await handle_tool_call(tool_call2, mock_conversation_history, mock_server_streams)
        
        # Verify second tool call was executed
        assert mock_send_tools_call.call_count == 2
        
        # Verify conversation history contains both tool calls and responses
        assert len(mock_conversation_history) == 6  # Original 2 + 2 tool calls + 2 responses
        
        # Verify tool1 call and response
        assert mock_conversation_history[2]["role"] == "assistant"
        assert mock_conversation_history[2]["tool_calls"][0]["function"]["name"] == "tool1"
        assert mock_conversation_history[3]["role"] == "tool"
        assert mock_conversation_history[3]["name"] == "tool1"
        
        # Verify tool2 call and response
        assert mock_conversation_history[4]["role"] == "assistant"
        assert mock_conversation_history[4]["tool_calls"][0]["function"]["name"] == "tool2"
        assert mock_conversation_history[5]["role"] == "tool"
        assert mock_conversation_history[5]["name"] == "tool2"