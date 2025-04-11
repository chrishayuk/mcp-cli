"""
Integration tests for the tools handler components with StreamManager.
"""
import pytest
import json
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

# Enable asyncio tests
pytest_plugins = ['pytest_asyncio']

from mcp_cli.llm.tools_handler import (
    convert_to_openai_tools,
    handle_tool_call,
    format_tool_response
)

class TestToolsIntegration:
    """Integration tests for the tools handler components with StreamManager."""
    
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
    @patch("mcp_cli.llm.tools_handler.format_tool_response")
    async def test_integration_complex_response(self, mock_format_tool_response, 
                                              mock_stream_manager, 
                                              mock_conversation_history,
                                              complex_tool_response):
        """Test full integration with a complex response and formatting using StreamManager."""
        # Setup mocks
        mock_stream_manager.call_tool.return_value = complex_tool_response
        # We want to test the actual formatting flow
        mock_format_tool_response.side_effect = format_tool_response
        
        # Create tool call
        tool_call = {
            "id": "call_123",
            "function": {
                "name": "complexTool",
                "arguments": {"param": "value"}
            }
        }
        
        # Call the function with StreamManager
        await handle_tool_call(tool_call, mock_conversation_history, stream_manager=mock_stream_manager)
        
        # Verify tool call was sent through StreamManager
        mock_stream_manager.call_tool.assert_called_once_with(
            tool_name="complexTool",
            arguments={"param": "value"}
        )
        
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
    async def test_full_tools_workflow_with_stream_manager(self, mock_stream_manager, mock_conversation_history):
        """Test the complete workflow using the StreamManager."""
        # Setup mock for tool call
        mock_stream_manager.call_tool.return_value = {
            "isError": False,
            "content": {"result": "Success"}
        }
        
        # Setup internal and display tools
        mock_stream_manager.get_internal_tools.return_value = [
            {
                "name": "Server1_testTool",
                "description": "Test tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {"param": {"type": "string"}}
                }
            }
        ]
        
        # Get the internal tools from the StreamManager
        tools = mock_stream_manager.get_internal_tools()
        
        # Verify tools were fetched
        assert len(tools) == 1
        assert tools[0]["name"] == "Server1_testTool"
        
        # Convert to OpenAI format
        openai_tools = convert_to_openai_tools(tools)
        
        # Verify conversion
        assert len(openai_tools) == 1
        assert openai_tools[0]["function"]["name"] == "Server1_testTool"
        
        # Create a tool call using the namespaced tool
        tool_call = {
            "id": "call_123",
            "function": {
                "name": "Server1_testTool",
                "arguments": json.dumps({"param": "test value"})
            }
        }
        
        # Execute the tool call with StreamManager
        await handle_tool_call(tool_call, mock_conversation_history, stream_manager=mock_stream_manager)
        
        # Verify tool call was executed through StreamManager
        mock_stream_manager.call_tool.assert_called_once_with(
            tool_name="Server1_testTool",
            arguments={"param": "test value"}
        )
        
        # Verify conversation history was updated
        assert len(mock_conversation_history) == 4
        
        # Verify last message has correct format and content
        last_message = mock_conversation_history[-1]
        assert last_message["role"] == "tool"
        assert last_message["name"] == "Server1_testTool"
    
    @pytest.mark.asyncio
    async def test_multiple_tool_calls_with_stream_manager(self, mock_stream_manager, mock_conversation_history):
        """Test multiple sequential tool calls in a conversation using StreamManager."""
        # Different responses for each tool call
        mock_stream_manager.call_tool.side_effect = [
            {"isError": False, "content": {"result": "First result"}},
            {"isError": False, "content": {"result": "Second result"}}
        ]
        
        # Create first tool call (with namespaced format)
        tool_call1 = {
            "id": "call_123",
            "function": {
                "name": "Server1_tool1",
                "arguments": {"param": "value1"}
            }
        }
        
        # Execute first tool call
        await handle_tool_call(tool_call1, mock_conversation_history, stream_manager=mock_stream_manager)
        
        # Verify first tool call was executed
        assert mock_stream_manager.call_tool.call_count == 1
        
        # Create second tool call (with namespaced format)
        tool_call2 = {
            "id": "call_456",
            "function": {
                "name": "Server2_tool2",
                "arguments": {"param": "value2"}
            }
        }
        
        # Execute second tool call
        await handle_tool_call(tool_call2, mock_conversation_history, stream_manager=mock_stream_manager)
        
        # Verify second tool call was executed
        assert mock_stream_manager.call_tool.call_count == 2
        
        # Verify conversation history contains both tool calls and responses
        assert len(mock_conversation_history) == 6  # Original 2 + 2 tool calls + 2 responses
        
        # Verify tool1 call and response
        assert mock_conversation_history[2]["role"] == "assistant"
        assert mock_conversation_history[2]["tool_calls"][0]["function"]["name"] == "Server1_tool1"
        assert mock_conversation_history[3]["role"] == "tool"
        assert mock_conversation_history[3]["name"] == "Server1_tool1"
        
        # Verify tool2 call and response
        assert mock_conversation_history[4]["role"] == "assistant"
        assert mock_conversation_history[4]["tool_calls"][0]["function"]["name"] == "Server2_tool2"
        assert mock_conversation_history[5]["role"] == "tool"
        assert mock_conversation_history[5]["name"] == "Server2_tool2"
    
    @pytest.mark.asyncio
    async def test_handling_tool_error_in_integration(self, mock_stream_manager, mock_conversation_history):
        """Test handling of tool errors in an integration context."""
        # Mock a tool error response
        mock_stream_manager.call_tool.return_value = {
            "isError": True,
            "error": "Database connection failed",
            "content": "Error: Database connection failed"
        }
        
        # Create a tool call
        tool_call = {
            "id": "call_error",
            "function": {
                "name": "Server1_dbQuery",
                "arguments": json.dumps({"query": "SELECT * FROM users"})
            }
        }
        
        # Execute the tool call
        await handle_tool_call(tool_call, mock_conversation_history, stream_manager=mock_stream_manager)
        
        # Verify the tool call was made
        mock_stream_manager.call_tool.assert_called_once()
        
        # Verify conversation history was updated with error
        assert len(mock_conversation_history) == 4
        
        # Verify tool call entry
        tool_call_entry = mock_conversation_history[2]
        assert tool_call_entry["role"] == "assistant"
        assert tool_call_entry["tool_calls"][0]["function"]["name"] == "Server1_dbQuery"
        
        # Verify error response
        error_entry = mock_conversation_history[3]
        assert error_entry["role"] == "tool"
        assert error_entry["name"] == "Server1_dbQuery"
        assert "Error" in error_entry["content"]
        assert "Database connection failed" in error_entry["content"]