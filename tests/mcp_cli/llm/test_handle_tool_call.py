"""
Tests for the handle_tool_call function.
"""
import pytest
import json
import asyncio
from unittest.mock import AsyncMock, patch

# Enable asyncio tests
pytest_plugins = ['pytest_asyncio']

from mcp_cli.llm.tools_handler import handle_tool_call

class TestHandleToolCall:
    """Tests for the handle_tool_call function."""
    
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
    def mock_tool_response(self):
        """Create a fixture for a successful tool response."""
        return {
            "isError": False,
            "content": {"result": "Success"}
        }
    
    @pytest.fixture
    def mock_error_response(self):
        """Create a fixture for a failed tool response."""
        return {
            "isError": True,
            "error": "Test error"
        }
    
    @pytest.fixture
    def mock_streaming_response(self):
        """Create a fixture for a streaming tool response."""
        return {
            "isError": False,
            "content": [
                {"type": "text", "text": "Streaming chunk 1"},
                {"type": "text", "text": "Streaming chunk 2"},
                {"type": "text", "text": "Streaming chunk 3"}
            ]
        }
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_call")
    async def test_handle_tool_call_openai_format(self, mock_send_tools_call, 
                                                mock_server_streams, 
                                                mock_conversation_history,
                                                mock_tool_response):
        """Test handling a tool call in OpenAI format."""
        # Setup mock
        mock_send_tools_call.return_value = mock_tool_response
        
        # Create tool call in OpenAI format
        tool_call = {
            "id": "call123",
            "type": "function",
            "function": {
                "name": "testTool",
                "arguments": json.dumps({"param": "value"})
            }
        }
        
        # Call the function
        await handle_tool_call(tool_call, mock_conversation_history, mock_server_streams)
        
        # Verify tool call was sent with correct arguments
        mock_send_tools_call.assert_called_once()
        args = mock_send_tools_call.call_args.kwargs
        assert args["name"] == "testTool"
        assert args["arguments"] == {"param": "value"}
        
        # Verify conversation history was updated
        assert len(mock_conversation_history) == 4  # Original 2 + tool call + tool response
        
        # Check the tool call entry
        tool_call_entry = mock_conversation_history[2]
        assert tool_call_entry["role"] == "assistant"
        assert tool_call_entry["content"] is None
        assert "tool_calls" in tool_call_entry
        
        # Check the tool response entry
        tool_response_entry = mock_conversation_history[3]
        assert tool_response_entry["role"] == "tool"
        assert tool_response_entry["name"] == "testTool"
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_call")
    async def test_handle_tool_call_object_format(self, mock_send_tools_call, 
                                                mock_server_streams, 
                                                mock_conversation_history,
                                                mock_tool_response):
        """Test handling a tool call with object attribute access format."""
        # Setup mock
        mock_send_tools_call.return_value = mock_tool_response
        
        # Create tool call with direct attributes (like from OpenAI client)
        class ToolFunction:
            def __init__(self):
                self.name = "testTool"
                self.arguments = json.dumps({"param": "value"})
        
        class ToolCall:
            def __init__(self):
                self.id = "call123"
                self.type = "function"
                self.function = ToolFunction()
        
        tool_call = ToolCall()
        
        # Call the function
        await handle_tool_call(tool_call, mock_conversation_history, mock_server_streams)
        
        # Verify tool call was sent with correct arguments
        mock_send_tools_call.assert_called_once()
        args = mock_send_tools_call.call_args.kwargs
        assert args["name"] == "testTool"
        assert args["arguments"] == {"param": "value"}
        
        # Verify conversation history was updated appropriately
        assert len(mock_conversation_history) == 4
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_call")
    async def test_handle_tool_call_xml_format(self, mock_send_tools_call, 
                                              mock_server_streams, 
                                              mock_conversation_history,
                                              mock_tool_response):
        """Test handling a tool call parsed from XML format."""
        # Setup mock
        mock_send_tools_call.return_value = mock_tool_response
        
        # Modify the last message to include XML tool call
        mock_conversation_history[-1]["content"] = "Some message <function=xmlTool>{\"param\":\"value\"}</function>"
        
        # Create a tool call that doesn't match the expected formats
        # This should trigger fallback to XML parsing
        tool_call = {"not_a_function": "something"}
        
        # Call the function
        await handle_tool_call(tool_call, mock_conversation_history, mock_server_streams)
        
        # Verify tool call was sent with correct arguments
        mock_send_tools_call.assert_called_once()
        args = mock_send_tools_call.call_args.kwargs
        assert args["name"] == "xmlTool"
        assert args["arguments"] == {"param": "value"}
        
        # Verify conversation history was updated appropriately
        assert len(mock_conversation_history) == 4
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_call")
    async def test_handle_tool_call_multiple_servers(self, mock_send_tools_call, 
                                                   mock_conversation_history,
                                                   mock_tool_response, 
                                                   mock_error_response):
        """Test handling a tool call with multiple servers, first fails."""
        # Setup mocks for two servers, first one fails
        mock_send_tools_call.side_effect = [mock_error_response, mock_tool_response]
        
        # Create two mock server streams
        server_streams = [
            (AsyncMock(), AsyncMock()),  # First server (will fail)
            (AsyncMock(), AsyncMock())   # Second server (will succeed)
        ]
        
        # Create tool call
        tool_call = {
            "function": {
                "name": "testTool",
                "arguments": {"param": "value"}
            }
        }
        
        # Call the function
        await handle_tool_call(tool_call, mock_conversation_history, server_streams)
        
        # Verify tool call was sent to both servers
        assert mock_send_tools_call.call_count == 2
        
        # Verify conversation history was updated with successful result
        assert len(mock_conversation_history) == 4
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_call")
    async def test_handle_tool_call_all_servers_fail(self, mock_send_tools_call, 
                                                   mock_conversation_history,
                                                   mock_error_response):
        """Test handling a tool call when all servers fail."""
        # Setup mocks for all servers to fail
        mock_send_tools_call.return_value = mock_error_response
        
        # Create two mock server streams
        server_streams = [
            (AsyncMock(), AsyncMock()),  # First server (will fail)
            (AsyncMock(), AsyncMock())   # Second server (will also fail)
        ]
        
        # Create tool call
        tool_call = {
            "function": {
                "name": "testTool",
                "arguments": {"param": "value"}
            }
        }
        
        # Call the function
        await handle_tool_call(tool_call, mock_conversation_history, server_streams)
        
        # Verify tool call was attempted on both servers
        assert mock_send_tools_call.call_count == 2
        
        # Verify conversation history was not updated (function returns early)
        assert len(mock_conversation_history) == 2
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_call")
    @patch("mcp_cli.llm.tools_handler.format_tool_response")
    async def test_handle_tool_call_streaming_response(self, mock_format_tool_response,
                                                     mock_send_tools_call, 
                                                     mock_server_streams, 
                                                     mock_conversation_history,
                                                     mock_streaming_response):
        """Test handling a tool call with streaming response."""
        # Setup mocks
        mock_send_tools_call.return_value = mock_streaming_response
        # Mock the formatter to return what we expect
        mock_format_tool_response.return_value = "Streaming chunk 1\nStreaming chunk 2\nStreaming chunk 3"
        
        # Create tool call
        tool_call = {
            "function": {
                "name": "streamingTool",
                "arguments": {"param": "value"}
            }
        }
        
        # Call the function
        await handle_tool_call(tool_call, mock_conversation_history, mock_server_streams)
        
        # Verify tool call was sent
        mock_send_tools_call.assert_called_once()
        
        # Verify format_tool_response was called with the streaming content
        mock_format_tool_response.assert_called_once_with(mock_streaming_response["content"])
        
        # Verify conversation history was updated
        assert len(mock_conversation_history) == 4
        
        # Check the tool response entry
        tool_response_entry = mock_conversation_history[3]
        assert tool_response_entry["role"] == "tool"
        assert tool_response_entry["name"] == "streamingTool"
        assert tool_response_entry["content"] == "Streaming chunk 1\nStreaming chunk 2\nStreaming chunk 3"
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_call")
    async def test_handle_tool_call_with_exception(self, mock_send_tools_call, 
                                                 mock_server_streams, 
                                                 mock_conversation_history):
        """Test handling when send_tools_call raises an exception."""
        # Setup mock to raise an exception
        mock_send_tools_call.side_effect = Exception("Test exception")
        
        # Create tool call
        tool_call = {
            "function": {
                "name": "testTool",
                "arguments": {"param": "value"}
            }
        }
        
        # Call the function - should not raise exception
        await handle_tool_call(tool_call, mock_conversation_history, mock_server_streams)
        
        # Verify function call was attempted
        mock_send_tools_call.assert_called_once()
        
        # Verify conversation history was not updated due to exception
        assert len(mock_conversation_history) == 2
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_with_json_decode_error(self, 
                                                         mock_server_streams, 
                                                         mock_conversation_history):
        """Test handling when the arguments cannot be decoded as JSON."""
        # Create tool call with invalid JSON
        tool_call = {
            "function": {
                "name": "testTool",
                "arguments": "this is not json"
            }
        }
        
        # Call the function directly - it should handle the JSON error gracefully
        await handle_tool_call(tool_call, mock_conversation_history, mock_server_streams)
        
        # Verify conversation history was not updated
        assert len(mock_conversation_history) == 2