"""
Tests for the handle_tool_call function.
"""
import pytest
import json
import asyncio
import uuid
from unittest.mock import AsyncMock
from mcp_cli.llm.tools_handler import handle_tool_call, parse_tool_response, format_tool_response

# A dummy stream manager that implements the required methods.
class DummyStreamManager:
    def __init__(self):
        # For testing purposes, these can be simple defaults.
        self._tools = [{"name": "testTool"}]
        self._server_info = [{"id": 1, "name": "DummyServer"}]
        self.tool_to_server_map = {"testTool": "DummyServer", "xmlTool": "DummyServer", "streamingTool": "DummyServer"}
    
    def get_internal_tools(self):
        # Return an empty list (or whatever is appropriate) since it's not used in tool call.
        return []
    
    def get_server_for_tool(self, tool_name: str) -> str:
        return self.tool_to_server_map.get(tool_name, "Unknown")
    
    async def call_tool(self, tool_name, arguments):
        # A basic placeholder; tests will replace this with an AsyncMock.
        return {"isError": False, "content": {"result": f"Success for {tool_name}"}}

# Fixtures for conversation history and responses.
@pytest.fixture
def dummy_stream_manager():
    return DummyStreamManager()

@pytest.fixture
def mock_conversation_history():
    """Create a fixture for conversation history."""
    return [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "User message"}
    ]

@pytest.fixture
def mock_tool_response():
    """Create a fixture for a successful tool response."""
    return {
        "isError": False,
        "content": {"result": "Success"}
    }

@pytest.fixture
def mock_error_response():
    """Create a fixture for a failed tool response."""
    return {
        "isError": True,
        "error": "Test error",
        "content": "Error: Test error"
    }

@pytest.fixture
def mock_streaming_response():
    """Create a fixture for a streaming tool response."""
    return {
        "isError": False,
        "content": [
            {"type": "text", "text": "Streaming chunk 1"},
            {"type": "text", "text": "Streaming chunk 2"},
            {"type": "text", "text": "Streaming chunk 3"}
        ]
    }

class TestHandleToolCall:
    """Tests for the handle_tool_call function."""
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_openai_format(self, dummy_stream_manager,
                                                   mock_conversation_history,
                                                   mock_tool_response):
        """Test handling a tool call in OpenAI format."""
        # Setup dummy_stream_manager.call_tool to return a successful response.
        dummy_stream_manager.call_tool = AsyncMock(return_value=mock_tool_response)
        
        # Create tool call in OpenAI format.
        tool_call = {
            "id": "call123",
            "type": "function",
            "function": {
                "name": "testTool",
                "arguments": json.dumps({"param": "value"})
            }
        }
        
        # Call the function.
        await handle_tool_call(tool_call, mock_conversation_history, None, dummy_stream_manager)
        
        # Verify that call_tool was called with correct arguments.
        dummy_stream_manager.call_tool.assert_called_once()
        call_kwargs = dummy_stream_manager.call_tool.call_args.kwargs
        assert call_kwargs["tool_name"] == "testTool"
        assert call_kwargs["arguments"] == {"param": "value"}
        
        # Verify conversation history was updated with two new entries:
        # one for the tool call and one for the tool response.
        assert len(mock_conversation_history) == 4
        tool_call_entry = mock_conversation_history[2]
        assert tool_call_entry["role"] == "assistant"
        assert tool_call_entry["content"] is None
        assert "tool_calls" in tool_call_entry
        
        tool_response_entry = mock_conversation_history[3]
        assert tool_response_entry["role"] == "tool"
        assert tool_response_entry["name"] == "testTool"
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_object_format(self, dummy_stream_manager,
                                                   mock_conversation_history,
                                                   mock_tool_response):
        """Test handling a tool call with object attribute access format."""
        dummy_stream_manager.call_tool = AsyncMock(return_value=mock_tool_response)
        
        # Create tool call with direct attributes.
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
        
        await handle_tool_call(tool_call, mock_conversation_history, None, dummy_stream_manager)
        
        dummy_stream_manager.call_tool.assert_called_once()
        call_kwargs = dummy_stream_manager.call_tool.call_args.kwargs
        assert call_kwargs["tool_name"] == "testTool"
        assert call_kwargs["arguments"] == {"param": "value"}
        
        # Conversation history updated.
        assert len(mock_conversation_history) == 4

    @pytest.mark.asyncio
    async def test_handle_tool_call_xml_format(self, dummy_stream_manager,
                                               mock_conversation_history,
                                               mock_tool_response):
        """Test handling a tool call parsed from XML format."""
        dummy_stream_manager.call_tool = AsyncMock(return_value=mock_tool_response)
        
        # Modify the last message to include an XML tool call.
        mock_conversation_history[-1]["content"] = 'Some message <function=xmlTool>{"param":"value"}</function>'
        
        # Create a tool call that does not have a "function" key,
        # so fallback to XML parsing is used.
        tool_call = {"not_a_function": "something"}
        
        await handle_tool_call(tool_call, mock_conversation_history, None, dummy_stream_manager)
        
        dummy_stream_manager.call_tool.assert_called_once()
        call_kwargs = dummy_stream_manager.call_tool.call_args.kwargs
        assert call_kwargs["tool_name"] == "xmlTool"
        assert call_kwargs["arguments"] == {"param": "value"}
        
        # Conversation history updated.
        assert len(mock_conversation_history) == 4

    @pytest.mark.asyncio
    async def test_handle_tool_call_multiple_servers(self, mock_conversation_history,
                                                      mock_tool_response, mock_error_response):
        """
        Test handling a tool call with simulated multiple server behavior.
        In the new implementation there is no automatic retry, so we expect a single call.
        """
        dummy_sm = DummyStreamManager()
        # Set up call_tool so that it would return an error if invoked.
        # (The new logic only makes a single call.)
        dummy_sm.call_tool = AsyncMock(return_value=mock_error_response)
        
        tool_call = {
            "function": {
                "name": "testTool",
                "arguments": json.dumps({"param": "value"})
            }
        }
        
        await handle_tool_call(tool_call, mock_conversation_history, None, dummy_sm)
        
        dummy_sm.call_tool.assert_called_once()
        # Expect conversation history to have two new entries reflecting the error response.
        assert len(mock_conversation_history) == 4
        tool_response_entry = mock_conversation_history[3]
        assert tool_response_entry["role"] == "tool"
        assert "Error:" in tool_response_entry["content"]
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_all_servers_fail(self, mock_conversation_history, mock_error_response):
        """Test handling a tool call when the server always returns an error."""
        dummy_sm = DummyStreamManager()
        dummy_sm.call_tool = AsyncMock(return_value=mock_error_response)
        
        tool_call = {
            "function": {
                "name": "testTool",
                "arguments": json.dumps({"param": "value"})
            }
        }
        
        await handle_tool_call(tool_call, mock_conversation_history, None, dummy_sm)
        
        dummy_sm.call_tool.assert_called_once()
        # In error case, conversation history has two new entries.
        assert len(mock_conversation_history) == 4

    @pytest.mark.asyncio
    async def test_handle_tool_call_streaming_response(self, dummy_stream_manager,
                                                       mock_conversation_history,
                                                       mock_streaming_response, monkeypatch):
        """Test handling a tool call with a streaming response."""
        dummy_stream_manager.call_tool = AsyncMock(return_value=mock_streaming_response)
        # Patch the formatter to return the expected string.
        monkeypatch.setattr(
            "mcp_cli.llm.tools_handler.format_tool_response",
            lambda content: "\n".join(chunk["text"] for chunk in content)
        )
        
        tool_call = {
            "function": {
                "name": "streamingTool",
                "arguments": json.dumps({"param": "value"})
            }
        }
        
        await handle_tool_call(tool_call, mock_conversation_history, None, dummy_stream_manager)
        
        dummy_stream_manager.call_tool.assert_called_once()
        # Expect the formatter to have been called indirectly.
        # Conversation history updated.
        assert len(mock_conversation_history) == 4
        tool_response_entry = mock_conversation_history[3]
        assert tool_response_entry["role"] == "tool"
        assert tool_response_entry["name"] == "streamingTool"
        assert tool_response_entry["content"] == "Streaming chunk 1\nStreaming chunk 2\nStreaming chunk 3"
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_with_exception(self, dummy_stream_manager, mock_conversation_history):
        """Test handling when call_tool raises an exception."""
        dummy_stream_manager.call_tool = AsyncMock(side_effect=Exception("Test exception"))
        
        tool_call = {
            "function": {
                "name": "testTool",
                "arguments": json.dumps({"param": "value"})
            }
        }
        
        # Call the function (it should catch the exception).
        await handle_tool_call(tool_call, mock_conversation_history, None, dummy_stream_manager)
        
        dummy_stream_manager.call_tool.assert_called_once()
        # On exception, assume that no new entries are appended.
        assert len(mock_conversation_history) == 2
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_with_json_decode_error(self, dummy_stream_manager, mock_conversation_history):
        """Test handling when the arguments cannot be decoded as JSON."""
        dummy_stream_manager.call_tool = AsyncMock(return_value={"isError": False, "content": "Success"})
        
        # Create a tool call with invalid JSON arguments.
        tool_call = {
            "function": {
                "name": "testTool",
                "arguments": "this is not json"
            }
        }
        
        await handle_tool_call(tool_call, mock_conversation_history, None, dummy_stream_manager)
        
        # Depending on the error-handling logic, conversation history may remain unchanged.
        # Here we expect no new entries to be added.
        assert len(mock_conversation_history) == 2
