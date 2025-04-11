"""
Tests for convert_to_openai_tools and other tool handler functions.
"""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

# Enable asyncio tests
pytest_plugins = ['pytest_asyncio']

from mcp_cli.llm.tools_handler import handle_tool_call, convert_to_openai_tools

class TestHandleToolCall:
    """Tests for the handle_tool_call function."""
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_with_stream_manager(self):
        """Test handling a tool call using StreamManager."""
        # Create mock StreamManager and conversation history
        stream_manager = MagicMock()
        stream_manager.call_tool = AsyncMock()
        stream_manager.call_tool.return_value = {
            "isError": False,
            "content": "Tool execution successful"
        }
        stream_manager.get_server_for_tool.return_value = "test_server"
        
        conversation_history = []
        
        # Create a tool call object (OpenAI format)
        tool_call = {
            "id": "call_test123",
            "function": {
                "name": "test_tool",
                "arguments": json.dumps({"param": "value"})
            }
        }
        
        # Call handle_tool_call with StreamManager
        await handle_tool_call(
            tool_call=tool_call,
            conversation_history=conversation_history,
            stream_manager=stream_manager
        )
        
        # Verify StreamManager was used correctly
        stream_manager.call_tool.assert_called_once_with(
            tool_name="test_tool",
            arguments={"param": "value"}
        )
        
        # Verify conversation history was updated correctly
        assert len(conversation_history) == 2
        assert conversation_history[0]["role"] == "assistant"
        assert conversation_history[0]["tool_calls"][0]["function"]["name"] == "test_tool"
        assert conversation_history[1]["role"] == "tool"
        assert conversation_history[1]["content"] == "Tool execution successful"
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_error_handling(self):
        """Test handling errors in tool calls."""
        # Create mock StreamManager that returns an error
        stream_manager = MagicMock()
        stream_manager.call_tool = AsyncMock()
        stream_manager.call_tool.return_value = {
            "isError": True,
            "error": "Test error",
            "content": "Error: Test error"
        }
        stream_manager.get_server_for_tool.return_value = "test_server"
        
        conversation_history = []
        
        # Create a tool call object
        tool_call = {
            "id": "call_test123",
            "function": {
                "name": "test_tool",
                "arguments": json.dumps({"param": "value"})
            }
        }
        
        # Call handle_tool_call
        await handle_tool_call(
            tool_call=tool_call,
            conversation_history=conversation_history,
            stream_manager=stream_manager
        )
        
        # Verify error was handled correctly
        assert len(conversation_history) == 2
        assert conversation_history[0]["role"] == "assistant"
        assert conversation_history[1]["role"] == "tool"
        assert "Error: Test error" in conversation_history[1]["content"]
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_missing_stream_manager(self):
        """Test handling a tool call without providing a StreamManager."""
        conversation_history = []
        
        # Create a tool call object
        tool_call = {
            "id": "call_test123",
            "function": {
                "name": "test_tool",
                "arguments": json.dumps({"param": "value"})
            }
        }
        
        # Call handle_tool_call without StreamManager
        await handle_tool_call(
            tool_call=tool_call,
            conversation_history=conversation_history,
            stream_manager=None
        )
        
        # Verify nothing was added to conversation history
        assert len(conversation_history) == 0
    
    @pytest.mark.asyncio
    async def test_handle_tool_call_with_object_attributes(self):
        """Test handling a tool call with object attributes instead of dict keys."""
        # Create mock StreamManager
        stream_manager = MagicMock()
        stream_manager.call_tool = AsyncMock()
        stream_manager.call_tool.return_value = {
            "isError": False,
            "content": "Tool execution successful"
        }
        stream_manager.get_server_for_tool.return_value = "test_server"
        
        conversation_history = []
        
        # Create a tool call object with attributes instead of dict keys
        class FunctionInfo:
            def __init__(self):
                self.name = "test_tool"
                self.arguments = json.dumps({"param": "value"})
                
        class ToolCall:
            def __init__(self):
                self.id = "call_test123"
                self.function = FunctionInfo()
        
        tool_call = ToolCall()
        
        # Call handle_tool_call
        await handle_tool_call(
            tool_call=tool_call,
            conversation_history=conversation_history,
            stream_manager=stream_manager
        )
        
        # Verify StreamManager was used correctly
        stream_manager.call_tool.assert_called_once_with(
            tool_name="test_tool",
            arguments={"param": "value"}
        )
        
        # Verify conversation history was updated correctly
        assert len(conversation_history) == 2
        assert conversation_history[0]["role"] == "assistant"
        assert conversation_history[0]["tool_calls"][0]["function"]["name"] == "test_tool"
        assert conversation_history[1]["role"] == "tool"
        assert conversation_history[1]["content"] == "Tool execution successful"


class TestConvertToOpenAITools:
    """Tests for the convert_to_openai_tools function."""
    
    def test_convert_to_openai_tools(self):
        """Test conversion of MCP tools format to OpenAI format."""
        # Sample MCP tools
        mcp_tools = [
            {
                "name": "tool1",
                "description": "First tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {"param1": {"type": "string"}}
                }
            },
            {
                "name": "tool2",
                "description": "Second tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {"param2": {"type": "number"}}
                }
            }
        ]
        
        # Convert to OpenAI format
        result = convert_to_openai_tools(mcp_tools)
        
        # Verify results
        assert len(result) == 2
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "tool1"
        assert result[0]["function"]["parameters"] == mcp_tools[0]["inputSchema"]
        assert result[1]["function"]["name"] == "tool2"
        
    def test_convert_tools_with_missing_schema(self):
        """Test conversion of tools with missing input schema."""
        mcp_tools = [
            {
                "name": "tool1",
                "description": "Tool with no schema"
            }
        ]
        
        result = convert_to_openai_tools(mcp_tools)
        
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "tool1"
        assert result[0]["function"]["parameters"] == {}
    
    def test_convert_empty_tools_list(self):
        """Test conversion of an empty tools list."""
        result = convert_to_openai_tools([])
        assert result == []
    
    def test_convert_tools_with_complex_schema(self):
        """Test conversion of tools with complex input schema."""
        mcp_tools = [
            {
                "name": "complexTool",
                "description": "Tool with complex schema",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "stringParam": {"type": "string", "description": "A string parameter"},
                        "numberParam": {"type": "number", "description": "A number parameter"},
                        "boolParam": {"type": "boolean", "description": "A boolean parameter"},
                        "arrayParam": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "An array parameter"
                        },
                        "objectParam": {
                            "type": "object",
                            "properties": {
                                "nestedParam": {"type": "string"}
                            },
                            "description": "An object parameter"
                        }
                    },
                    "required": ["stringParam"]
                }
            }
        ]
        
        result = convert_to_openai_tools(mcp_tools)
        
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "complexTool"
        # The complex schema should be preserved exactly as is
        assert result[0]["function"]["parameters"] == mcp_tools[0]["inputSchema"]
        
    def test_convert_namespaced_tools(self):
        """Test conversion of namespaced tools to OpenAI format."""
        # Sample MCP tools with namespaced names
        namespaced_tools = [
            {
                "name": "Server1_tool1",
                "description": "First tool from Server1",
                "inputSchema": {
                    "type": "object",
                    "properties": {"param1": {"type": "string"}}
                }
            },
            {
                "name": "Server2_tool2",
                "description": "Second tool from Server2",
                "inputSchema": {
                    "type": "object",
                    "properties": {"param2": {"type": "number"}}
                }
            }
        ]
        
        # Convert to OpenAI format
        result = convert_to_openai_tools(namespaced_tools)
        
        # Verify results - namespaced names should be preserved
        assert len(result) == 2
        assert result[0]["function"]["name"] == "Server1_tool1"
        assert result[1]["function"]["name"] == "Server2_tool2"