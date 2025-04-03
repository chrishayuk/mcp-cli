"""
Tests for fetch_tools and convert_to_openai_tools functions.
"""
import pytest
import json
from unittest.mock import AsyncMock, patch

# Enable asyncio tests
pytest_plugins = ['pytest_asyncio']

from mcp_cli.llm.tools_handler import fetch_tools, convert_to_openai_tools

class TestFetchTools:
    """Tests for the fetch_tools function."""
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_list")
    async def test_fetch_tools_success(self, mock_send_tools_list):
        """Test successful fetching of tools."""
        # Mock tools list response
        tools_response = {
            "tools": [
                {"name": "tool1", "description": "First tool"},
                {"name": "tool2", "description": "Second tool"}
            ]
        }
        mock_send_tools_list.return_value = tools_response
        
        # Create mock streams
        read_stream = AsyncMock()
        write_stream = AsyncMock()
        
        # Call the function
        result = await fetch_tools(read_stream, write_stream)
        
        # Verify results
        assert result == tools_response["tools"]
        mock_send_tools_list.assert_called_once_with(
            read_stream=read_stream, write_stream=write_stream
        )
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_list")
    async def test_fetch_tools_empty(self, mock_send_tools_list):
        """Test fetching tools when none are returned."""
        # Mock empty tools response
        mock_send_tools_list.return_value = {"tools": []}
        
        # Create mock streams
        read_stream = AsyncMock()
        write_stream = AsyncMock()
        
        # Call the function
        result = await fetch_tools(read_stream, write_stream)
        
        # Verify results
        assert result == []
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_list")
    async def test_fetch_tools_invalid_format(self, mock_send_tools_list):
        """Test fetching tools with invalid format."""
        # Mock invalid tools response
        mock_send_tools_list.return_value = {"tools": "not a list"}
        
        # Create mock streams
        read_stream = AsyncMock()
        write_stream = AsyncMock()
        
        # Call the function
        result = await fetch_tools(read_stream, write_stream)
        
        # Verify results
        assert result is None
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_list")
    async def test_fetch_tools_with_unexpected_response(self, mock_send_tools_list):
        """Test fetching tools with unexpected response structure."""
        # Mock response without tools key
        mock_send_tools_list.return_value = {"something_else": []}
        
        # Create mock streams
        read_stream = AsyncMock()
        write_stream = AsyncMock()
        
        # Call the function
        result = await fetch_tools(read_stream, write_stream)
        
        # Empty list is a reasonable default
        assert result == []
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_list")
    async def test_fetch_tools_with_exception(self, mock_send_tools_list):
        """Test fetching tools when an exception occurs."""
        # Mock to raise exception
        mock_send_tools_list.side_effect = Exception("Connection error")
        
        # Create mock streams
        read_stream = AsyncMock()
        write_stream = AsyncMock()
        
        # Call should raise exception
        with pytest.raises(Exception) as excinfo:
            await fetch_tools(read_stream, write_stream)
        
        assert "Connection error" in str(excinfo.value)


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