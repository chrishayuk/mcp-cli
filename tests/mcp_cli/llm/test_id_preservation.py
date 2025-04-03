"""
Tests that verify tool call IDs are properly preserved.
"""
import pytest
import json
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

# Enable asyncio tests
pytest_plugins = ['pytest_asyncio']

from mcp_cli.llm.tools_handler import handle_tool_call
from mcp_cli.llm.providers.ollama_client import OllamaLLMClient

class TestToolCallIdPreservation:
    """Tests specifically verifying that original tool call IDs are preserved."""
    
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
    async def test_preserve_original_id(self, mock_send_tools_call, 
                                       mock_server_streams, 
                                       mock_conversation_history,
                                       mock_weather_tool_response):
        """Test that the original ID from OpenAI is preserved."""
        # Setup mock
        mock_send_tools_call.return_value = mock_weather_tool_response
        
        # Create tool call with a specific ID
        original_id = "call_12345xyz"
        tool_call = {
            "id": original_id,
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\":\"Paris, France\"}"
            }
        }
        
        # Call the function
        await handle_tool_call(tool_call, mock_conversation_history, mock_server_streams)
        
        # Verify the original ID was preserved in the tool call entry
        tool_call_entry = mock_conversation_history[2]
        assert tool_call_entry["tool_calls"][0]["id"] == original_id
        
        # Verify the original ID was preserved in the tool response entry
        tool_response_entry = mock_conversation_history[3]
        assert tool_response_entry["tool_call_id"] == original_id
    
    @pytest.mark.asyncio
    @patch("mcp_cli.llm.tools_handler.send_tools_call")
    async def test_generate_id_when_missing(self, mock_send_tools_call, 
                                           mock_server_streams, 
                                           mock_conversation_history,
                                           mock_weather_tool_response):
        """Test that an ID is generated if one is not provided."""
        # Setup mock
        mock_send_tools_call.return_value = mock_weather_tool_response
        
        # Create tool call without an ID
        tool_call = {
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\":\"Paris, France\"}"
            }
        }
        
        # Call the function
        await handle_tool_call(tool_call, mock_conversation_history, mock_server_streams)
        
        # Verify an ID was generated and used consistently
        tool_call_entry = mock_conversation_history[2]
        generated_id = tool_call_entry["tool_calls"][0]["id"]
        assert generated_id is not None
        assert generated_id.startswith("call_")
        
        # Verify the generated ID was used in the tool response entry
        tool_response_entry = mock_conversation_history[3]
        assert tool_response_entry["tool_call_id"] == generated_id
    
    # Remove the asyncio decorator as it's not needed for synchronous tests
    def test_ollama_client_preserves_ids(self):
        """Test that the Ollama client preserves original IDs when available."""
        # Create a mock response from Ollama with a tool call
        class MockFunctionCall:
            def __init__(self, name, args, id=None):
                self.name = name
                self.arguments = args
                self.id = id
                
        class MockToolCall:
            def __init__(self, fn_name, fn_args, id=None):
                self.function = MockFunctionCall(fn_name, fn_args, id)
                self.id = id
                
        class MockMessage:
            def __init__(self, content, tool_calls=None):
                self.content = content
                self.tool_calls = tool_calls
                
        class MockResponse:
            def __init__(self, message):
                self.message = message
        
        # Create a response with a tool call that has an ID
        original_id = "call_original_abc123"
        tool_call = MockToolCall("get_weather", '{"location":"Paris"}', original_id)
        message = MockMessage("I'll check the weather", [tool_call])
        
        # Patch the ollama.chat function
        with patch('ollama.chat', return_value=MockResponse(message)):
            # Create the client and make a request
            client = OllamaLLMClient(model="test-model")
            result = client.create_completion(
                messages=[{"role": "user", "content": "Weather in Paris?"}],
                tools=[{"type": "function", "function": {"name": "get_weather"}}]
            )
            
            # Verify the original ID was preserved
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["id"] == original_id
        
    # Remove the asyncio decorator as it's not needed for synchronous tests
    def test_ollama_client_generates_ids_when_missing(self):
        """Test that the Ollama client generates IDs when they're not provided."""
        # Create a mock response from Ollama with a tool call that has no ID
        class MockFunctionCall:
            def __init__(self, name, args):
                self.name = name
                self.arguments = args
                # No ID attribute
                
        class MockToolCall:
            def __init__(self, fn_name, fn_args):
                self.function = MockFunctionCall(fn_name, fn_args)
                # No ID attribute
                
        class MockMessage:
            def __init__(self, content, tool_calls=None):
                self.content = content
                self.tool_calls = tool_calls
                
        class MockResponse:
            def __init__(self, message):
                self.message = message
        
        # Create a response with a tool call without an ID
        tool_call = MockToolCall("get_weather", '{"location":"Paris"}')
        message = MockMessage("I'll check the weather", [tool_call])
        
        # Patch the ollama.chat function
        with patch('ollama.chat', return_value=MockResponse(message)):
            # Create the client and make a request
            client = OllamaLLMClient(model="test-model")
            result = client.create_completion(
                messages=[{"role": "user", "content": "Weather in Paris?"}],
                tools=[{"type": "function", "function": {"name": "get_weather"}}]
            )
            
            # Verify an ID was generated
            assert len(result["tool_calls"]) == 1
            generated_id = result["tool_calls"][0]["id"]
            assert generated_id is not None
            assert generated_id.startswith("call_")