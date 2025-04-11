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

# A dummy stream manager for use in these tests.
class DummyStreamManager:
    def __init__(self):
        self._tools = [{"name": "get_weather"}]
        self._server_info = [{"id": 1, "name": "DummyServer"}]
        self.tool_to_server_map = {"get_weather": "DummyServer"}
    
    def get_internal_tools(self):
        return []  # Not used in these tests.
    
    def get_server_for_tool(self, tool_name: str) -> str:
        return self.tool_to_server_map.get(tool_name, "Unknown")
    
    async def call_tool(self, tool_name, arguments):
        # This method will be replaced by an AsyncMock in the tests.
        return {"isError": False, "content": {"result": "Success"}}

class TestToolCallIdPreservation:
    """Tests specifically verifying that original tool call IDs are preserved."""
    
    @pytest.fixture
    def dummy_stream_manager(self):
        return DummyStreamManager()
    
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
    async def test_preserve_original_id(self, dummy_stream_manager, mock_conversation_history, mock_weather_tool_response):
        """Test that the original ID from OpenAI is preserved."""
        # Replace the call_tool method with an AsyncMock.
        dummy_stream_manager.call_tool = AsyncMock(return_value=mock_weather_tool_response)
        
        # Create tool call with a specific ID.
        original_id = "call_12345xyz"
        tool_call = {
            "id": original_id,
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\":\"Paris, France\"}"
            }
        }
        
        # Call the function.
        await handle_tool_call(tool_call, mock_conversation_history, None, dummy_stream_manager)
        
        # Verify the original ID was preserved in the tool call entry.
        tool_call_entry = mock_conversation_history[2]
        assert tool_call_entry["tool_calls"][0]["id"] == original_id
        
        # Verify the original ID was preserved in the tool response entry.
        tool_response_entry = mock_conversation_history[3]
        assert tool_response_entry["tool_call_id"] == original_id
    
    @pytest.mark.asyncio
    async def test_generate_id_when_missing(self, dummy_stream_manager, mock_conversation_history, mock_weather_tool_response):
        """Test that an ID is generated if one is not provided."""
        dummy_stream_manager.call_tool = AsyncMock(return_value=mock_weather_tool_response)
        
        # Create tool call without an ID.
        tool_call = {
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\":\"Paris, France\"}"
            }
        }
        
        # Call the function.
        await handle_tool_call(tool_call, mock_conversation_history, None, dummy_stream_manager)
        
        # Verify an ID was generated and used consistently.
        tool_call_entry = mock_conversation_history[2]
        generated_id = tool_call_entry["tool_calls"][0]["id"]
        assert generated_id is not None
        assert generated_id.startswith("call_")
        
        # Verify the generated ID was used in the tool response entry.
        tool_response_entry = mock_conversation_history[3]
        assert tool_response_entry["tool_call_id"] == generated_id
    
    # These tests are synchronous.
    def test_ollama_client_preserves_ids(self):
        """Test that the Ollama client preserves original IDs when available."""
        # Create mock classes to simulate an Ollama response.
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
        
        # Create a response with a tool call that has an ID.
        original_id = "call_original_abc123"
        tool_call = MockToolCall("get_weather", '{"location":"Paris"}', original_id)
        message = MockMessage("I'll check the weather", [tool_call])
        
        # Patch the ollama.chat function to return our mock response.
        with patch('ollama.chat', return_value=MockResponse(message)):
            client = OllamaLLMClient(model="test-model")
            result = client.create_completion(
                messages=[{"role": "user", "content": "Weather in Paris?"}],
                tools=[{"type": "function", "function": {"name": "get_weather"}}]
            )
            # Verify the original ID was preserved.
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["id"] == original_id
        
    def test_ollama_client_generates_ids_when_missing(self):
        """Test that the Ollama client generates IDs when they're not provided."""
        class MockFunctionCall:
            def __init__(self, name, args):
                self.name = name
                self.arguments = args
                # No ID attribute.
                
        class MockToolCall:
            def __init__(self, fn_name, fn_args):
                self.function = MockFunctionCall(fn_name, fn_args)
                # No ID attribute.
                
        class MockMessage:
            def __init__(self, content, tool_calls=None):
                self.content = content
                self.tool_calls = tool_calls
                
        class MockResponse:
            def __init__(self, message):
                self.message = message
        
        # Create a response with a tool call without an ID.
        tool_call = MockToolCall("get_weather", '{"location":"Paris"}')
        message = MockMessage("I'll check the weather", [tool_call])
        
        with patch('ollama.chat', return_value=MockResponse(message)):
            client = OllamaLLMClient(model="test-model")
            result = client.create_completion(
                messages=[{"role": "user", "content": "Weather in Paris?"}],
                tools=[{"type": "function", "function": {"name": "get_weather"}}]
            )
            # Verify an ID was generated.
            assert len(result["tool_calls"]) == 1
            generated_id = result["tool_calls"][0]["id"]
            assert generated_id is not None
            assert generated_id.startswith("call_")
