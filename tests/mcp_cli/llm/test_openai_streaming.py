"""
Tests for OpenAI-style streaming tool calls.
"""
import pytest
import json
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

# Enable asyncio tests
pytest_plugins = ['pytest_asyncio']

class TestOpenAIStreaming:
    """Tests for handling OpenAI-style streaming tool calls."""
    
    @pytest.fixture
    def mock_streaming_chunks(self):
        """Create mock streaming chunks for tool calls."""
        # This simulates how OpenAI returns streaming chunks for tool calls
        return [
            # First chunk establishes the tool call with id, type and function name
            {"tool_calls": [{"index": 0, "id": "call_12345xyz", "function": {"arguments": "", "name": "get_weather"}, "type": "function"}]},
            # Subsequent chunks provide arguments piece by piece
            {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": "{\"", "name": None}, "type": None}]},
            {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": "location", "name": None}, "type": None}]},
            {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": "\":\"", "name": None}, "type": None}]},
            {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": "Paris", "name": None}, "type": None}]},
            {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": ",", "name": None}, "type": None}]},
            {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": " France", "name": None}, "type": None}]},
            {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": "\"}", "name": None}, "type": None}]},
            # Final chunk with no tool_calls indicates completion
            {"tool_calls": None}
        ]
    
    def test_accumulate_streaming_tool_calls(self, mock_streaming_chunks):
        """Test accumulating streaming tool call deltas into a final tool_calls object."""
        # This test replicates the behavior shown in OpenAI's documentation
        final_tool_calls = {}
        
        for chunk in mock_streaming_chunks:
            for tool_call in chunk.get("tool_calls") or []:
                index = tool_call.get("index")
                
                if index is not None:
                    if index not in final_tool_calls:
                        final_tool_calls[index] = {
                            "id": tool_call.get("id"),
                            "type": tool_call.get("type"),
                            "function": {
                                "name": tool_call.get("function", {}).get("name"),
                                "arguments": tool_call.get("function", {}).get("arguments", "")
                            }
                        }
                    else:
                        # Append to existing arguments
                        if tool_call.get("function") and "arguments" in tool_call.get("function", {}):
                            final_tool_calls[index]["function"]["arguments"] += tool_call["function"]["arguments"]
        
        # Verify the accumulated tool call
        assert 0 in final_tool_calls
        assert final_tool_calls[0]["id"] == "call_12345xyz"
        assert final_tool_calls[0]["type"] == "function"
        assert final_tool_calls[0]["function"]["name"] == "get_weather"
        assert final_tool_calls[0]["function"]["arguments"] == "{\"location\":\"Paris, France\"}"
        
        # Parse the arguments to verify they're valid JSON
        parsed_args = json.loads(final_tool_calls[0]["function"]["arguments"])
        assert parsed_args["location"] == "Paris, France"
    
    def test_multiple_streaming_tool_calls(self):
        """Test handling multiple tool calls in a streaming response."""
        # Create chunks with two different tool calls
        streaming_chunks = [
            # First tool call initialization
            {"tool_calls": [{"index": 0, "id": "call_weather", "function": {"arguments": "", "name": "get_weather"}, "type": "function"}]},
            {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": "{\"location\":\"Paris\"}", "name": None}, "type": None}]},
            
            # Second tool call initialization (can come in same or different chunk)
            {"tool_calls": [{"index": 1, "id": "call_email", "function": {"arguments": "", "name": "send_email"}, "type": "function"}]},
            {"tool_calls": [{"index": 1, "id": None, "function": {"arguments": "{\"to\":\"bob@example.com\"}", "name": None}, "type": None}]},
            
            # Final chunk
            {"tool_calls": None}
        ]
        
        # Accumulate the tool calls
        final_tool_calls = {}
        
        for chunk in streaming_chunks:
            for tool_call in chunk.get("tool_calls") or []:
                index = tool_call.get("index")
                
                if index is not None:
                    if index not in final_tool_calls:
                        final_tool_calls[index] = {
                            "id": tool_call.get("id"),
                            "type": tool_call.get("type"),
                            "function": {
                                "name": tool_call.get("function", {}).get("name"),
                                "arguments": tool_call.get("function", {}).get("arguments", "")
                            }
                        }
                    else:
                        # Append to existing arguments
                        if tool_call.get("function") and "arguments" in tool_call.get("function", {}):
                            final_tool_calls[index]["function"]["arguments"] += tool_call["function"]["arguments"]
        
        # Verify we accumulated both tool calls correctly
        assert len(final_tool_calls) == 2
        
        # Check weather tool call
        assert final_tool_calls[0]["id"] == "call_weather"
        assert final_tool_calls[0]["function"]["name"] == "get_weather"
        assert final_tool_calls[0]["function"]["arguments"] == "{\"location\":\"Paris\"}"
        
        # Check email tool call
        assert final_tool_calls[1]["id"] == "call_email"
        assert final_tool_calls[1]["function"]["name"] == "send_email"
        assert final_tool_calls[1]["function"]["arguments"] == "{\"to\":\"bob@example.com\"}"
    
    @pytest.mark.asyncio
    async def test_streaming_helper_function(self):
        """Test a helper function that processes streaming tool calls."""
        # This test demonstrates a helpful utility function that could be used in your application
        
        async def process_streaming_tool_calls(chunks):
            """
            Process streaming chunks and return accumulated tool calls.
            
            Args:
                chunks: Iterator of streaming chunks
                
            Returns:
                Dict mapping tool call indices to complete tool call objects
            """
            final_tool_calls = {}
            
            for chunk in chunks:
                for tool_call in chunk.get("tool_calls") or []:
                    index = tool_call.get("index")
                    
                    if index is not None:
                        if index not in final_tool_calls:
                            final_tool_calls[index] = {
                                "id": tool_call.get("id"),
                                "type": tool_call.get("type"),
                                "function": {
                                    "name": tool_call.get("function", {}).get("name"),
                                    "arguments": tool_call.get("function", {}).get("arguments", "")
                                }
                            }
                        else:
                            # Update any non-None fields
                            if tool_call.get("id") is not None:
                                final_tool_calls[index]["id"] = tool_call["id"]
                            if tool_call.get("type") is not None:
                                final_tool_calls[index]["type"] = tool_call["type"]
                            if tool_call.get("function") and tool_call["function"].get("name") is not None:
                                final_tool_calls[index]["function"]["name"] = tool_call["function"]["name"]
                            # Always append arguments
                            if tool_call.get("function") and "arguments" in tool_call.get("function", {}):
                                final_tool_calls[index]["function"]["arguments"] += tool_call["function"]["arguments"]
            
            # Parse JSON arguments into Python objects for easier use
            for index, tool_call in final_tool_calls.items():
                try:
                    args_str = tool_call["function"]["arguments"]
                    if args_str:
                        parsed_args = json.loads(args_str)
                        # Replace the JSON string with parsed object
                        tool_call["function"]["parsed_arguments"] = parsed_args
                except json.JSONDecodeError:
                    tool_call["function"]["parsed_arguments"] = None
            
            return final_tool_calls
        
        # Create test data
        streaming_chunks = [
            {"tool_calls": [{"index": 0, "id": "call_test", "function": {"arguments": "{\"key", "name": "test_tool"}, "type": "function"}]},
            {"tool_calls": [{"index": 0, "id": None, "function": {"arguments": "\":\"value\"}", "name": None}, "type": None}]},
            {"tool_calls": None}
        ]
        
        # Process the chunks
        result = await process_streaming_tool_calls(streaming_chunks)
        
        # Verify the result
        assert 0 in result
        assert result[0]["id"] == "call_test"
        assert result[0]["function"]["name"] == "test_tool"
        assert result[0]["function"]["arguments"] == "{\"key\":\"value\"}"
        assert result[0]["function"]["parsed_arguments"] == {"key": "value"}