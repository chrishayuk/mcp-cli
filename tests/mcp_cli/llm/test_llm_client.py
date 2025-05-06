# tests/mcp_cli/llm/test_llm_client.py
import pytest
from unittest.mock import patch, MagicMock

from mcp_cli.llm.llm_client import get_llm_client
from mcp_cli.llm.providers.openai_client import OpenAILLMClient


class TestGetLLMClient:
    """Test the get_llm_client factory function."""

    def test_get_ollama_client(self):
        """Test that an Ollama client is returned for 'ollama' provider."""
        # The import happens inside the function, so we need to patch the import path
        with patch("mcp_cli.llm.providers.ollama_client.OllamaLLMClient") as mock_ollama:
            # Configure the mock to return a fake client
            mock_instance = MagicMock()
            mock_ollama.return_value = mock_instance
            
            # Call the function that should create an Ollama client
            client = get_llm_client(provider="ollama")
            
            # Verify OllamaLLMClient was called
            mock_ollama.assert_called_once()
            
            # Verify we got the mock client back
            assert client == mock_instance


@pytest.mark.asyncio
class TestOpenAIClient:
    """Test the OpenAI client implementation."""

    @patch("openai.OpenAI")
    async def test_create_completion(self, mock_openai):
        """Test that create_completion calls the OpenAI API correctly."""
        # Set up mock client and response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_message = MagicMock()
        mock_message.content = "Test response"
        mock_message.tool_calls = None
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]
        
        # Mock run_in_executor to avoid real API calls
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            
            # Configure run_in_executor to return our mock response
            async def mock_run_in_executor(*args, **kwargs):
                return mock_response
                
            mock_loop.run_in_executor = mock_run_in_executor
            
            # Create client and call method
            client = OpenAILLMClient(model="test-model", api_key="test-key")
            result = await client.create_completion(
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            # Verify the result
            assert isinstance(result, dict)
            assert "response" in result
            assert result["response"] == "Test response"
            assert "tool_calls" in result
            assert isinstance(result["tool_calls"], list)
            assert len(result["tool_calls"]) == 0

    @patch("openai.OpenAI")
    async def test_create_completion_with_tools(self, mock_openai):
        """Test create_completion with tool calls."""
        # Set up mock client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        # Set up the mock response with tool calls
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function = MagicMock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"test": "value"}'
        
        mock_message = MagicMock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]
        
        # Mock run_in_executor to avoid real API calls
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            
            # Configure run_in_executor to return our mock response
            async def mock_run_in_executor(*args, **kwargs):
                return mock_response
                
            mock_loop.run_in_executor = mock_run_in_executor
            
            # Create client and call method
            client = OpenAILLMClient(model="test-model", api_key="test-key")
            result = await client.create_completion(
                messages=[{"role": "user", "content": "Use tool"}],
                tools=[{"type": "function", "function": {"name": "test_tool"}}]
            )
            
            # Verify the result structure for tool calls
            assert isinstance(result, dict)
            assert "response" in result
            assert result["response"] is None  # None when tools are used
            assert "tool_calls" in result
            assert len(result["tool_calls"]) == 1
            assert result["tool_calls"][0]["function"]["name"] == "test_tool"
            assert result["tool_calls"][0]["id"] == "call_1"