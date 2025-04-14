import pytest
import asyncio
import sys
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch, mock_open, call
from io import StringIO

# Import the module to test
from mcp_cli.commands import cmd

@pytest.fixture
def mock_stream_manager():
    """Create a mock StreamManager with predefined test data."""
    mock_manager = MagicMock()
    
    # Set up tools
    mock_manager.get_internal_tools.return_value = [
        {"name": "TestServer1_tool1", "description": "Test tool 1"},
        {"name": "TestServer1_tool2", "description": "Test tool 2"},
        {"name": "TestServer2_tool3", "description": "Test tool 3"}
    ]
    
    # Set up call_tool method
    mock_manager.call_tool = AsyncMock()
    
    return mock_manager

@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    mock_client = MagicMock()
    mock_client.create_completion = MagicMock()
    return mock_client

@pytest.mark.asyncio
async def test_run_single_tool(mock_stream_manager):
    """Test running a single tool with arguments."""
    # Setup the mock to return a successful result
    mock_stream_manager.call_tool.return_value = {
        "isError": False,
        "content": {"result": "success", "data": "test data"}
    }
    
    # Run the function
    result = await cmd.run_single_tool("test_tool", '{"param": "value"}', mock_stream_manager)
    
    # Check that call_tool was called with the right arguments
    mock_stream_manager.call_tool.assert_called_once_with(
        tool_name="test_tool",
        arguments={"param": "value"}
    )
    
    # Check the result
    assert "success" in result
    assert "test data" in result

@pytest.mark.asyncio
async def test_run_single_tool_with_error(mock_stream_manager):
    """Test running a single tool that returns an error."""
    # Setup the mock to return an error
    mock_stream_manager.call_tool.return_value = {
        "isError": True,
        "error": "Test error message",
        "content": "Error details"
    }
    
    # When the tool returns an error, it should exit the program
    with pytest.raises(SystemExit) as excinfo:
        await cmd.run_single_tool("error_tool", None, mock_stream_manager)
    
    # Check that sys.exit was called with error code 1
    assert excinfo.value.code == 1
    
    # Check that call_tool was called with the right arguments
    mock_stream_manager.call_tool.assert_called_once_with(
        tool_name="error_tool",
        arguments={}
    )

@pytest.mark.asyncio
async def test_run_single_tool_with_invalid_json(mock_stream_manager):
    """Test running a single tool with invalid JSON arguments."""
    # When invalid JSON is provided, it should exit the program
    with pytest.raises(SystemExit) as excinfo:
        await cmd.run_single_tool("test_tool", "invalid json", mock_stream_manager)
    
    # Check that sys.exit was called with error code 1
    assert excinfo.value.code == 1
    
    # The call_tool method should not have been called
    mock_stream_manager.call_tool.assert_not_called()

@pytest.mark.asyncio
async def test_process_tool_calls(mock_stream_manager):
    """Test processing tool calls."""
    # Create a mock conversation
    conversation = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User message"}
    ]
    
    # Create mock tool calls
    tool_calls = [
        {
            "function": {
                "name": "test_tool1",
                "arguments": '{"param": "value1"}'
            }
        },
        {
            "function": {
                "name": "test_tool2",
                "arguments": '{"param": "value2"}'
            }
        }
    ]
    
    # Mock the handle_tool_call function
    with patch("mcp_cli.commands.cmd.handle_tool_call", new=AsyncMock()) as mock_handle_tool_call:
        # Process the tool calls
        await cmd.process_tool_calls(tool_calls, conversation, mock_stream_manager)
        
        # Check that handle_tool_call was called twice with the right arguments
        assert mock_handle_tool_call.call_count == 2
        
        # Check that handle_tool_call was called with the correct arguments
        # The actual function call is: await handle_tool_call(tool_call, conversation, [], stream_manager=stream_manager)
        mock_handle_tool_call.assert_any_call(
            tool_calls[0], 
            conversation, 
            [], 
            stream_manager=mock_stream_manager
        )
        
        mock_handle_tool_call.assert_any_call(
            tool_calls[1], 
            conversation, 
            [], 
            stream_manager=mock_stream_manager
        )

@pytest.mark.asyncio
async def test_run_llm_with_tools_success(mock_stream_manager, mock_llm_client):
    """Test running LLM with tools - successful case."""
    # Mock the LLM client to return a simple response
    mock_llm_client.create_completion.return_value = {
        "response": "This is a test response"
    }
    
    # Mock the get_llm_client function to return our mock client
    with patch("mcp_cli.commands.cmd.get_llm_client", return_value=mock_llm_client):
        # Mock the convert_to_openai_tools function
        with patch("mcp_cli.commands.cmd.convert_to_openai_tools", return_value=[{"name": "mock_tool"}]):
            # Mock the generate_system_prompt function
            with patch("mcp_cli.commands.cmd.generate_system_prompt", return_value="System prompt"):
                # Run the function
                result = await cmd.run_llm_with_tools(
                    "test-provider",
                    "test-model",
                    "Test input",
                    None,
                    None,
                    mock_stream_manager
                )
                
                # Check the result
                assert result == "This is a test response"
                
                # Check that create_completion was called with the right arguments
                mock_llm_client.create_completion.assert_called_once()
                call_args = mock_llm_client.create_completion.call_args[1]
                assert len(call_args["messages"]) == 2
                assert call_args["messages"][0]["role"] == "system"
                assert call_args["messages"][1]["role"] == "user"
                assert call_args["messages"][1]["content"] == "Test input"
                assert call_args["tools"] == [{"name": "mock_tool"}]

@pytest.mark.asyncio
async def test_run_llm_with_tools_with_prompt_template(mock_stream_manager, mock_llm_client):
    """Test running LLM with a prompt template."""
    # Mock the LLM client to return a simple response
    mock_llm_client.create_completion.return_value = {
        "response": "This is a test response"
    }
    
    # Mock the required functions
    with patch("mcp_cli.commands.cmd.get_llm_client", return_value=mock_llm_client):
        with patch("mcp_cli.commands.cmd.convert_to_openai_tools", return_value=[]):
            with patch("mcp_cli.commands.cmd.generate_system_prompt", return_value="System prompt"):
                # Run the function with a prompt template
                result = await cmd.run_llm_with_tools(
                    "test-provider",
                    "test-model",
                    "Test input",
                    "Template with {{input}}",
                    None,
                    mock_stream_manager
                )
                
                # Check that create_completion was called with the right arguments
                mock_llm_client.create_completion.assert_called_once()
                call_args = mock_llm_client.create_completion.call_args[1]
                assert call_args["messages"][1]["content"] == "Template with Test input"

@pytest.mark.asyncio
async def test_run_llm_with_tools_with_custom_system_prompt(mock_stream_manager, mock_llm_client):
    """Test running LLM with a custom system prompt."""
    # Mock the LLM client to return a simple response
    mock_llm_client.create_completion.return_value = {
        "response": "This is a test response"
    }
    
    # Mock the required functions
    with patch("mcp_cli.commands.cmd.get_llm_client", return_value=mock_llm_client):
        with patch("mcp_cli.commands.cmd.convert_to_openai_tools", return_value=[]):
            # Don't need to mock generate_system_prompt as it shouldn't be called
            
            # Run the function with a custom system prompt
            result = await cmd.run_llm_with_tools(
                "test-provider",
                "test-model",
                "Test input",
                None,
                "Custom system prompt",
                mock_stream_manager
            )
            
            # Check that create_completion was called with the right system prompt
            mock_llm_client.create_completion.assert_called_once()
            call_args = mock_llm_client.create_completion.call_args[1]
            assert call_args["messages"][0]["content"] == "Custom system prompt"

@pytest.mark.asyncio
async def test_run_llm_with_tools_client_error(mock_stream_manager):
    """Test handling of errors when creating the LLM client."""
    # Mock get_llm_client to raise an exception
    with patch("mcp_cli.commands.cmd.get_llm_client", side_effect=Exception("Test client error")):
        # Run the function
        result = await cmd.run_llm_with_tools(
            "test-provider",
            "test-model",
            "Test input",
            None,
            None,
            mock_stream_manager
        )
        
        # Check that the result includes the error message
        assert "Error" in result
        assert "Test client error" in result

@pytest.mark.asyncio
async def test_run_llm_with_tools_completion_error(mock_stream_manager, mock_llm_client):
    """Test handling of errors during LLM completion."""
    # Mock the LLM client to raise an exception during create_completion
    mock_llm_client.create_completion.side_effect = Exception("Test completion error")
    
    # Mock the required functions
    with patch("mcp_cli.commands.cmd.get_llm_client", return_value=mock_llm_client):
        with patch("mcp_cli.commands.cmd.convert_to_openai_tools", return_value=[]):
            with patch("mcp_cli.commands.cmd.generate_system_prompt", return_value="System prompt"):
                # Run the function
                result = await cmd.run_llm_with_tools(
                    "test-provider",
                    "test-model",
                    "Test input",
                    None,
                    None,
                    mock_stream_manager
                )
                
                # Check that the result includes the error message
                assert "Error" in result
                assert "Test completion error" in result

@pytest.mark.asyncio
async def test_run_llm_with_tools_none_completion(mock_stream_manager, mock_llm_client):
    """Test handling of None response from LLM."""
    # Mock the LLM client to return None
    mock_llm_client.create_completion.return_value = None
    
    # Mock the required functions
    with patch("mcp_cli.commands.cmd.get_llm_client", return_value=mock_llm_client):
        with patch("mcp_cli.commands.cmd.convert_to_openai_tools", return_value=[]):
            with patch("mcp_cli.commands.cmd.generate_system_prompt", return_value="System prompt"):
                # Run the function
                result = await cmd.run_llm_with_tools(
                    "test-provider",
                    "test-model",
                    "Test input",
                    None,
                    None,
                    mock_stream_manager
                )
                
                # Check that the result includes an error message
                assert "Error" in result
                assert "no response" in result

@pytest.mark.asyncio
async def test_run_llm_with_tools_tool_calls(mock_stream_manager, mock_llm_client):
    """Test running LLM that returns tool calls."""
    # Set up the mock client to return tool calls and then a response
    first_completion = {
        "tool_calls": [
            {
                "function": {
                    "name": "test_tool",
                    "arguments": '{"param": "value"}'
                }
            }
        ]
    }
    second_completion = {
        "response": "This is the final response"
    }
    
    mock_llm_client.create_completion.side_effect = [first_completion, second_completion]
    
    # Mock the required functions
    with patch("mcp_cli.commands.cmd.get_llm_client", return_value=mock_llm_client):
        with patch("mcp_cli.commands.cmd.convert_to_openai_tools", return_value=[]):
            with patch("mcp_cli.commands.cmd.generate_system_prompt", return_value="System prompt"):
                # Mock process_tool_calls
                with patch("mcp_cli.commands.cmd.process_tool_calls", new=AsyncMock()) as mock_process_tool_calls:
                    # Run the function
                    result = await cmd.run_llm_with_tools(
                        "test-provider",
                        "test-model",
                        "Test input",
                        None,
                        None,
                        mock_stream_manager
                    )
                    
                    # Check that process_tool_calls was called
                    mock_process_tool_calls.assert_called_once()
                    
                    # Check that create_completion was called twice
                    assert mock_llm_client.create_completion.call_count == 2
                    
                    # Check the result
                    assert result == "This is the final response"

@pytest.mark.asyncio
async def test_run_llm_with_tools_multiple_tool_calls(mock_stream_manager, mock_llm_client):
    """Test handling of multiple rounds of tool calls."""
    # Set up the mock client to return multiple rounds of tool calls and then a response
    first_completion = {
        "tool_calls": [{"function": {"name": "tool1", "arguments": '{}'}}]
    }
    second_completion = {
        "tool_calls": [{"function": {"name": "tool2", "arguments": '{}'}}]
    }
    third_completion = {
        "tool_calls": [{"function": {"name": "tool3", "arguments": '{}'}}]
    }
    final_completion = {
        "response": "Final response after tools"
    }
    
    mock_llm_client.create_completion.side_effect = [
        first_completion, second_completion, third_completion, final_completion
    ]
    
    # Mock the required functions
    with patch("mcp_cli.commands.cmd.get_llm_client", return_value=mock_llm_client):
        with patch("mcp_cli.commands.cmd.convert_to_openai_tools", return_value=[]):
            with patch("mcp_cli.commands.cmd.generate_system_prompt", return_value="System prompt"):
                # Mock process_tool_calls
                with patch("mcp_cli.commands.cmd.process_tool_calls", new=AsyncMock()) as mock_process_tool_calls:
                    # Run the function
                    result = await cmd.run_llm_with_tools(
                        "test-provider",
                        "test-model",
                        "Test input",
                        None,
                        None,
                        mock_stream_manager
                    )
                    
                    # Check that process_tool_calls was called three times
                    assert mock_process_tool_calls.call_count == 3
                    
                    # Check that create_completion was called four times
                    assert mock_llm_client.create_completion.call_count == 4
                    
                    # Check the result
                    assert result == "Final response after tools"

@pytest.mark.asyncio
async def test_run_llm_with_tools_max_iterations(mock_stream_manager, mock_llm_client):
    """Test that tool calls stop after max iterations."""
    # Set up the mock client to return tool calls indefinitely
    tool_call_completion = {
        "tool_calls": [{"function": {"name": "test_tool", "arguments": '{}'}}]
    }
    
    # Always return tool calls (would be infinite without the max iterations limit)
    mock_llm_client.create_completion.return_value = tool_call_completion
    
    # Mock the required functions
    with patch("mcp_cli.commands.cmd.get_llm_client", return_value=mock_llm_client):
        with patch("mcp_cli.commands.cmd.convert_to_openai_tools", return_value=[]):
            with patch("mcp_cli.commands.cmd.generate_system_prompt", return_value="System prompt"):
                # Instead of mocking process_tool_calls completely, we'll implement a version
                # that adds tool messages to the conversation, similar to what the real function does
                async def mock_process_tool_calls_with_messages(tool_calls, conversation, stream_manager):
                    # Add a tool message to the conversation for each tool call
                    for i, tool_call in enumerate(tool_calls):
                        tool_name = tool_call.get("function", {}).get("name", f"tool{i}")
                        conversation.append({
                            "role": "tool",
                            "name": tool_name,
                            "content": f"Tool {tool_name} result content"
                        })
                
                with patch("mcp_cli.commands.cmd.process_tool_calls", 
                          new=AsyncMock(side_effect=mock_process_tool_calls_with_messages)) as mock_process_tool_calls:
                    # Run the function
                    result = await cmd.run_llm_with_tools(
                        "test-provider",
                        "test-model",
                        "Test input",
                        None,
                        None,
                        mock_stream_manager
                    )
                    
                    # Verify process_tool_calls was called the expected number of times (4)
                    assert mock_process_tool_calls.call_count == 4
                    
                    # The number of create_completion calls might be 5 because:
                    # 1. Initial call
                    # 2-4. After each tool call (3 iterations)
                    # 5. Final attempt to get a completion without tool calls
                    assert mock_llm_client.create_completion.call_count == 5
                    
                    # Now that we've properly mocked the tool messages, check for the summary
                    assert "Based on the tools executed" in result
                    assert "Tool test_tool result content" in result
                    
def test_write_output_to_stdout(capsys):
    """Test writing output to stdout."""
    # Call the function with no output path (defaults to stdout)
    cmd.write_output("Test content", None)
    
    # Check that the content was printed to stdout
    captured = capsys.readouterr()
    assert captured.out == "Test content\n"

def test_write_output_to_file():
    """Test writing output to a file."""
    # Mock the open function
    mock_file = mock_open()
    
    # Call the function with an output path
    with patch("builtins.open", mock_file):
        cmd.write_output("Test content", "test_output.txt")
    
    # Check that the file was opened and written to
    mock_file.assert_called_once_with("test_output.txt", "w")
    mock_file().write.assert_called_once_with("Test content")

def test_write_output_to_stdout_with_dash(capsys):
    """Test writing output to stdout with '-' as output path."""
    # Call the function with '-' as output path (means stdout)
    cmd.write_output("Test content", "-")
    
    # Check that the content was printed to stdout
    captured = capsys.readouterr()
    assert captured.out == "Test content\n"

def test_write_output_with_file_error():
    """Test handling of errors when writing to a file."""
    # Mock the open function to raise an exception
    with patch("builtins.open", side_effect=Exception("Test file error")):
        # When there's an error writing to the file, it should exit
        with pytest.raises(SystemExit) as excinfo:
            cmd.write_output("Test content", "test_output.txt")
        
        # Check that sys.exit was called with error code 1
        assert excinfo.value.code == 1

def test_write_output_with_none_content(capsys):
    """Test writing None content."""
    # Call the function with None content
    cmd.write_output(None, None)
    
    # Check that a message was printed
    captured = capsys.readouterr()
    assert "No content returned from command" in captured.out

def test_write_output_with_raw_flag(capsys):
    """Test writing output with the raw flag."""
    # Call the function with the raw flag
    cmd.write_output({"key": "value"}, None, raw=True)
    
    # Check that the content was printed as a string representation
    captured = capsys.readouterr()
    assert "{'key': 'value'}" in captured.out

@pytest.mark.asyncio
async def test_cmd_run_with_tool(mock_stream_manager):
    """Test the cmd_run function with a tool."""
    # Mock run_single_tool
    with patch("mcp_cli.commands.cmd.run_single_tool", new=AsyncMock(return_value="Tool result")) as mock_run_tool:
        # Mock write_output
        with patch("mcp_cli.commands.cmd.write_output") as mock_write_output:
            # Call the function
            await cmd.cmd_run(
                tool="test_tool",
                tool_args='{"param": "value"}',
                stream_manager=mock_stream_manager
            )
            
            # Check that run_single_tool was called with the right arguments
            mock_run_tool.assert_called_once_with(
                "test_tool",
                '{"param": "value"}',
                mock_stream_manager
            )
            
            # Check that write_output was called with the tool result
            mock_write_output.assert_called_once_with("Tool result", None, False)

@pytest.mark.asyncio
async def test_cmd_run_with_llm(mock_stream_manager):
    """Test the cmd_run function with LLM."""
    # Mock run_llm_with_tools
    with patch("mcp_cli.commands.cmd.run_llm_with_tools", 
               new=AsyncMock(return_value="LLM result")) as mock_run_llm:
        # Mock write_output
        with patch("mcp_cli.commands.cmd.write_output") as mock_write_output:
            # Mock open to prevent file not found error
            with patch("builtins.open", mock_open(read_data="File content")):
                # Call the function
                await cmd.cmd_run(
                    input="test_input",
                    prompt="test_prompt",
                    stream_manager=mock_stream_manager
                )
                
                # Check that run_llm_with_tools was called
                mock_run_llm.assert_called_once()
                
                # Check that the input text was read from the file
                call_args = mock_run_llm.call_args[0]
                assert call_args[2] == "File content"
                
                # Check that write_output was called with the LLM result
                mock_write_output.assert_called_once_with("LLM result", None, False)

@pytest.mark.asyncio
async def test_cmd_run_with_file_input(mock_stream_manager):
    """Test the cmd_run function with file input."""
    # Mock open to return test content
    with patch("builtins.open", mock_open(read_data="File content")) as mock_file:
        # Mock run_llm_with_tools
        with patch("mcp_cli.commands.cmd.run_llm_with_tools", 
                  new=AsyncMock(return_value="LLM result")) as mock_run_llm:
            # Mock write_output
            with patch("mcp_cli.commands.cmd.write_output") as mock_write_output:
                # Call the function
                await cmd.cmd_run(
                    input="test_file.txt",
                    stream_manager=mock_stream_manager
                )
                
                # Check that the file was opened
                mock_file.assert_called_once_with("test_file.txt", "r")
                
                # Check that run_llm_with_tools was called with the file content
                mock_run_llm.assert_called_once()
                call_args = mock_run_llm.call_args[0]
                assert call_args[2] == "File content"
                
                # Check that write_output was called with the LLM result
                mock_write_output.assert_called_once_with("LLM result", None, False)

@pytest.mark.asyncio
async def test_cmd_run_with_stdin_input(mock_stream_manager):
    """Test the cmd_run function with stdin input."""
    # Mock sys.stdin to return test content
    stdin_content = "Stdin content"
    with patch("sys.stdin.read", return_value=stdin_content):
        # Mock run_llm_with_tools
        with patch("mcp_cli.commands.cmd.run_llm_with_tools", 
                  new=AsyncMock(return_value="LLM result")) as mock_run_llm:
            # Mock write_output
            with patch("mcp_cli.commands.cmd.write_output") as mock_write_output:
                # Call the function
                await cmd.cmd_run(
                    input="-",  # "-" means read from stdin
                    stream_manager=mock_stream_manager
                )
                
                # Check that run_llm_with_tools was called with the stdin content
                mock_run_llm.assert_called_once()
                call_args = mock_run_llm.call_args[0]
                assert call_args[2] == stdin_content
                
                # Check that write_output was called with the LLM result
                mock_write_output.assert_called_once_with("LLM result", None, False)

@pytest.mark.asyncio
async def test_cmd_run_with_file_input_error(mock_stream_manager):
    """Test the cmd_run function with file input error."""
    # Mock open to raise an exception
    with patch("builtins.open", side_effect=Exception("Test file error")):
        # When there's an error reading the input file, it should exit
        with pytest.raises(SystemExit) as excinfo:
            await cmd.cmd_run(
                input="nonexistent_file.txt",
                stream_manager=mock_stream_manager
            )
        
        # Check that sys.exit was called with error code 1
        assert excinfo.value.code == 1

@pytest.mark.asyncio
async def test_cmd_run_with_error(mock_stream_manager):
    """Test the cmd_run function with a general error."""
    # Mock run_llm_with_tools to raise an exception
    with patch("mcp_cli.commands.cmd.run_llm_with_tools", 
              side_effect=Exception("Test general error")):
        # When there's a general error, it should exit
        with pytest.raises(SystemExit) as excinfo:
            await cmd.cmd_run(
                stream_manager=mock_stream_manager
            )
        
        # Check that sys.exit was called with error code 1
        assert excinfo.value.code == 1

@pytest.mark.asyncio
async def test_cmd_run_with_output_file(mock_stream_manager):
    """Test the cmd_run function with output to a file."""
    # Mock run_llm_with_tools
    with patch("mcp_cli.commands.cmd.run_llm_with_tools", 
              new=AsyncMock(return_value="LLM result")) as mock_run_llm:
        # Mock write_output
        with patch("mcp_cli.commands.cmd.write_output") as mock_write_output:
            # Call the function
            await cmd.cmd_run(
                output="output_file.txt",
                stream_manager=mock_stream_manager
            )
            
            # Check that write_output was called with the output file
            mock_write_output.assert_called_once_with("LLM result", "output_file.txt", False)

@pytest.mark.asyncio
async def test_cmd_run_with_raw_flag(mock_stream_manager):
    """Test the cmd_run function with the raw flag."""
    # Mock run_llm_with_tools
    with patch("mcp_cli.commands.cmd.run_llm_with_tools", 
              new=AsyncMock(return_value="LLM result")) as mock_run_llm:
        # Mock write_output
        with patch("mcp_cli.commands.cmd.write_output") as mock_write_output:
            # Call the function
            await cmd.cmd_run(
                raw=True,
                stream_manager=mock_stream_manager
            )
            
            # Check that write_output was called with raw=True
            mock_write_output.assert_called_once_with("LLM result", None, True)