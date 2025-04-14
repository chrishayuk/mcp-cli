import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, call, mock_open
from io import StringIO

# Import the module to test
from mcp_cli.commands import tools

@pytest.fixture
def mock_stream_manager():
    """Create a mock StreamManager with predefined test data."""
    mock_manager = MagicMock()
    
    # Set up tools
    mock_manager.get_all_tools.return_value = [
        {
            "name": "tool1", 
            "description": "Tool 1 description",
            "parameters": {
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"},
                    "param2": {"type": "integer", "description": "Second parameter"}
                },
                "required": ["param1"]
            }
        },
        {
            "name": "tool2", 
            "description": "Tool 2 description",
            "parameters": {
                "properties": {
                    "option": {"type": "boolean", "description": "Option parameter"}
                },
                "required": []
            }
        },
        {
            "name": "tool3", 
            "description": "This is a very long description that should be truncated in the table display because it exceeds the maximum length allowed",
            "inputSchema": {
                "properties": {
                    "data": {"type": "object", "description": "Data object"}
                },
                "required": ["data"]
            }
        }
    ]
    
    # Set up server info
    mock_manager.get_server_info.return_value = [
        {
            "id": 1, 
            "name": "TestServer1", 
            "tools": 2, 
            "status": "Connected"
        },
        {
            "id": 2, 
            "name": "TestServer2", 
            "tools": 1, 
            "status": "Connected"
        }
    ]
    
    # Set up server mapping for tools
    mock_manager.get_server_for_tool = MagicMock()
    mock_manager.get_server_for_tool.side_effect = lambda name: {
        "tool1": "TestServer1",
        "tool2": "TestServer1",
        "tool3": "TestServer2"
    }.get(name, "Unknown")
    
    # Set up call_tool method
    mock_manager.call_tool = AsyncMock()
    
    return mock_manager

@pytest.mark.asyncio
async def test_tools_list(mock_stream_manager, capsys):
    """Test listing tools from all servers."""
    # Run the function
    await tools.tools_list(mock_stream_manager)
    
    # Capture the output
    captured = capsys.readouterr()
    
    # Print the raw output for debugging
    print(f"Raw output: {repr(captured.out)}")
    
    # Check the most essential elements that should be in the output
    # Verify server names appear
    assert "TestServer1" in captured.out
    assert "TestServer2" in captured.out
    
    # Check that tool names appear
    assert "tool1" in captured.out
    assert "tool2" in captured.out
    assert "tool3" in captured.out
    
    # Check for description parts
    assert "Tool 1" in captured.out
    assert "Tool 2" in captured.out
    
    # Check the summary (which shouldn't have any complex formatting)
    assert "Total tools available: 3" in captured.out

@pytest.mark.asyncio
async def test_tools_list_no_tools(mock_stream_manager, capsys):
    """Test listing tools when none are available."""
    # Set up the mock to return no tools
    mock_stream_manager.get_all_tools.return_value = []
    
    # Run the function
    await tools.tools_list(mock_stream_manager)
    
    # Capture the output
    captured = capsys.readouterr()
    
    # Check the message
    assert "No tools available from any server" in captured.out

@pytest.mark.asyncio
async def test_tools_call_select_tool(mock_stream_manager, monkeypatch, capsys):
    """Test selecting and calling a tool."""
    # Set up the mock to return a successful result
    mock_stream_manager.call_tool.return_value = {
        "isError": False,
        "content": {"result": "success", "data": "test data"}
    }
    
    # Mock the input function to select tool 1 and provide arguments
    inputs = iter(["1", '{"param1": "value1", "param2": 42}'])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    
    # Run the function
    await tools.tools_call(mock_stream_manager)
    
    # Capture the output
    captured = capsys.readouterr()
    
    # Check that the tool selection appears
    assert "Available tools:" in captured.out
    assert "1. tool1 (from TestServer1)" in captured.out
    assert "2. tool2 (from TestServer1)" in captured.out
    assert "3. tool3 (from TestServer2)" in captured.out
    
    # Check that the tool details appear
    assert "Selected: tool1 from TestServer1" in captured.out
    assert "Description: Tool 1 description" in captured.out
    
    # Check that the parameters appear
    assert "Parameters:" in captured.out
    assert "param1 (string) [Required]" in captured.out
    assert "param2 (integer) [Optional]" in captured.out
    
    # Check that the call was made with the right arguments
    mock_stream_manager.call_tool.assert_called_once_with(
        tool_name="tool1",
        arguments={"param1": "value1", "param2": 42},
        server_name="TestServer1"
    )
    
    # Check that the response appears
    assert "Tool response:" in captured.out
    assert "success" in captured.out
    assert "test data" in captured.out

@pytest.mark.asyncio
async def test_tools_call_no_tools(mock_stream_manager, capsys):
    """Test calling tools when none are available."""
    # Set up the mock to return no tools
    mock_stream_manager.get_all_tools.return_value = []
    
    # Run the function
    await tools.tools_call(mock_stream_manager)
    
    # Capture the output
    captured = capsys.readouterr()
    
    # Check the message
    assert "No tools available from any server" in captured.out

@pytest.mark.asyncio
async def test_tools_call_invalid_selection(mock_stream_manager, monkeypatch, capsys):
    """Test selecting an invalid tool index."""
    # Mock the input function to select an invalid index
    monkeypatch.setattr("builtins.input", lambda _: "10")
    
    # Run the function
    await tools.tools_call(mock_stream_manager)
    
    # Capture the output
    captured = capsys.readouterr()
    
    # Check the error message
    assert "Invalid selection" in captured.out
    
    # Check that call_tool was not called
    mock_stream_manager.call_tool.assert_not_called()

@pytest.mark.asyncio
async def test_tools_call_non_numeric_selection(mock_stream_manager, monkeypatch, capsys):
    """Test entering a non-numeric tool selection."""
    # Mock the input function to enter a non-numeric value
    monkeypatch.setattr("builtins.input", lambda _: "abc")
    
    # Run the function
    await tools.tools_call(mock_stream_manager)
    
    # Capture the output
    captured = capsys.readouterr()
    
    # Check the error message
    assert "Please enter a number" in captured.out
    
    # Check that call_tool was not called
    mock_stream_manager.call_tool.assert_not_called()

@pytest.mark.asyncio
async def test_tools_call_invalid_json(mock_stream_manager, monkeypatch, capsys):
    """Test entering invalid JSON for tool arguments."""
    # Mock the input function to select tool 1 and provide invalid JSON
    inputs = iter(["1", "invalid json"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    
    # Run the function
    await tools.tools_call(mock_stream_manager)
    
    # Capture the output
    captured = capsys.readouterr()
    
    # Check the error message
    assert "Invalid JSON" in captured.out
    
    # Check that call_tool was not called
    mock_stream_manager.call_tool.assert_not_called()

@pytest.mark.asyncio
async def test_tools_call_empty_args(mock_stream_manager, monkeypatch, capsys):
    """Test calling a tool with empty arguments."""
    # Set up the mock to return a successful result
    mock_stream_manager.call_tool.return_value = {
        "isError": False,
        "content": "success"
    }
    
    # Mock the input function to select tool 1 and provide empty arguments
    inputs = iter(["1", ""])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    
    # Run the function
    await tools.tools_call(mock_stream_manager)
    
    # Check that call_tool was called with empty arguments
    mock_stream_manager.call_tool.assert_called_once_with(
        tool_name="tool1",
        arguments={},
        server_name="TestServer1"
    )

@pytest.mark.asyncio
async def test_tools_call_error_response(mock_stream_manager, monkeypatch, capsys):
    """Test calling a tool that returns an error."""
    # Set up the mock to return an error
    mock_stream_manager.call_tool.return_value = {
        "isError": True,
        "error": "Test error message"
    }
    
    # Mock the input function to select tool 1 and provide arguments
    inputs = iter(["1", '{"param1": "value1"}'])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    
    # Run the function
    await tools.tools_call(mock_stream_manager)
    
    # Capture the output
    captured = capsys.readouterr()
    
    # Check the error message
    assert "Error calling tool: Test error message" in captured.out

@pytest.mark.asyncio
async def test_tools_call_exception(mock_stream_manager, monkeypatch, capsys):
    """Test handling of exceptions when calling a tool."""
    # Set up the mock to raise an exception
    mock_stream_manager.call_tool.side_effect = Exception("Test exception")
    
    # Mock the input function to select tool 1 and provide arguments
    inputs = iter(["1", '{"param1": "value1"}'])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    
    # Run the function
    await tools.tools_call(mock_stream_manager)
    
    # Capture the output
    captured = capsys.readouterr()
    
    # Check the error message
    assert "Error: Test exception" in captured.out

@pytest.mark.asyncio
async def test_tools_call_list_response(mock_stream_manager, monkeypatch, capsys):
    """Test calling a tool that returns a list of objects."""
    # Set up the mock to return a list of objects
    mock_stream_manager.call_tool.return_value = {
        "isError": False,
        "content": [
            {"name": "item1", "value": 1},
            {"name": "item2", "value": 2}
        ]
    }
    
    # Mock the input function to select tool 1 and provide arguments
    inputs = iter(["1", '{"param1": "value1"}'])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    
    # Run the function
    await tools.tools_call(mock_stream_manager)
    
    # Capture the output
    captured = capsys.readouterr()
    
    # Check that the response appears as JSON
    assert "Tool response:" in captured.out
    assert "item1" in captured.out
    assert "item2" in captured.out

@pytest.mark.asyncio
async def test_tools_call_scalar_response(mock_stream_manager, monkeypatch, capsys):
    """Test calling a tool that returns a scalar value."""
    # Set up the mock to return a scalar value
    mock_stream_manager.call_tool.return_value = {
        "isError": False,
        "content": "Simple text response"
    }
    
    # Mock the input function to select tool 1 and provide arguments
    inputs = iter(["1", '{"param1": "value1"}'])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    
    # Run the function
    await tools.tools_call(mock_stream_manager)
    
    # Capture the output
    captured = capsys.readouterr()
    
    # Check that the response appears as text
    assert "Tool response: Simple text response" in captured.out

@pytest.mark.asyncio
async def test_tools_call_no_content(mock_stream_manager, monkeypatch, capsys):
    """Test calling a tool that returns no content."""
    # Set up the mock to return an empty response
    mock_stream_manager.call_tool.return_value = {
        "isError": False
    }
    
    # Mock the input function to select tool 1 and provide arguments
    inputs = iter(["1", '{"param1": "value1"}'])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    
    # Run the function
    await tools.tools_call(mock_stream_manager)
    
    # Capture the output
    captured = capsys.readouterr()
    
    # Check that the default message appears
    assert "Tool response: No content returned" in captured.out

@pytest.mark.asyncio
async def test_tools_call_inputSchema_format(mock_stream_manager, monkeypatch, capsys):
    """Test calling a tool that uses inputSchema instead of parameters."""
    # Mock the input function to select tool 3 (which has inputSchema) and provide arguments
    inputs = iter(["3", '{"data": {"field": "value"}}'])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    
    # Set up the mock to return a successful result
    mock_stream_manager.call_tool.return_value = {
        "isError": False,
        "content": "Success"
    }
    
    # Run the function
    await tools.tools_call(mock_stream_manager)
    
    # Capture the output
    captured = capsys.readouterr()
    
    # Check that the parameters appear correctly
    assert "Parameters:" in captured.out
    assert "data (object) [Required]" in captured.out
    
    # Check that call_tool was called with the right arguments
    mock_stream_manager.call_tool.assert_called_once_with(
        tool_name="tool3",
        arguments={"data": {"field": "value"}},
        server_name="TestServer2"
    )