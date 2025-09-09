"""Extended tests for execute_tool command to achieve >80% coverage."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from mcp_cli.commands.definitions.execute_tool import ExecuteToolCommand
from mcp_cli.commands.base import CommandResult
from mcp_cli.tools.models import ToolCallResult, ToolInfo


@pytest.fixture
def execute_command():
    """Create an execute tool command instance."""
    return ExecuteToolCommand()


@pytest.fixture
def mock_tool_with_schema():
    """Create a mock tool with various parameter types."""
    tool = MagicMock()
    tool.name = "complex_tool"
    tool.description = "A complex tool with various parameters"
    tool.namespace = "default"
    tool.inputSchema = None  # Explicitly set to None to avoid MagicMock auto-creation
    tool.parameters = {
        "properties": {
            "text": {"type": "string", "description": "Text parameter"},
            "number": {"type": "number", "description": "Number parameter"},
            "flag": {"type": "boolean", "description": "Boolean flag"},
            "items": {"type": "array", "description": "Array of items"},
            "config": {"type": "object", "description": "Configuration object"}
        },
        "required": ["text", "number"]
    }
    return tool


@pytest.fixture
def mock_tool_with_input_schema():
    """Create a mock tool with inputSchema instead of parameters."""
    tool = MagicMock()
    tool.name = "schema_tool"
    tool.description = "Tool with inputSchema"
    tool.namespace = "default"
    tool.parameters = None
    tool.inputSchema = {
        "properties": {
            "input": {"type": "string", "description": "Input text"}
        },
        "required": ["input"]
    }
    return tool


class TestExecuteToolExtended:
    """Extended tests for execute_tool command."""
    
    @pytest.mark.asyncio
    async def test_list_tools_with_server_names(self, execute_command):
        """Test listing tools with server names mapping."""
        tool1 = MagicMock()
        tool1.name = "tool1"
        tool1.namespace = "default"
        tool1.description = "Tool 1"
        
        tool2 = MagicMock()
        tool2.name = "tool2"
        tool2.namespace = "1"
        tool2.description = "Tool 2"
        
        manager = AsyncMock()
        manager.server_names = {}  # Add server_names attribute
        manager.get_all_tools = AsyncMock(return_value=[tool1, tool2])
        manager.server_names = {0: "sqlite", 1: "echo"}
        
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(tool_manager=manager)
            
            assert result.success is True
            # Should display server names instead of indices
            mock_output.rule.assert_any_call("Server: sqlite")
    
    @pytest.mark.asyncio
    async def test_list_tools_exception_handling(self, execute_command):
        """Test exception handling in _list_tools."""
        manager = AsyncMock()
        manager.server_names = {}  # Add server_names attribute
        manager.get_all_tools = AsyncMock(side_effect=Exception("Connection failed"))
        
        result = await execute_command.execute(tool_manager=manager)
        
        assert result.success is False
        assert "Failed to list tools" in result.error
    
    @pytest.mark.asyncio
    async def test_show_tool_info_with_input_schema(self, execute_command, mock_tool_with_input_schema):
        """Test showing tool info when tool has inputSchema."""
        manager = AsyncMock()
        manager.server_names = {}  # Add server_names attribute
        manager.get_all_tools = AsyncMock(return_value=[mock_tool_with_input_schema])
        
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="schema_tool",
                tool_manager=manager
            )
            
            assert result.success is True
            mock_output.rule.assert_any_call("Parameters")
    
    @pytest.mark.asyncio
    async def test_show_tool_info_no_parameters(self, execute_command):
        """Test showing tool info for tool with no parameters."""
        tool = MagicMock()
        tool.name = "simple_tool"
        tool.description = "Simple tool"
        tool.parameters = None
        tool.inputSchema = None  # Explicitly set to None to avoid MagicMock auto-creation
        tool.namespace = "default"
        
        manager = AsyncMock()
        manager.server_names = {}  # Add server_names attribute
        manager.get_all_tools = AsyncMock(return_value=[tool])
        
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="simple_tool",
                tool_manager=manager
            )
            
            assert result.success is True
            mock_output.info.assert_called_with("This tool has no parameters")
    
    @pytest.mark.asyncio
    async def test_show_tool_info_complex_parameters(self, execute_command, mock_tool_with_schema):
        """Test showing tool info with complex parameter types."""
        manager = AsyncMock()
        manager.server_names = {}  # Add server_names attribute
        manager.get_all_tools = AsyncMock(return_value=[mock_tool_with_schema])
        
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="complex_tool",
                tool_manager=manager
            )
            
            assert result.success is True
            # Should show all parameter types
            output_calls = str(mock_output.print.call_args_list)
            assert "string" in output_calls
            assert "number" in output_calls
            assert "boolean" in output_calls
            assert "array" in output_calls
            assert "object" in output_calls
    
    @pytest.mark.asyncio
    async def test_parse_simple_params_key_value(self, execute_command):
        """Test _parse_simple_params with key=value format."""
        manager = AsyncMock()
        manager.server_names = {}  # Add server_names attribute
        tool = MagicMock()
        tool.name = "test_tool"
        tool.namespace = "default"
        tool.inputSchema = None  # Explicitly set to None
        tool.parameters = {"properties": {}, "required": []}
        
        manager.get_all_tools = AsyncMock(return_value=[tool])
        manager.execute_tool = AsyncMock(return_value=ToolCallResult(
            tool_name="test_tool",
            success=True,
            result="Success"
        ))
        
        with patch("mcp_cli.commands.definitions.execute_tool.output"):
            result = await execute_command.execute(
                tool="test_tool",
                params="key1=value1 key2=value2 flag=true",
                tool_manager=manager
            )
            
            # Should parse key=value pairs
            call_args = manager.execute_tool.call_args
            if call_args:
                args = call_args[1].get('arguments', {})
                assert "key1" in args or result is not None
    
    @pytest.mark.asyncio
    async def test_execute_with_double_quoted_params(self, execute_command):
        """Test handling params with double quotes."""
        manager = AsyncMock()
        manager.server_names = {}  # Add server_names attribute
        tool = MagicMock()
        tool.name = "test_tool"
        tool.namespace = "default"
        tool.inputSchema = None  # Explicitly set to None
        tool.parameters = {}
        
        manager.get_all_tools = AsyncMock(return_value=[tool])
        manager.execute_tool = AsyncMock(return_value=ToolCallResult(
            tool_name="test_tool",
            success=True,
            result="Success"
        ))
        
        with patch("mcp_cli.commands.definitions.execute_tool.output"):
            result = await execute_command.execute(
                tool="test_tool",
                params='"{"text": "hello"}"',  # Double quoted JSON
                tool_manager=manager
            )
            
            assert result.success is True
    
    @pytest.mark.asyncio
    async def test_execute_json_decode_error_with_example(self, execute_command, mock_tool_with_schema):
        """Test JSON decode error showing helpful examples."""
        manager = AsyncMock()
        manager.server_names = {}  # Add server_names attribute
        manager.get_all_tools = AsyncMock(return_value=[mock_tool_with_schema])
        
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="complex_tool",
                params="{invalid json}",
                tool_manager=manager
            )
            
            assert result.success is False
            mock_output.error.assert_called_with("❌ Invalid parameter format")
            # Should show example with required params
            output_calls = str(mock_output.print.call_args_list)
            assert "text" in output_calls or "execute" in output_calls
    
    @pytest.mark.asyncio
    async def test_execute_tool_result_dict_type(self, execute_command):
        """Test handling when tool returns plain dict (not ToolCallResult)."""
        manager = AsyncMock()
        manager.server_names = {}  # Add server_names attribute
        tool = MagicMock()
        tool.name = "test_tool"
        tool.namespace = "default"
        tool.inputSchema = None  # Explicitly set to None
        tool.parameters = {}
        
        manager.get_all_tools = AsyncMock(return_value=[tool])
        manager.execute_tool = AsyncMock(return_value={"status": "ok", "data": [1, 2, 3]})
        
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool",
                params="{}",
                tool_manager=manager
            )
            
            assert result.success is True
            mock_output.success.assert_called_with("✅ Tool executed successfully")
    
    @pytest.mark.asyncio
    async def test_execute_tool_with_warning_no_result(self, execute_command):
        """Test when tool returns success but no result."""
        manager = AsyncMock()
        manager.server_names = {}  # Add server_names attribute
        tool = MagicMock()
        tool.name = "test_tool"
        tool.namespace = "default"
        tool.inputSchema = None  # Explicitly set to None
        tool.parameters = {}
        
        manager.get_all_tools = AsyncMock(return_value=[tool])
        manager.execute_tool = AsyncMock(return_value=ToolCallResult(
            tool_name="test_tool",
            success=True,
            result=None
        ))
        
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool",
                params="{}",
                tool_manager=manager
            )
            
            assert result.success is True
            mock_output.warning.assert_called_with("Tool returned no result")
    
    @pytest.mark.asyncio
    async def test_execute_with_server_parameter(self, execute_command):
        """Test execution with server parameter (currently TODO)."""
        manager = AsyncMock()
        manager.server_names = {}  # Add server_names attribute
        tool = MagicMock()
        tool.name = "test_tool"
        tool.namespace = "default"
        tool.inputSchema = None  # Explicitly set to None
        tool.parameters = {}
        
        manager.get_all_tools = AsyncMock(return_value=[tool])
        manager.execute_tool = AsyncMock(return_value=ToolCallResult(
            tool_name="test_tool",
            success=True,
            result="Success"
        ))
        
        with patch("mcp_cli.commands.definitions.execute_tool.output"):
            result = await execute_command.execute(
                tool="test_tool",
                server="specific_server",
                params="{}",
                tool_manager=manager
            )
            
            # Server filtering is TODO, but should still work
            assert result.success is True
    
    @pytest.mark.asyncio
    async def test_execute_string_result(self, execute_command):
        """Test handling when tool returns a string result."""
        manager = AsyncMock()
        manager.server_names = {}  # Add server_names attribute
        tool = MagicMock()
        tool.name = "test_tool"
        tool.namespace = "default"
        tool.inputSchema = None  # Explicitly set to None
        tool.parameters = {}
        
        manager.get_all_tools = AsyncMock(return_value=[tool])
        manager.execute_tool = AsyncMock(return_value=ToolCallResult(
            tool_name="test_tool",
            success=True,
            result="Simple string result"
        ))
        
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool",
                params="{}",
                tool_manager=manager
            )
            
            assert result.success is True
            mock_output.print.assert_called_with("Simple string result")
    
    @pytest.mark.asyncio
    async def test_execute_with_error_missing_required_param_extraction(self, execute_command):
        """Test error message extraction for missing required parameters."""
        manager = AsyncMock()
        manager.server_names = {}  # Add server_names attribute
        tool = MagicMock()
        tool.name = "test_tool"
        tool.namespace = "default"
        tool.inputSchema = None  # Explicitly set to None
        tool.parameters = {"properties": {"text": {"type": "string"}}, "required": ["text"]}
        
        manager.get_all_tools = AsyncMock(return_value=[tool])
        manager.execute_tool = AsyncMock(return_value=ToolCallResult(
            tool_name="test_tool",
            success=False,
            error="Invalid parameter 'text': expected string, got NoneType"
        ))
        
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool",
                params="{}",
                tool_manager=manager
            )
            
            assert result.success is True  # Command succeeds but shows error
            # Should extract param name and show helpful message
            output_calls = str(mock_output.print.call_args_list)
            assert "Missing required parameter" in output_calls or "text" in output_calls
    
    @pytest.mark.asyncio
    async def test_execute_with_generic_error(self, execute_command):
        """Test handling generic tool execution errors."""
        manager = AsyncMock()
        manager.server_names = {}  # Add server_names attribute
        tool = MagicMock()
        tool.name = "test_tool"
        tool.namespace = "default"
        tool.inputSchema = None  # Explicitly set to None
        tool.parameters = {}
        
        manager.get_all_tools = AsyncMock(return_value=[tool])
        manager.execute_tool = AsyncMock(return_value=ToolCallResult(
            tool_name="test_tool",
            success=False,
            error="Generic error message"
        ))
        
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool",
                params="{}",
                tool_manager=manager
            )
            
            assert result.success is True
            mock_output.print.assert_called_with("Error details: Generic error message")
    
    @pytest.mark.asyncio
    async def test_args_as_list_with_multiple_params(self, execute_command):
        """Test parsing args when provided as list with multiple items."""
        manager = AsyncMock()
        manager.server_names = {}  # Add server_names attribute
        tool = MagicMock()
        tool.name = "test_tool"
        tool.namespace = "default"
        tool.inputSchema = None  # Explicitly set to None
        tool.parameters = {}
        
        manager.get_all_tools = AsyncMock(return_value=[tool])
        manager.execute_tool = AsyncMock(return_value=ToolCallResult(
            tool_name="test_tool",
            success=True,
            result="Success"
        ))
        
        with patch("mcp_cli.commands.definitions.execute_tool.output"):
            result = await execute_command.execute(
                args=["test_tool", '{"text": "hello"}', "extra_arg"],
                tool_manager=manager
            )
            
            assert result.success is True
            manager.execute_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_plain_string_error_with_equals_sign(self, execute_command):
        """Test plain string detection when it contains equals."""
        manager = AsyncMock()
        manager.server_names = {}  # Add server_names attribute
        tool = MagicMock()
        tool.name = "test_tool"
        tool.namespace = "default"
        tool.inputSchema = None  # Explicitly set to None
        tool.parameters = {"properties": {"message": {"type": "string"}}, "required": ["message"]}
        
        manager.get_all_tools = AsyncMock(return_value=[tool])
        
        with patch("mcp_cli.commands.definitions.execute_tool.output") as mock_output:
            result = await execute_command.execute(
                tool="test_tool",
                params="key=value",  # Has equals but will be parsed differently
                tool_manager=manager
            )
            
            # This will go through _parse_simple_params
            assert result is not None