# tests/mcp/transport/stdio/test_stdio_server_parameters.py
import pytest
from pydantic import ValidationError

from mcp.transport.stdio.stdio_server_parameters import StdioServerParameters

def test_stdio_server_parameters_creation():
    """Test creating StdioServerParameters with valid data."""
    # Test with minimum required fields
    params = StdioServerParameters(command="python")
    assert params.command == "python"
    assert params.args == []
    assert params.env is None
    
    # Test with all fields
    params = StdioServerParameters(
        command="python",
        args=["-m", "mcp.server"],
        env={"TEST_ENV": "value"}
    )
    assert params.command == "python"
    assert params.args == ["-m", "mcp.server"]
    assert params.env == {"TEST_ENV": "value"}


def test_stdio_server_parameters_validation():
    """Test validation of StdioServerParameters."""
    # Test with missing required field
    with pytest.raises(ValidationError):
        StdioServerParameters()
    
    # Test with empty command (should pass validation but might be rejected by application logic)
    params = StdioServerParameters(command="")
    assert params.command == ""
    
    # Test with non-list args
    with pytest.raises(ValidationError):
        StdioServerParameters(command="python", args="not-a-list")
    
    # Test with non-string command
    with pytest.raises(ValidationError):
        StdioServerParameters(command=123)
    
    # Test with invalid env type
    with pytest.raises(ValidationError):
        StdioServerParameters(command="python", env="not-a-dict")


def test_stdio_server_parameters_default_factory():
    """Test the default factory for args field."""
    # Create two instances to ensure the default factory creates new instances
    params1 = StdioServerParameters(command="python")
    params2 = StdioServerParameters(command="python")
    
    # Modify one instance's args
    params1.args.append("arg1")
    
    # Verify the other instance is not affected
    assert params1.args == ["arg1"]
    assert params2.args == []


def test_stdio_server_parameters_model_dump():
    """Test the model_dump method."""
    params = StdioServerParameters(
        command="python",
        args=["-m", "mcp.server"],
        env={"TEST_ENV": "value"}
    )
    
    # Check model_dump
    dump = params.model_dump()
    assert dump == {
        "command": "python",
        "args": ["-m", "mcp.server"],
        "env": {"TEST_ENV": "value"}
    }
    
    # Check model_dump with exclude
    dump = params.model_dump(exclude={"env"})
    assert dump == {
        "command": "python",
        "args": ["-m", "mcp.server"]
    }
    
    # Check model_dump_json
    json_str = params.model_dump_json()
    assert '"command":"python"' in json_str
    assert '"args":["-m","mcp.server"]' in json_str
    assert '"env":{"TEST_ENV":"value"}' in json_str