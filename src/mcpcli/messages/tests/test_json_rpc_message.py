import pytest
from mcpcli.messages.message_types.json_rpc_message import JSONRPCMessage


def test_create_basic_message():
    """Test creating a basic JSONRPC message with minimal fields."""
    message = JSONRPCMessage()
    assert message.jsonrpc == "2.0"
    assert message.id is None
    assert message.method is None
    assert message.params is None
    assert message.result is None
    assert message.error is None


def test_create_request_message():
    """Test creating a JSONRPC request message with method and params."""
    message = JSONRPCMessage(
        id="123",
        method="test_method",
        params={"key": "value"}
    )
    assert message.jsonrpc == "2.0"
    assert message.id == "123"
    assert message.method == "test_method"
    assert message.params == {"key": "value"}
    assert message.result is None
    assert message.error is None


def test_create_response_message():
    """Test creating a JSONRPC response message with result."""
    message = JSONRPCMessage(
        id="123",
        result={"status": "success"}
    )
    assert message.jsonrpc == "2.0"
    assert message.id == "123"
    assert message.method is None
    assert message.params is None
    assert message.result == {"status": "success"}
    assert message.error is None


def test_create_error_message():
    """Test creating a JSONRPC error message."""
    error_data = {
        "code": -32600,
        "message": "Invalid Request",
        "data": {"details": "Missing required field"}
    }
    message = JSONRPCMessage(
        id="123",
        error=error_data
    )
    assert message.jsonrpc == "2.0"
    assert message.id == "123"
    assert message.method is None
    assert message.params is None
    assert message.result is None
    assert message.error == error_data


def test_extra_fields():
    """Test that extra fields are allowed due to Config.extra = 'allow'."""
    extra_data = {
        "extra_field": "extra_value",
        "another_field": 123
    }
    message = JSONRPCMessage(
        id="123",
        method="test_method",
        **extra_data
    )
    assert message.jsonrpc == "2.0"
    assert message.id == "123"
    assert message.method == "test_method"
    assert hasattr(message, "extra_field")
    assert message.extra_field == "extra_value"
    assert hasattr(message, "another_field")
    assert message.another_field == 123


def test_model_serialization():
    """Test that the model can be serialized to and from JSON correctly."""
    original_data = {
        "jsonrpc": "2.0",
        "id": "123",
        "method": "test_method",
        "params": {"key": "value"},
        "extra_field": "extra_value"
    }
    
    # Create message from dict
    message = JSONRPCMessage(**original_data)
    
    # Convert to dict (using model_dump instead of deprecated dict method)
    serialized = message.model_dump(exclude_unset=True)
    
    # Check all fields are preserved
    for key, value in original_data.items():
        assert serialized[key] == value


def test_custom_jsonrpc_version():
    """Test that the jsonrpc field can be set to a custom value."""
    message = JSONRPCMessage(jsonrpc="1.0")
    assert message.jsonrpc == "1.0"


@pytest.mark.parametrize("field,value", [
    ("id", "test-id"),
    ("method", "test-method"),
    ("params", {"test": "param"}),
    ("result", {"test": "result"}),
    ("error", {"code": -32600, "message": "error"})
])
def test_optional_fields(field, value):
    """Test setting each optional field individually."""
    message_data = {field: value}
    message = JSONRPCMessage(**message_data)
    assert getattr(message, field) == value