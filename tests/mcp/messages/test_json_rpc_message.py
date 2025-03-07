# tests/mcp/test_json_rpc_message.py
from mcp.messages.json_rpc_message import JSONRPCMessage
from pydantic import ValidationError

# Check for Pydantic version to handle compatibility
try:
    import pydantic
    PYDANTIC_V2 = pydantic.__version__.startswith('2')
except (ImportError, AttributeError):
    PYDANTIC_V2 = False


class TestJSONRPCMessage:
    def test_default_initialization(self):
        """Test that JSONRPCMessage initializes with default values."""
        message = JSONRPCMessage()
        assert message.jsonrpc == "2.0"
        assert message.id is None
        assert message.method is None
        assert message.params is None
        assert message.result is None
        assert message.error is None

    def test_initialization_with_values(self):
        """Test JSONRPCMessage initialization with specific values."""
        message = JSONRPCMessage(
            id="123",
            method="test_method",
            params={"param1": "value1"},
        )
        assert message.jsonrpc == "2.0"
        assert message.id == "123"
        assert message.method == "test_method"
        assert message.params == {"param1": "value1"}
        assert message.result is None
        assert message.error is None

    def test_initialization_with_result(self):
        """Test JSONRPCMessage initialization with result."""
        message = JSONRPCMessage(
            id="123",
            result={"success": True, "data": "some_data"},
        )
        assert message.jsonrpc == "2.0"
        assert message.id == "123"
        assert message.method is None
        assert message.params is None
        assert message.result == {"success": True, "data": "some_data"}
        assert message.error is None

    def test_initialization_with_error(self):
        """Test JSONRPCMessage initialization with error."""
        message = JSONRPCMessage(
            id="123",
            error={"code": -32700, "message": "Parse error"},
        )
        assert message.jsonrpc == "2.0"
        assert message.id == "123"
        assert message.method is None
        assert message.params is None
        assert message.result is None
        assert message.error == {"code": -32700, "message": "Parse error"}

    def test_to_dict(self):
        """Test conversion of JSONRPCMessage to dictionary."""
        message = JSONRPCMessage(
            id="123",
            method="test_method",
            params={"param1": "value1"},
        )
        # Use model_dump() for Pydantic v2 compatibility
        # Fall back to dict() if model_dump isn't available
        if hasattr(message, 'model_dump'):
            message_dict = message.model_dump()
        else:
            message_dict = message.dict()
            
        assert message_dict["jsonrpc"] == "2.0"
        assert message_dict["id"] == "123"
        assert message_dict["method"] == "test_method"
        assert message_dict["params"] == {"param1": "value1"}
        assert "result" in message_dict
        assert "error" in message_dict

    def test_to_json(self):
        """Test conversion of JSONRPCMessage to JSON."""
        message = JSONRPCMessage(
            id="123",
            method="test_method",
            params={"param1": "value1"},
        )
        # Use model_dump_json() for Pydantic v2 compatibility
        # Fall back to json() if model_dump_json isn't available
        if hasattr(message, 'model_dump_json'):
            json_str = message.model_dump_json()
        else:
            json_str = message.json()
            
        assert isinstance(json_str, str)
        # Check for content without assuming exact formatting
        assert '"jsonrpc":"2.0"' in json_str
        assert '"id":"123"' in json_str
        assert '"method":"test_method"' in json_str
        assert '"param1":"value1"' in json_str

    def test_extra_fields(self):
        """Test that extra fields are allowed."""
        message = JSONRPCMessage(
            id="123",
            method="test_method",
            extra_field="extra_value",
            another_field=123,
        )
        assert message.jsonrpc == "2.0"
        assert message.id == "123"
        assert message.method == "test_method"
        assert hasattr(message, "extra_field")
        assert message.extra_field == "extra_value"
        assert hasattr(message, "another_field")
        assert message.another_field == 123

    def test_nested_params(self):
        """Test with nested parameters."""
        nested_params = {
            "user": {
                "name": "John Doe",
                "age": 30,
                "address": {
                    "street": "123 Main St",
                    "city": "Anytown",
                    "zip": "12345"
                },
                "hobbies": ["reading", "gaming", "hiking"]
            },
            "options": {
                "verbose": True,
                "count": 5
            }
        }
        
        message = JSONRPCMessage(
            id="456",
            method="complex_method",
            params=nested_params
        )
        
        assert message.params == nested_params
        assert message.params["user"]["name"] == "John Doe"
        assert message.params["user"]["address"]["city"] == "Anytown"
        assert message.params["user"]["hobbies"][2] == "hiking"
        assert message.params["options"]["verbose"] is True

    def test_complex_result(self):
        """Test with complex result structure."""
        complex_result = {
            "status": "success",
            "data": [
                {"id": 1, "name": "Item 1", "active": True},
                {"id": 2, "name": "Item 2", "active": False},
                {"id": 3, "name": "Item 3", "active": True}
            ],
            "pagination": {
                "total": 10,
                "page": 1,
                "limit": 3
            }
        }
        
        message = JSONRPCMessage(
            id="789",
            result=complex_result
        )
        
        assert message.result == complex_result
        assert message.result["status"] == "success"
        assert len(message.result["data"]) == 3
        assert message.result["data"][0]["name"] == "Item 1"
        assert message.result["pagination"]["total"] == 10

    def test_standard_errors(self):
        """Test with standard JSON-RPC error codes."""
        standard_errors = [
            {"code": -32700, "message": "Parse error"},
            {"code": -32600, "message": "Invalid Request"},
            {"code": -32601, "message": "Method not found"},
            {"code": -32602, "message": "Invalid params"},
            {"code": -32603, "message": "Internal error"}
        ]
        
        for error in standard_errors:
            message = JSONRPCMessage(
                id="error_test",
                error=error
            )
            assert message.error == error
            assert message.error["code"] == error["code"]
            assert message.error["message"] == error["message"]

    def test_error_with_data(self):
        """Test error with additional data."""
        error_with_data = {
            "code": -32000,
            "message": "Server error",
            "data": {
                "exception": "ValueError",
                "trace": "File 'app.py', line 42",
                "timestamp": "2025-03-05T12:34:56Z"
            }
        }
        
        message = JSONRPCMessage(
            id="data_error",
            error=error_with_data
        )
        
        assert message.error == error_with_data
        assert message.error["code"] == -32000
        assert message.error["data"]["exception"] == "ValueError"
        assert message.error["data"]["timestamp"] == "2025-03-05T12:34:56Z"

    def test_request_and_response_roundtrip(self):
        """Test creating a request and then a response."""
        # Create request
        request = JSONRPCMessage(
            id="req123",
            method="get_user",
            params={"user_id": 42}
        )
        
        # Create successful response
        success_response = JSONRPCMessage(
            id=request.id,
            result={"id": 42, "name": "Jane Smith", "email": "jane@example.com"}
        )
        
        assert success_response.id == request.id
        assert success_response.result["name"] == "Jane Smith"
        
        # Create error response
        error_response = JSONRPCMessage(
            id=request.id,
            error={"code": -32001, "message": "User not found"}
        )
        
        assert error_response.id == request.id
        assert error_response.error["code"] == -32001

    def test_notification(self):
        """Test JSON-RPC notification (no id)."""
        notification = JSONRPCMessage(
            method="update",
            params={"status": "completed"}
        )
        
        assert notification.id is None
        assert notification.method == "update"
        assert notification.params["status"] == "completed"