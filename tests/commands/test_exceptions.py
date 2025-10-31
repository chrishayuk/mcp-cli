"""Tests for command exceptions."""

from mcp_cli.commands.exceptions import (
    CommandError,
    InvalidParameterError,
    CommandExecutionError,
    CommandNotFoundError,
    ValidationError,
)


class TestCommandError:
    """Test CommandError base exception."""

    def test_basic_error(self):
        """Test basic error with just message."""
        error = CommandError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.command is None

    def test_error_with_command(self):
        """Test error with command name."""
        error = CommandError("Something went wrong", command="test-command")
        assert error.message == "Something went wrong"
        assert error.command == "test-command"


class TestInvalidParameterError:
    """Test InvalidParameterError exception."""

    def test_basic_parameter_error(self):
        """Test basic parameter error."""
        error = InvalidParameterError("Invalid value")
        assert str(error) == "Invalid value"
        assert error.message == "Invalid value"
        assert error.parameter is None
        assert error.command is None

    def test_parameter_error_with_parameter(self):
        """Test parameter error with parameter name."""
        error = InvalidParameterError("Invalid value", parameter="test_param")
        assert error.message == "Invalid value"
        assert error.parameter == "test_param"
        assert error.command is None

    def test_parameter_error_with_command(self):
        """Test parameter error with command and parameter."""
        error = InvalidParameterError(
            "Invalid value", parameter="test_param", command="test-command"
        )
        assert error.message == "Invalid value"
        assert error.parameter == "test_param"
        assert error.command == "test-command"


class TestCommandExecutionError:
    """Test CommandExecutionError exception."""

    def test_basic_execution_error(self):
        """Test basic execution error."""
        error = CommandExecutionError("Execution failed")
        assert str(error) == "Execution failed"
        assert error.message == "Execution failed"
        assert error.command is None
        assert error.cause is None

    def test_execution_error_with_command(self):
        """Test execution error with command name."""
        error = CommandExecutionError("Execution failed", command="test-command")
        assert error.message == "Execution failed"
        assert error.command == "test-command"
        assert error.cause is None

    def test_execution_error_with_cause(self):
        """Test execution error with original cause."""
        original = ValueError("Original error")
        error = CommandExecutionError(
            "Execution failed", command="test-command", cause=original
        )
        assert error.message == "Execution failed"
        assert error.command == "test-command"
        assert error.cause is original


class TestCommandNotFoundError:
    """Test CommandNotFoundError exception."""

    def test_command_not_found(self):
        """Test command not found error."""
        error = CommandNotFoundError("Command not found", command="missing-command")
        assert str(error) == "Command not found"
        assert error.message == "Command not found"
        assert error.command == "missing-command"


class TestValidationError:
    """Test ValidationError exception."""

    def test_basic_validation_error(self):
        """Test basic validation error."""
        error = ValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert error.message == "Validation failed"
        assert error.errors == []
        assert error.command is None

    def test_validation_error_with_errors(self):
        """Test validation error with error list."""
        errors = ["Error 1", "Error 2"]
        error = ValidationError("Validation failed", errors=errors)
        assert error.message == "Validation failed"
        assert error.errors == errors
        assert error.command is None

    def test_validation_error_with_command(self):
        """Test validation error with command name."""
        errors = ["Error 1"]
        error = ValidationError(
            "Validation failed", errors=errors, command="test-command"
        )
        assert error.message == "Validation failed"
        assert error.errors == errors
        assert error.command == "test-command"

    def test_validation_error_none_errors(self):
        """Test validation error with None errors defaults to empty list."""
        error = ValidationError("Validation failed", errors=None)
        assert error.errors == []
