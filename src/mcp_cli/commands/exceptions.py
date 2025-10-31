# src/mcp_cli/commands/exceptions.py
"""Custom exceptions for command operations."""

from __future__ import annotations


class CommandError(Exception):
    """Base exception for all command errors."""

    def __init__(self, message: str, command: str | None = None):
        """
        Initialize command error.

        Args:
            message: Error message
            command: Command name that failed (optional)
        """
        self.message = message
        self.command = command
        super().__init__(message)


class InvalidParameterError(CommandError):
    """Invalid parameter provided to command."""

    def __init__(
        self, message: str, parameter: str | None = None, command: str | None = None
    ):
        """
        Initialize invalid parameter error.

        Args:
            message: Error message
            parameter: Parameter name that is invalid
            command: Command name
        """
        self.parameter = parameter
        super().__init__(message, command)


class CommandExecutionError(CommandError):
    """Command execution failed."""

    def __init__(
        self, message: str, command: str | None = None, cause: Exception | None = None
    ):
        """
        Initialize command execution error.

        Args:
            message: Error message
            command: Command name that failed
            cause: Original exception that caused this error
        """
        self.cause = cause
        super().__init__(message, command)


class CommandNotFoundError(CommandError):
    """Requested command was not found."""

    pass


class ValidationError(CommandError):
    """Command parameter validation failed."""

    def __init__(
        self, message: str, errors: list | None = None, command: str | None = None
    ):
        """
        Initialize validation error.

        Args:
            message: Error message
            errors: List of validation errors
            command: Command name
        """
        self.errors = errors or []
        super().__init__(message, command)
