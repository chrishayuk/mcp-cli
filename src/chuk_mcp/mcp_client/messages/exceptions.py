# chuk_mcp/mcp_client/messages/exceptions.py
"""
Custom exception classes for JSON-RPC error handling.
"""
class JSONRPCError(Exception):
    """Base class for JSON-RPC errors."""
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


class RetryableError(JSONRPCError):
    """Exception for JSON-RPC errors that can be retried."""
    pass


class NonRetryableError(JSONRPCError):
    """Exception for JSON-RPC errors that should not be retried."""
    pass