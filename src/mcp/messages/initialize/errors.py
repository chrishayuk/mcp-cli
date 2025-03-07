# messages/initialize/errors.py
from typing import List

class VersionMismatchError(Exception):
    """Error raised when client and server protocol versions don't match."""
    def __init__(self, requested: str, supported: List[str]):
        self.requested = requested
        self.supported = supported
        super().__init__(f"Protocol version mismatch. Requested: {requested}, Supported: {supported}")

