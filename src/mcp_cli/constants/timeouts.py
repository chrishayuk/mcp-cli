"""Timeout constants - no more magic numbers!"""

# Discovery timeouts
DISCOVERY_TIMEOUT = 10.0  # Provider discovery HTTP timeout

# UI timeouts
REFRESH_TIMEOUT = 1.0  # Display refresh timeout
SHUTDOWN_TIMEOUT = 0.5  # Graceful shutdown timeout

# Network timeouts (duplicated from defaults for direct usage)
DEFAULT_HTTP_CONNECT_TIMEOUT = 30.0
DEFAULT_HTTP_REQUEST_TIMEOUT = 120.0

__all__ = [
    "DISCOVERY_TIMEOUT",
    "REFRESH_TIMEOUT",
    "SHUTDOWN_TIMEOUT",
    "DEFAULT_HTTP_CONNECT_TIMEOUT",
    "DEFAULT_HTTP_REQUEST_TIMEOUT",
]
