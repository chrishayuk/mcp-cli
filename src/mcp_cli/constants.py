"""Central constants for MCP CLI."""

from importlib.metadata import version, PackageNotFoundError

# Application namespace for token storage
# This is used to namespace all tokens stored by mcp-cli
# to avoid conflicts with other applications using the same libraries
NAMESPACE = "mcp-cli"

# Token type namespaces
OAUTH_NAMESPACE = NAMESPACE  # OAuth tokens for MCP servers
PROVIDER_NAMESPACE = "provider"  # LLM provider API keys
GENERIC_NAMESPACE = "generic"  # Generic bearer tokens and API keys

# Application metadata
APP_NAME = "mcp-cli"

# Get version from package metadata
try:
    APP_VERSION = version("mcp-cli")
except PackageNotFoundError:
    # Fallback for development/editable installs
    APP_VERSION = "0.0.0-dev"
