"""Central constants for MCP CLI."""

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
APP_VERSION = "0.1.0"  # TODO: Get from package metadata
