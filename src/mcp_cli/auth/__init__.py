# mcp_cli/auth/__init__.py
"""Authentication and OAuth support for MCP CLI."""

# Import OAuth functionality from chuk-mcp-client-oauth library
from chuk_mcp_client_oauth import (
    OAuthConfig,
    OAuthTokens,
    OAuthFlow,
    TokenManager,
    TokenStoreBackend,
    TokenStoreFactory,
    SecureTokenStore,
    MCPOAuthClient,
    OAuthHandler,
    DynamicClientRegistration,
    TokenType,
    StoredToken,
    APIKeyToken,
    BearerToken,
    TokenRegistry,
)

# Import MCP-CLI specific provider token functionality (kept local)
from .provider_tokens import (
    get_provider_token_with_hierarchy,
    check_provider_token_status,
    set_provider_token,
    delete_provider_token,
    get_provider_env_var_name,
    get_provider_token_display_status,
    list_all_provider_tokens,
)

__all__ = [
    # OAuth library exports
    "OAuthConfig",
    "OAuthTokens",
    "OAuthFlow",
    "TokenManager",
    "TokenStoreBackend",
    "TokenStoreFactory",
    "SecureTokenStore",
    "MCPOAuthClient",
    "OAuthHandler",
    "DynamicClientRegistration",
    "TokenType",
    "StoredToken",
    "APIKeyToken",
    "BearerToken",
    "TokenRegistry",
    # MCP-CLI specific exports
    "get_provider_token_with_hierarchy",
    "check_provider_token_status",
    "set_provider_token",
    "delete_provider_token",
    "get_provider_env_var_name",
    "get_provider_token_display_status",
    "list_all_provider_tokens",
]
