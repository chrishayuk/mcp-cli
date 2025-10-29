# mcp_cli/auth/oauth_config.py
"""OAuth configuration models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OAuthConfig(BaseModel):
    """OAuth 2.0 configuration for an MCP server."""

    # OAuth endpoints
    authorization_url: str
    token_url: str

    # Client credentials
    client_id: str
    client_secret: Optional[str] = None  # Not required for public clients

    # OAuth parameters
    scopes: List[str] = Field(default_factory=list)
    redirect_uri: str = "http://localhost:8080/callback"

    # PKCE support (recommended for security)
    use_pkce: bool = True

    # Additional parameters for authorization request
    extra_auth_params: Dict[str, str] = Field(default_factory=dict)

    model_config = {"frozen": False}


class OAuthTokens(BaseModel):
    """OAuth tokens and metadata."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None

    # Metadata
    issued_at: Optional[float] = None  # Unix timestamp

    model_config = {"frozen": False}

    def is_expired(self) -> bool:
        """Check if token is expired."""
        if not self.expires_in or not self.issued_at:
            return False

        import time

        age = time.time() - self.issued_at
        # Consider expired if within 5 minutes of expiry
        return age >= (self.expires_in - 300)

    def get_authorization_header(self) -> str:
        """Get the Authorization header value."""
        # Ensure Bearer is capitalized per RFC 6750
        token_type = (
            self.token_type.capitalize()
            if self.token_type.lower() == "bearer"
            else self.token_type
        )
        return f"{token_type} {self.access_token}"
