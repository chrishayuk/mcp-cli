"""OAuth configuration models."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OAuthConfig:
    """OAuth 2.0 configuration for an MCP server."""

    # OAuth endpoints
    authorization_url: str
    token_url: str

    # Client credentials
    client_id: str
    client_secret: Optional[str] = None  # Not required for public clients

    # OAuth parameters
    scopes: List[str] = field(default_factory=list)
    redirect_uri: str = "http://localhost:8080/callback"

    # PKCE support (recommended for security)
    use_pkce: bool = True

    # Additional parameters for authorization request
    extra_auth_params: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OAuthConfig":
        """Create from dictionary format."""
        return cls(
            authorization_url=data["authorization_url"],
            token_url=data["token_url"],
            client_id=data["client_id"],
            client_secret=data.get("client_secret"),
            scopes=data.get("scopes", []),
            redirect_uri=data.get("redirect_uri", "http://localhost:8080/callback"),
            use_pkce=data.get("use_pkce", True),
            extra_auth_params=data.get("extra_auth_params", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result: Dict[str, Any] = {
            "authorization_url": self.authorization_url,
            "token_url": self.token_url,
            "client_id": self.client_id,
            "scopes": self.scopes,
            "redirect_uri": self.redirect_uri,
            "use_pkce": self.use_pkce,
        }
        if self.client_secret:
            result["client_secret"] = self.client_secret
        if self.extra_auth_params:
            result["extra_auth_params"] = self.extra_auth_params
        return result


@dataclass
class OAuthTokens:
    """OAuth tokens and metadata."""

    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None

    # Metadata
    issued_at: Optional[float] = None  # Unix timestamp

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OAuthTokens":
        """Create from dictionary format."""
        return cls(
            access_token=data["access_token"],
            token_type=data.get("token_type", "Bearer"),
            expires_in=data.get("expires_in"),
            refresh_token=data.get("refresh_token"),
            scope=data.get("scope"),
            issued_at=data.get("issued_at"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result: Dict[str, Any] = {
            "access_token": self.access_token,
            "token_type": self.token_type,
        }
        if self.expires_in is not None:
            result["expires_in"] = self.expires_in
        if self.refresh_token:
            result["refresh_token"] = self.refresh_token
        if self.scope:
            result["scope"] = self.scope
        if self.issued_at:
            result["issued_at"] = self.issued_at
        return result

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
