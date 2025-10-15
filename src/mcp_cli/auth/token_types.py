"""Token types and models for secure storage."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class TokenType(str, Enum):
    """Types of tokens that can be stored."""

    OAUTH = "oauth"  # OAuth tokens with refresh
    BEARER = "bearer"  # Simple bearer tokens
    API_KEY = "api_key"  # API keys for providers
    BASIC_AUTH = "basic_auth"  # Basic authentication credentials


@dataclass
class StoredToken:
    """Generic token storage model."""

    token_type: TokenType
    name: str  # Identifier (server name, provider name, etc.)
    data: Dict[str, Any]  # Token-specific data
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "token_type": self.token_type.value,
            "name": self.name,
            "data": self.data,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredToken":
        """Create from dictionary format."""
        return cls(
            token_type=TokenType(data["token_type"]),
            name=data["name"],
            data=data["data"],
            metadata=data.get("metadata"),
        )

    def get_display_info(self) -> Dict[str, Any]:
        """Get safe display information (no sensitive data)."""
        info: Dict[str, Any] = {
            "name": self.name,
            "type": self.token_type.value,
        }

        # Add type-specific safe metadata
        if self.token_type == TokenType.OAUTH:
            info["has_refresh_token"] = "refresh_token" in self.data
            if "expires_at" in self.data:
                info["expires_at"] = self.data["expires_at"]
            if "issued_at" in self.data:
                info["issued_at"] = self.data["issued_at"]

        elif self.token_type == TokenType.BEARER:
            info["has_token"] = "token" in self.data
            if "expires_at" in self.data:
                info["expires_at"] = self.data["expires_at"]

        elif self.token_type == TokenType.API_KEY:
            info["provider"] = self.data.get("provider", "unknown")
            # Show masked version of key if present
            if "key" in self.data:
                key = self.data["key"]
                if len(key) > 8:
                    info["key_preview"] = f"{key[:4]}...{key[-4:]}"
                else:
                    info["key_preview"] = "****"

        elif self.token_type == TokenType.BASIC_AUTH:
            info["username"] = self.data.get("username", "unknown")

        # Add custom metadata
        if self.metadata:
            info["metadata"] = self.metadata

        return info


@dataclass
class BearerToken:
    """Simple bearer token."""

    token: str
    expires_at: Optional[float] = None

    def to_stored_token(self, name: str) -> StoredToken:
        """Convert to StoredToken format."""
        data: Dict[str, Any] = {"token": self.token}
        if self.expires_at:
            data["expires_at"] = self.expires_at

        return StoredToken(
            token_type=TokenType.BEARER,
            name=name,
            data=data,
        )

    @classmethod
    def from_stored_token(cls, stored: StoredToken) -> "BearerToken":
        """Create from StoredToken format."""
        if stored.token_type != TokenType.BEARER:
            raise ValueError(f"Expected BEARER token, got {stored.token_type}")

        return cls(
            token=stored.data["token"],
            expires_at=stored.data.get("expires_at"),
        )

    def is_expired(self, buffer_seconds: int = 300) -> bool:
        """Check if token is expired (with buffer)."""
        if self.expires_at is None:
            return False

        import time

        return time.time() >= (self.expires_at - buffer_seconds)


@dataclass
class APIKeyToken:
    """API key for provider authentication."""

    provider: str  # openai, anthropic, google, etc.
    key: str
    organization_id: Optional[str] = None
    project_id: Optional[str] = None

    def to_stored_token(self, name: str) -> StoredToken:
        """Convert to StoredToken format."""
        data = {
            "provider": self.provider,
            "key": self.key,
        }
        if self.organization_id:
            data["organization_id"] = self.organization_id
        if self.project_id:
            data["project_id"] = self.project_id

        return StoredToken(
            token_type=TokenType.API_KEY,
            name=name,
            data=data,
        )

    @classmethod
    def from_stored_token(cls, stored: StoredToken) -> "APIKeyToken":
        """Create from StoredToken format."""
        if stored.token_type != TokenType.API_KEY:
            raise ValueError(f"Expected API_KEY token, got {stored.token_type}")

        return cls(
            provider=stored.data["provider"],
            key=stored.data["key"],
            organization_id=stored.data.get("organization_id"),
            project_id=stored.data.get("project_id"),
        )


@dataclass
class BasicAuthToken:
    """Basic authentication credentials."""

    username: str
    password: str

    def to_stored_token(self, name: str) -> StoredToken:
        """Convert to StoredToken format."""
        return StoredToken(
            token_type=TokenType.BASIC_AUTH,
            name=name,
            data={
                "username": self.username,
                "password": self.password,
            },
        )

    @classmethod
    def from_stored_token(cls, stored: StoredToken) -> "BasicAuthToken":
        """Create from StoredToken format."""
        if stored.token_type != TokenType.BASIC_AUTH:
            raise ValueError(f"Expected BASIC_AUTH token, got {stored.token_type}")

        return cls(
            username=stored.data["username"],
            password=stored.data["password"],
        )

    def get_auth_header(self) -> str:
        """Get Basic Auth header value."""
        import base64

        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"
