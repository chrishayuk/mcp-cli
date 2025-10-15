"""MCP OAuth 2.0 implementation following the MCP authorization specification.

This implements:
- OAuth Authorization Server Metadata discovery (RFC 8414)
- Dynamic Client Registration (RFC 7591)
- Authorization Code Flow with PKCE
- Token management and refresh
"""

import asyncio
import base64
import hashlib
import secrets
import urllib.parse
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import httpx

from .oauth_config import OAuthTokens


@dataclass
class MCPAuthorizationMetadata:
    """OAuth Authorization Server Metadata from .well-known endpoint."""

    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: Optional[str] = None
    scopes_supported: Optional[list[str]] = None
    response_types_supported: Optional[list[str]] = None
    grant_types_supported: Optional[list[str]] = None
    code_challenge_methods_supported: Optional[list[str]] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "MCPAuthorizationMetadata":
        """Create from OAuth discovery response."""
        return cls(
            authorization_endpoint=data["authorization_endpoint"],
            token_endpoint=data["token_endpoint"],
            registration_endpoint=data.get("registration_endpoint"),
            scopes_supported=data.get("scopes_supported", []),
            response_types_supported=data.get("response_types_supported", ["code"]),
            grant_types_supported=data.get(
                "grant_types_supported", ["authorization_code", "refresh_token"]
            ),
            code_challenge_methods_supported=data.get(
                "code_challenge_methods_supported", ["S256"]
            ),
        )


@dataclass
class DynamicClientRegistration:
    """OAuth client credentials from dynamic registration."""

    client_id: str
    client_secret: Optional[str] = None
    client_id_issued_at: Optional[int] = None
    client_secret_expires_at: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "DynamicClientRegistration":
        """Create from registration response."""
        return cls(
            client_id=data["client_id"],
            client_secret=data.get("client_secret"),
            client_id_issued_at=data.get("client_id_issued_at"),
            client_secret_expires_at=data.get("client_secret_expires_at", 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        result: Dict[str, Any] = {"client_id": self.client_id}
        if self.client_secret:
            result["client_secret"] = self.client_secret
        if self.client_id_issued_at:
            result["client_id_issued_at"] = self.client_id_issued_at
        if self.client_secret_expires_at:
            result["client_secret_expires_at"] = self.client_secret_expires_at
        return result


class MCPOAuthClient:
    """MCP OAuth client following the MCP authorization specification."""

    def __init__(
        self, server_url: str, redirect_uri: str = "http://localhost:8080/callback"
    ):
        """
        Initialize MCP OAuth client.

        Args:
            server_url: Base URL of the MCP server (e.g., https://mcp.notion.com/mcp)
            redirect_uri: OAuth callback URI
        """
        self.server_url = server_url.rstrip("/")
        self.redirect_uri = redirect_uri
        self._auth_metadata: Optional[MCPAuthorizationMetadata] = None
        self._client_registration: Optional[DynamicClientRegistration] = None
        self._code_verifier: Optional[str] = None
        self._auth_result: Optional[Dict[str, str]] = None

    async def discover_authorization_server(self) -> MCPAuthorizationMetadata:
        """
        Discover OAuth authorization server metadata.

        Per MCP spec, this is at /.well-known/oauth-authorization-server
        relative to the server URL root.

        Returns:
            Authorization server metadata
        """
        # Extract the base URL (scheme + host)
        parsed = urllib.parse.urlparse(self.server_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # Discovery endpoint is at the root
        discovery_url = urljoin(base_url, "/.well-known/oauth-authorization-server")

        async with httpx.AsyncClient() as client:
            response = await client.get(discovery_url)
            response.raise_for_status()
            metadata = MCPAuthorizationMetadata.from_dict(response.json())
            self._auth_metadata = metadata
            return metadata

    async def register_client(
        self, client_name: str = "MCP CLI", redirect_uris: Optional[list[str]] = None
    ) -> DynamicClientRegistration:
        """
        Perform Dynamic Client Registration (RFC 7591).

        Args:
            client_name: Name of the OAuth client
            redirect_uris: List of redirect URIs (defaults to self.redirect_uri)

        Returns:
            Client registration credentials
        """
        if not self._auth_metadata:
            await self.discover_authorization_server()

        if not self._auth_metadata or not self._auth_metadata.registration_endpoint:
            raise ValueError("Server does not support dynamic client registration")

        if redirect_uris is None:
            redirect_uris = [self.redirect_uri]

        registration_data = {
            "client_name": client_name,
            "redirect_uris": redirect_uris,
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",  # Public client
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._auth_metadata.registration_endpoint,
                json=registration_data,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            registration = DynamicClientRegistration.from_dict(response.json())
            self._client_registration = registration
            return registration

    def _generate_pkce_pair(self) -> tuple[str, str]:
        """Generate PKCE code verifier and challenge."""
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(
            "utf-8"
        )
        code_verifier = code_verifier.rstrip("=")

        challenge_bytes = hashlib.sha256(code_verifier.encode("utf-8")).digest()
        code_challenge = base64.urlsafe_b64encode(challenge_bytes).decode("utf-8")
        code_challenge = code_challenge.rstrip("=")

        return code_verifier, code_challenge

    def get_authorization_url(self, scopes: Optional[list[str]] = None) -> str:
        """
        Generate authorization URL for user consent.

        Args:
            scopes: List of OAuth scopes to request

        Returns:
            Authorization URL
        """
        if not self._auth_metadata or not self._client_registration:
            raise ValueError("Must discover and register before authorization")

        # Generate PKCE parameters
        self._code_verifier, code_challenge = self._generate_pkce_pair()

        params = {
            "client_id": self._client_registration.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": secrets.token_urlsafe(16),
        }

        if scopes:
            params["scope"] = " ".join(scopes)

        query_string = urllib.parse.urlencode(params)
        return f"{self._auth_metadata.authorization_endpoint}?{query_string}"

    def _create_callback_handler(self):
        """Create HTTP callback handler."""
        oauth_client = self

        class CallbackHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Log to stdout for debugging
                print(f"[Callback Server] {format % args}")

            def do_GET(self):
                print(f"[Callback Server] Received request: {self.path}")
                parsed = urllib.parse.urlparse(self.path)

                # Ignore non-callback requests (favicon, etc.)
                if not parsed.path.startswith("/callback"):
                    self.send_response(404)
                    self.send_header("Content-type", "text/html")
                    self.end_headers()
                    self.wfile.write(b"<html><body>Not Found</body></html>")
                    return

                params = dict(urllib.parse.parse_qsl(parsed.query))
                print(f"[Callback Server] Query params: {params}")

                # Only set _auth_result if we haven't already got a successful result
                if oauth_client._auth_result is None:
                    if "error" in params:
                        error_description = params.get(
                            "error_description", params["error"]
                        )
                        oauth_client._auth_result = {"error": error_description}
                        response = f"<html><body><h1>Authorization Failed</h1><p>{error_description}</p></body></html>"
                        self.send_response(400)
                    elif "code" in params:
                        oauth_client._auth_result = params
                        response = "<html><body><h1>Authorization Successful</h1><p>You can close this window and return to the terminal.</p></body></html>"
                        self.send_response(200)
                        print(
                            "[Callback Server] Authorization successful, code received"
                        )
                    else:
                        # Invalid callback request (no code, no error)
                        response = "<html><body><h1>Invalid Callback</h1><p>No authorization code received</p></body></html>"
                        self.send_response(400)
                else:
                    # Already got result, just return success page
                    response = "<html><body><h1>Authorization Successful</h1><p>You can close this window.</p></body></html>"
                    self.send_response(200)

                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(response.encode())

        return CallbackHandler

    async def _run_callback_server(self, port: int) -> None:
        """Run local callback server."""
        handler_class = self._create_callback_handler()
        server = HTTPServer(("localhost", port), handler_class)

        print(f"[Callback Server] Starting server on localhost:{port}")
        server_thread = Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        print("[Callback Server] Server started, waiting for callback...")

        # Wait for callback or timeout (5 minutes)
        timeout_seconds = 300
        for i in range(timeout_seconds):
            if self._auth_result is not None:
                print(f"[Callback Server] Callback received after {i} seconds")
                break
            await asyncio.sleep(1)

            # Print progress every 30 seconds
            if i > 0 and i % 30 == 0:
                remaining = timeout_seconds - i
                print(f"[Callback Server] Still waiting... ({remaining}s remaining)")

        if self._auth_result is None:
            print(f"[Callback Server] Timeout after {timeout_seconds} seconds")

        print("[Callback Server] Shutting down server...")
        server.shutdown()
        print("[Callback Server] Server stopped")

    async def exchange_code_for_token(self, code: str) -> OAuthTokens:
        """
        Exchange authorization code for access token.

        Args:
            code: Authorization code from callback

        Returns:
            OAuth tokens
        """
        if not self._auth_metadata or not self._client_registration:
            raise ValueError("Must discover and register before token exchange")

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.redirect_uri,
            "client_id": self._client_registration.client_id,
            "code_verifier": self._code_verifier,
        }

        if self._client_registration.client_secret:
            data["client_secret"] = self._client_registration.client_secret

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._auth_metadata.token_endpoint,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            return OAuthTokens.from_dict(response.json())

    async def refresh_token(self, refresh_token: str) -> OAuthTokens:
        """
        Refresh access token.

        Args:
            refresh_token: Refresh token

        Returns:
            New OAuth tokens
        """
        if not self._auth_metadata or not self._client_registration:
            raise ValueError("Must discover and register before token refresh")

        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self._client_registration.client_id,
        }

        if self._client_registration.client_secret:
            data["client_secret"] = self._client_registration.client_secret

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._auth_metadata.token_endpoint,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            return OAuthTokens.from_dict(response.json())

    async def authorize(self, scopes: Optional[list[str]] = None) -> OAuthTokens:
        """
        Perform full MCP OAuth authorization flow.

        This includes:
        1. Discovery of authorization server metadata
        2. Dynamic client registration
        3. User authorization via browser
        4. Token exchange

        Args:
            scopes: OAuth scopes to request

        Returns:
            OAuth tokens
        """
        # Step 1: Discover authorization server
        print("\n🔍 Discovering authorization server...")
        await self.discover_authorization_server()

        # Step 2: Register as OAuth client
        print("📝 Registering OAuth client...")
        await self.register_client()

        # Step 3: Get authorization from user
        print("🔐 Opening browser for authorization...")
        auth_url = self.get_authorization_url(scopes)
        print(f"If browser doesn't open, visit: {auth_url}\n")
        webbrowser.open(auth_url)

        # Step 4: Wait for callback
        parsed = urllib.parse.urlparse(self.redirect_uri)
        port = parsed.port or 8080
        server_task = asyncio.create_task(self._run_callback_server(port))
        await server_task

        if self._auth_result is None:
            raise Exception("Authorization timed out")

        if "error" in self._auth_result:
            raise Exception(f"Authorization failed: {self._auth_result['error']}")

        # Step 5: Exchange code for token
        print("🔄 Exchanging code for token...")
        code = self._auth_result["code"]
        tokens = await self.exchange_code_for_token(code)

        print("✅ Authorization complete!\n")
        return tokens
