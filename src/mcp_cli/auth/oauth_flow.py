"""OAuth 2.0 authorization flow implementation."""

import asyncio
import base64
import hashlib
import secrets
import urllib.parse
import webbrowser
from typing import Dict, Optional
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import httpx

from .oauth_config import OAuthConfig, OAuthTokens


class OAuthFlow:
    """Handles OAuth 2.0 authorization flow."""

    def __init__(self, config: OAuthConfig):
        """
        Initialize OAuth flow.

        Args:
            config: OAuth configuration
        """
        self.config = config
        self._auth_result: Optional[Dict[str, str]] = None
        self._code_verifier: Optional[str] = None

    def _generate_pkce_pair(self) -> tuple[str, str]:
        """Generate PKCE code verifier and challenge."""
        # Generate random code verifier
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode(
            "utf-8"
        )
        code_verifier = code_verifier.rstrip("=")

        # Generate code challenge from verifier
        challenge_bytes = hashlib.sha256(code_verifier.encode("utf-8")).digest()
        code_challenge = base64.urlsafe_b64encode(challenge_bytes).decode("utf-8")
        code_challenge = code_challenge.rstrip("=")

        return code_verifier, code_challenge

    def get_authorization_url(self) -> str:
        """
        Generate authorization URL for user consent.

        Returns:
            Authorization URL to open in browser
        """
        params = {
            "client_id": self.config.client_id,
            "response_type": "code",
            "redirect_uri": self.config.redirect_uri,
        }

        if self.config.scopes:
            params["scope"] = " ".join(self.config.scopes)

        # Add PKCE if enabled
        if self.config.use_pkce:
            self._code_verifier, code_challenge = self._generate_pkce_pair()
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"

        # Add any extra parameters
        params.update(self.config.extra_auth_params)

        # Build URL
        query_string = urllib.parse.urlencode(params)
        return f"{self.config.authorization_url}?{query_string}"

    def _create_callback_handler(self):
        """Create callback handler class."""
        auth_flow = self

        class CallbackHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress logging

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                params = dict(urllib.parse.parse_qsl(parsed.query))

                # Check for error
                if "error" in params:
                    error_description = params.get("error_description", params["error"])
                    auth_flow._auth_result = {"error": error_description}
                    response = f"<html><body><h1>Authorization Failed</h1><p>{error_description}</p></body></html>"
                    self.send_response(400)
                # Get authorization code
                elif "code" not in params:
                    auth_flow._auth_result = {"error": "No authorization code received"}
                    response = "<html><body><h1>Authorization Failed</h1><p>No code received</p></body></html>"
                    self.send_response(400)
                else:
                    auth_flow._auth_result = params
                    response = "<html><body><h1>Authorization Successful</h1><p>You can close this window and return to the terminal.</p></body></html>"
                    self.send_response(200)

                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(response.encode())

        return CallbackHandler

    async def _run_callback_server(self, port: int) -> None:
        """Run local callback server."""
        handler_class = self._create_callback_handler()
        server = HTTPServer(("localhost", port), handler_class)

        # Run server in thread
        server_thread = Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        # Wait for callback or timeout
        timeout = 300  # 5 minutes
        for _ in range(timeout):
            if self._auth_result is not None:
                break
            await asyncio.sleep(1)

        server.shutdown()

    async def exchange_code_for_token(self, code: str) -> OAuthTokens:
        """
        Exchange authorization code for access token.

        Args:
            code: Authorization code from callback

        Returns:
            OAuth tokens

        Raises:
            Exception: If token exchange fails
        """
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.config.redirect_uri,
            "client_id": self.config.client_id,
        }

        if self.config.client_secret:
            data["client_secret"] = self.config.client_secret

        # Add PKCE verifier if used
        if self.config.use_pkce and self._code_verifier:
            data["code_verifier"] = self._code_verifier

        async with httpx.AsyncClient() as client:
            response = await client.post(self.config.token_url, data=data)
            if response.status_code != 200:
                raise Exception(f"Token exchange failed: {response.text}")

            token_data = response.json()
            return OAuthTokens.from_dict(token_data)

    async def refresh_token(self, refresh_token: str) -> OAuthTokens:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token

        Returns:
            New OAuth tokens

        Raises:
            Exception: If token refresh fails
        """
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.config.client_id,
        }

        if self.config.client_secret:
            data["client_secret"] = self.config.client_secret

        async with httpx.AsyncClient() as client:
            response = await client.post(self.config.token_url, data=data)
            if response.status_code != 200:
                raise Exception(f"Token refresh failed: {response.text}")

            token_data = response.json()
            return OAuthTokens.from_dict(token_data)

    async def authorize(self) -> OAuthTokens:
        """
        Perform full authorization flow.

        Returns:
            OAuth tokens

        Raises:
            Exception: If authorization fails
        """
        # Extract port from redirect URI
        parsed = urllib.parse.urlparse(self.config.redirect_uri)
        port = parsed.port or 8080

        # Get authorization URL
        auth_url = self.get_authorization_url()

        print("\nOpening browser for authorization...")
        print(f"If browser doesn't open, visit: {auth_url}\n")

        # Open browser
        webbrowser.open(auth_url)

        # Start callback server
        server_task = asyncio.create_task(self._run_callback_server(port))

        try:
            await server_task
        except Exception as e:
            raise Exception(f"Callback server error: {e}")

        # Check result
        if self._auth_result is None:
            raise Exception("Authorization timed out")

        if "error" in self._auth_result:
            raise Exception(f"Authorization failed: {self._auth_result['error']}")

        # Exchange code for token
        code = self._auth_result["code"]
        tokens = await self.exchange_code_for_token(code)

        return tokens
