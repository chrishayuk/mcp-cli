# mcp_cli/tools/manager.py
"""
Centralized tool management using CHUK Tool Processor.

This module provides a unified interface for all tool-related operations in MCP CLI,
leveraging the async-native capabilities of CHUK Tool Processor.

Supports STDIO, HTTP, and SSE servers with automatic detection.
ENHANCED: Now includes schema validation and tool filtering capabilities.
CLEAN: Tool name resolution using registry lookup without sanitization.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncIterator
from pathlib import Path

from chuk_tool_processor.core.processor import ToolProcessor
from chuk_tool_processor.registry import ToolRegistryProvider
from chuk_tool_processor.mcp.stream_manager import StreamManager
from chuk_tool_processor.models.tool_result import ToolResult
from chuk_tool_processor.models.tool_call import ToolCall
from chuk_tool_processor.execution.strategies.inprocess_strategy import (
    InProcessStrategy,
)
from chuk_tool_processor.execution.tool_executor import ToolExecutor

from mcp_cli.auth import OAuthHandler
from mcp_cli.constants import NAMESPACE
from mcp_cli.tools.models import ServerInfo, ToolCallResult, ToolInfo
from mcp_cli.tools.validation import ToolSchemaValidator
from mcp_cli.tools.filter import ToolFilter

logger = logging.getLogger(__name__)


class ToolManager:
    """
    Central interface for all tool operations in MCP CLI.

    This class wraps CHUK Tool Processor and provides a clean API for:
    - Tool discovery and listing
    - Tool execution with streaming support
    - Server management (STDIO, HTTP, and SSE)
    - LLM-compatible tool conversion
    - Schema validation and tool filtering (ENHANCED)
    """

    def __init__(
        self,
        config_file: str,
        servers: List[str],
        server_names: Optional[Dict[int, str]] = None,
        tool_timeout: Optional[float] = None,
        max_concurrency: int = 4,
        initialization_timeout: float = 120.0,
    ):
        self.config_file = config_file
        self.servers = servers
        self.server_names = server_names or {}
        self.tool_timeout = self._determine_timeout(tool_timeout)
        self.max_concurrency = max_concurrency
        self.initialization_timeout = initialization_timeout

        # CHUK components
        self.processor: Optional[ToolProcessor] = None
        self.stream_manager: Optional[StreamManager] = None
        self._registry = None
        self._executor: Optional[ToolExecutor] = None

        # Server type detection
        self._http_servers: List[Any] = []
        self._stdio_servers: List[Any] = []
        self._sse_servers: List[Any] = []
        self._config_cache: Optional[Dict[str, Any]] = None

        # Effective timeout (will be set after server detection)
        self._effective_timeout: Optional[float] = None
        self._effective_max_retries: Optional[int] = None

        # OAuth support with mcp-cli namespace
        from mcp_cli.auth import TokenManager, TokenStoreBackend

        token_manager = TokenManager(
            backend=TokenStoreBackend.AUTO,
            namespace=NAMESPACE,
            service_name="mcp-cli",
        )
        self.oauth_handler = OAuthHandler(token_manager=token_manager)

        # ENHANCED: Tool validation and filtering
        self.tool_filter = ToolFilter()
        self.validation_results: Dict[str, Any] = {}
        self.last_validation_provider: Optional[str] = None

    def _determine_timeout(self, explicit_timeout: Optional[float]) -> float:
        """Determine timeout with environment variable fallback."""
        if explicit_timeout is not None:
            return explicit_timeout

        # Check environment variables in order of preference
        for env_var in [
            "MCP_TOOL_TIMEOUT",
            "CHUK_TOOL_TIMEOUT",
            "MCP_CLI_INIT_TIMEOUT",
        ]:
            env_timeout = os.getenv(env_var)
            if env_timeout:
                try:
                    return float(env_timeout)
                except ValueError:
                    logger.warning(f"Invalid timeout in {env_var}: {env_timeout}")

        return 120.0  # Default 2 minutes

    def _load_config(self) -> Dict[str, Any]:
        """Load and cache the configuration file with fallback to bundled package config."""
        if self._config_cache is not None:
            return self._config_cache

        try:
            config_path = Path(self.config_file)

            # Try explicit path or current directory first
            if config_path.exists():
                with open(config_path, "r") as f:
                    self._config_cache = json.load(f)
                    # Resolve token placeholders first (before any other processing)
                    self._resolve_token_placeholders(self._config_cache)
                    # Inject logging environment variables into STDIO servers
                    self._inject_logging_env_vars(self._config_cache)
                    return self._config_cache

            # If not found and using default name, try package bundle
            if self.config_file == "server_config.json":
                try:
                    import importlib.resources as resources

                    # Try Python 3.9+ API
                    if hasattr(resources, "files"):
                        package_files = resources.files("mcp_cli")
                        bundled_config = package_files / "server_config.json"
                        if bundled_config.is_file():
                            data_str = bundled_config.read_text()
                            self._config_cache = json.loads(data_str)
                            # Resolve token placeholders first (before any other processing)
                            self._resolve_token_placeholders(self._config_cache)
                            # Inject logging environment variables into STDIO servers
                            self._inject_logging_env_vars(self._config_cache)
                            logger.info("Loaded bundled server configuration")
                            return self._config_cache
                except (ImportError, FileNotFoundError, AttributeError, TypeError) as e:
                    logger.debug(f"Could not load bundled config: {e}")

        except Exception as e:
            logger.warning(f"Could not load config file: {e}")

        self._config_cache = {}
        return self._config_cache

    def _resolve_token_placeholders(self, config: Dict[str, Any]) -> None:
        """
        Resolve ${TOKEN:namespace:name} placeholders in environment variables.

        Replaces placeholders like ${TOKEN:bearer:brave_search} with actual token
        values from the secure token store.
        """
        import re
        from mcp_cli.auth import TokenManager, TokenStoreBackend

        mcp_servers = config.get("mcpServers", {})
        token_pattern = re.compile(r"\$\{TOKEN:([^:]+):([^}]+)\}")

        # Get token manager for looking up tokens
        token_manager = TokenManager(
            backend=TokenStoreBackend.AUTO,
            namespace=NAMESPACE,
            service_name="mcp-cli",
        )

        for server_name, server_config in mcp_servers.items():
            # Process both env vars and headers
            for field_name in ["env", "headers"]:
                if field_name in server_config:
                    for key, value in server_config[field_name].items():
                        if isinstance(value, str):
                            # Find all token placeholders in this value
                            matches = token_pattern.findall(value)
                            for namespace, name in matches:
                                # Retrieve token from store
                                try:
                                    store = token_manager.token_store
                                    # Use retrieve_generic method to get token value
                                    token_value = store.retrieve_generic(
                                        key=name, namespace=namespace
                                    )

                                    if token_value:
                                        # Replace placeholder with actual token
                                        placeholder = f"${{TOKEN:{namespace}:{name}}}"
                                        server_config[field_name][key] = value.replace(
                                            placeholder, token_value
                                        )
                                        # Log token with partial masking for security
                                        masked_token = (
                                            token_value[:8] + "..." + token_value[-4:]
                                            if len(token_value) > 12
                                            else "***"
                                        )
                                        logger.debug(
                                            f"Resolved token placeholder for {server_name} ({field_name}.{key}): "
                                            f"{namespace}:{name} -> {masked_token} (length: {len(token_value)})"
                                        )
                                    else:
                                        logger.warning(
                                            f"Token not found for {server_name} ({field_name}.{key}): "
                                            f"{namespace}:{name}"
                                        )
                                except Exception as e:
                                    logger.error(
                                        f"Error resolving token for {server_name} ({field_name}.{key}) "
                                        f"({namespace}:{name}): {e}"
                                    )

    def _get_max_server_timeout(self, servers: List[Any]) -> float:
        """
        Get the maximum timeout from server configurations.
        Falls back to self.tool_timeout if no server-specific timeouts are set.
        """
        max_timeout = self.tool_timeout
        for server in servers:
            if isinstance(server, dict) and "timeout" in server:
                server_timeout = float(server["timeout"])
                max_timeout = max(max_timeout, server_timeout)
        return max_timeout

    def _get_min_server_max_retries(self, servers: List[Any]) -> int:
        """
        Get the minimum max_retries from server configurations.
        Falls back to 2 if no server-specific max_retries are set.
        If any server has max_retries=0, returns 0 (no retries).
        """
        min_retries = 2  # Default from line 817
        has_server_config = False

        for server in servers:
            if isinstance(server, dict) and "max_retries" in server:
                has_server_config = True
                server_retries = int(server["max_retries"])
                if server_retries == 0:
                    return 0  # If any server wants no retries, disable retries
                min_retries = min(min_retries, server_retries)

        return min_retries if has_server_config else 2

    def _inject_logging_env_vars(self, config: Dict[str, Any]) -> None:
        """
        Inject logging environment variables into STDIO server configs.

        Note: Subprocess stderr from MCP servers may still appear during startup
        as PYTHONSTARTUP doesn't work for non-interactive scripts. To fully suppress,
        use: mcp-cli --server name 2>/dev/null
        """
        mcp_servers = config.get("mcpServers", {})

        # Environment variables to inject for quiet logging (generic)
        # These work if the subprocess code checks them
        log_env_vars = {
            "LOG_LEVEL": "ERROR",
            "LOGGING_LEVEL": "ERROR",
            "PYTHONWARNINGS": "ignore",
            "PYTHONIOENCODING": "utf-8",
        }

        for server_name, server_config in mcp_servers.items():
            # Only inject for STDIO servers (those with "command" field)
            if "command" in server_config:
                # Ensure env dict exists
                if "env" not in server_config:
                    server_config["env"] = {}

                # Inject logging vars if not already present
                for key, value in log_env_vars.items():
                    if key not in server_config["env"]:
                        server_config["env"][key] = value

    def _detect_server_types(self):
        """Detect server transport types from configuration."""
        config = self._load_config()
        mcp_servers = config.get("mcpServers", {})

        if not mcp_servers:
            # No config, assume all are STDIO
            self._stdio_servers = self.servers.copy()
            logger.debug("No config found, assuming all servers are STDIO")
            return

        for server in self.servers:
            server_config = mcp_servers.get(server, {})
            transport_type = server_config.get("transport", "").lower()

            if transport_type == "sse":
                server_entry = {
                    "name": server,
                    "url": server_config["url"],
                    "headers": server_config.get("headers", {}),
                }
                # Mark if OAuth is configured (will be processed during initialization)
                if "oauth" in server_config:
                    server_entry["oauth"] = server_config["oauth"]
                # Add per-server timeout and max_retries if configured
                if "timeout" in server_config:
                    server_entry["timeout"] = server_config["timeout"]
                if "max_retries" in server_config:
                    server_entry["max_retries"] = server_config["max_retries"]
                self._sse_servers.append(server_entry)
                logger.debug(f"Detected SSE server: {server}")
            elif "url" in server_config:
                server_entry = {
                    "name": server,
                    "url": server_config["url"],
                    "headers": server_config.get("headers", {}),
                }
                # Mark if OAuth is configured (will be processed during initialization)
                if "oauth" in server_config:
                    server_entry["oauth"] = server_config["oauth"]
                # Add per-server timeout and max_retries if configured
                if "timeout" in server_config:
                    server_entry["timeout"] = server_config["timeout"]
                    logger.info(
                        f"Server '{server}' configured with timeout: {server_config['timeout']}s"
                    )
                if "max_retries" in server_config:
                    server_entry["max_retries"] = server_config["max_retries"]
                    logger.info(
                        f"Server '{server}' configured with max_retries: {server_config['max_retries']}"
                    )
                self._http_servers.append(server_entry)
                logger.debug(f"Detected HTTP server: {server}")
            else:
                self._stdio_servers.append(server)
                logger.debug(f"Detected STDIO server: {server}")

        logger.info(
            f"Detected {len(self._http_servers)} HTTP, {len(self._sse_servers)} SSE, {len(self._stdio_servers)} STDIO servers"
        )

    async def _process_oauth_for_servers(self, servers: List[Dict[str, Any]]) -> None:
        """Process OAuth authentication for servers that require it."""
        from mcp_cli.config.config_manager import initialize_config

        # Initialize config manager if not already initialized
        config = initialize_config(Path(self.config_file))

        for server_entry in servers:
            server_name = server_entry["name"]
            server_config = config.get_server(server_name)

            if not server_config:
                continue

            # Remote MCP servers (with URL) automatically use MCP OAuth
            # Servers with explicit oauth config use that config
            # Skip servers that already have Authorization headers (they don't need OAuth)
            is_remote_mcp = server_config.url and not server_config.command
            has_explicit_oauth = server_config.oauth is not None
            has_auth_header = (
                server_config.headers and "Authorization" in server_config.headers
            )

            # Skip servers that:
            # 1. Don't need OAuth (not remote MCP and no explicit OAuth config)
            # 2. Already have Authorization headers (pre-configured auth)
            if (not is_remote_mcp and not has_explicit_oauth) or has_auth_header:
                # Skip servers that don't need OAuth
                continue

            try:
                # Perform OAuth and get authorization header
                headers = await self.oauth_handler.prepare_server_headers(server_config)

                # IMPORTANT: HTTP Streamable transport prioritizes configured_headers over api_key
                # To avoid conflicts, we should set EITHER headers OR api_key, not both
                # For OAuth, we use api_key field (which gets converted to Authorization header)
                if "Authorization" in headers:
                    # Extract just the token value (remove "Bearer " prefix if present)
                    # The http_streamable_transport will add "Bearer " prefix automatically
                    auth_value = headers["Authorization"]
                    if auth_value.startswith("Bearer "):
                        auth_value = auth_value[7:]  # Remove "Bearer " prefix
                    server_entry["api_key"] = auth_value
                    logger.debug(f"Set api_key for {server_name}: {auth_value[:20]}...")

                # Merge any other headers (non-Authorization)
                other_headers = {
                    k: v for k, v in headers.items() if k != "Authorization"
                }
                if other_headers:
                    if "headers" not in server_entry:
                        server_entry["headers"] = {}
                    server_entry["headers"].update(other_headers)

                logger.info(f"OAuth authentication completed for {server_name}")
            except Exception as e:
                error_msg = str(e).lower()
                # Check if this is an invalid token error (401)
                if (
                    "401" in error_msg
                    or "invalid_token" in error_msg
                    or "invalid access token" in error_msg
                    or "unauthorized" in error_msg
                ):
                    logger.warning(
                        f"Invalid or expired token detected for {server_name}: {e}"
                    )
                    logger.info(
                        f"Clearing stored tokens and re-authenticating {server_name}..."
                    )

                    # Clear the invalid tokens (this is now handled in oauth_handler.ensure_authenticated_mcp)
                    # but we do it again here to ensure memory cache is cleared
                    self.oauth_handler.clear_tokens(server_name)

                    # NOTE: We don't delete client registration anymore - it's still valid
                    # Only tokens need to be refreshed. The oauth_handler will handle the full flow.

                    # Retry authentication (this will trigger the full OAuth flow with browser)
                    try:
                        logger.info(f"Retrying authentication for {server_name}...")
                        headers = await self.oauth_handler.prepare_server_headers(
                            server_config
                        )

                        if "Authorization" in headers:
                            auth_value = headers["Authorization"]
                            if auth_value.startswith("Bearer "):
                                auth_value = auth_value[7:]  # Remove "Bearer " prefix
                            server_entry["api_key"] = auth_value
                            logger.debug(
                                f"Set api_key for {server_name} after re-auth: {auth_value[:20]}..."
                            )

                        other_headers = {
                            k: v for k, v in headers.items() if k != "Authorization"
                        }
                        if other_headers:
                            if "headers" not in server_entry:
                                server_entry["headers"] = {}
                            server_entry["headers"].update(other_headers)

                        logger.info(
                            f"✅ Re-authentication successful for {server_name}"
                        )
                    except Exception as retry_e:
                        logger.error(
                            f"Re-authentication failed for {server_name}: {retry_e}"
                        )
                        raise
                else:
                    # Not an auth error, re-raise
                    logger.error(f"OAuth failed for {server_name}: {e}")
                    raise

    def _create_oauth_refresh_callback(self):
        """Create OAuth refresh callback for HTTP/SSE transports (NEW)."""

        async def refresh_oauth_token():
            """
            Callback to refresh OAuth tokens when they expire.

            Returns:
                Dict with updated Authorization header or None if refresh failed
            """
            logger.warning("=" * 80)
            logger.warning("OAUTH REFRESH CALLBACK TRIGGERED!")
            logger.warning("=" * 80)
            try:
                # Try HTTP servers first, then SSE servers
                server_entry = None
                if self._http_servers:
                    server_entry = self._http_servers[0]
                elif self._sse_servers:
                    server_entry = self._sse_servers[0]
                else:
                    logger.warning("No HTTP/SSE servers configured for OAuth refresh")
                    return None

                server_name = server_entry["name"]
                logger.info(f"Refreshing OAuth token for {server_name}")

                from mcp_cli.config.config_manager import initialize_config

                config = initialize_config(Path(self.config_file))
                server_config = config.get_server(server_name)

                if not server_config:
                    logger.error(f"Server config not found for {server_name}")
                    return None

                # Clear old tokens to force full reauthentication
                # This is necessary when the server restarts and loses sessions
                logger.info(
                    f"Clearing stored tokens for {server_name} to force reauthentication"
                )
                self.oauth_handler.clear_tokens(server_name)

                # Delete client registration as well to force full OAuth flow
                try:
                    token_dir = Path.home() / ".mcp_cli" / "tokens"
                    client_file = token_dir / f"{server_name}_client.json"
                    if client_file.exists():
                        os.remove(client_file)
                        logger.debug(f"Deleted client registration for {server_name}")
                except Exception as e:
                    logger.debug(f"Could not delete client registration: {e}")

                # Get new headers (this will trigger full OAuth flow with browser)
                headers = await self.oauth_handler.prepare_server_headers(server_config)

                logger.info(f"OAuth reauthentication completed for {server_name}")

                # Update the api_key in the server entry for the transport
                if "Authorization" in headers:
                    auth_value = headers["Authorization"]
                    if auth_value.startswith("Bearer "):
                        auth_value = auth_value[7:]  # Remove "Bearer " prefix
                    server_entry["api_key"] = auth_value
                    logger.debug(
                        f"Updated server_entry api_key for {server_name} after reauth"
                    )

                    # CRITICAL: Also update the transport's api_key directly
                    # The transport's _attempt_recovery() will reinitialize using self.api_key
                    # So we need to update it before the transport tries to reconnect
                    if self.stream_manager and hasattr(
                        self.stream_manager, "transports"
                    ):
                        transport = self.stream_manager.transports.get(server_name)
                        if transport and hasattr(transport, "api_key"):
                            transport.api_key = auth_value
                            logger.debug(
                                f"Updated transport.api_key for {server_name} after reauth"
                            )

                return headers

            except Exception as e:
                logger.error(f"OAuth token refresh failed: {e}")
                return None

        return refresh_oauth_token

    async def initialize(self, namespace: str = "stdio") -> bool:
        """Connect to MCP servers and initialize the tool registry."""
        try:
            from chuk_term.ui import output

            logger.info(f"Initializing ToolManager with {len(self.servers)} servers")

            self._detect_server_types()

            # Calculate effective timeout and max_retries from all servers
            all_servers = self._http_servers + self._sse_servers
            if all_servers:
                self._effective_timeout = self._get_max_server_timeout(all_servers)
                self._effective_max_retries = self._get_min_server_max_retries(
                    all_servers
                )
                logger.info(
                    f"Effective timeout: {self._effective_timeout}s, max_retries: {self._effective_max_retries}"
                )
            else:
                self._effective_timeout = self.tool_timeout
                self._effective_max_retries = 2

            # Determine which servers we're connecting to for better messaging
            server_names = []
            if self._http_servers:
                server_names.extend([s["name"] for s in self._http_servers])
            if self._sse_servers:
                server_names.extend([s["name"] for s in self._sse_servers])
            if self._stdio_servers:
                server_names.extend(self._stdio_servers)

            server_list = ", ".join(server_names) if server_names else "servers"

            # Show spinner during entire initialization (this will auto-clear when done)
            with output.loading(f"🔐 Connecting to {server_list}...", spinner="dots"):
                # Process OAuth for HTTP/SSE servers before connecting
                if self._http_servers:
                    await self._process_oauth_for_servers(self._http_servers)
                if self._sse_servers:
                    await self._process_oauth_for_servers(self._sse_servers)

                # Try transports in priority order: SSE > HTTP > STDIO
                success = False

                if self._sse_servers:
                    logger.info("Setting up SSE servers")
                    success = await self._setup_sse_servers(
                        self._sse_servers[0]["name"]
                    )
                elif self._http_servers:
                    logger.info("Setting up HTTP servers")
                    success = await self._setup_http_servers(
                        self._http_servers[0]["name"]
                    )
                elif self._stdio_servers:
                    logger.info("Setting up STDIO servers")
                    success = await self._setup_stdio_servers(namespace)
                else:
                    logger.info(
                        "No servers configured - initializing with empty tool list"
                    )
                    success = await self._setup_empty_toolset()

            if not success:
                logger.error("Server setup failed")
                return False

            await self._setup_common_components()
            logger.info("ToolManager initialized successfully")
            return True

        except asyncio.TimeoutError:
            logger.error(
                f"Initialization timed out after {self.initialization_timeout}s"
            )
            return False
        except Exception as exc:
            logger.error(f"Error initializing tool manager: {exc}")
            return False

    async def _setup_sse_servers(self, namespace: str) -> bool:
        """Setup SSE servers."""
        try:
            from chuk_tool_processor.mcp.setup_mcp_sse import setup_mcp_sse

            # Create OAuth refresh callback (NEW)
            oauth_refresh_callback = self._create_oauth_refresh_callback()

            # Extract per-server timeout and max_retries configuration
            server_timeout = self._get_max_server_timeout(self._sse_servers)
            server_max_retries = self._get_min_server_max_retries(self._sse_servers)

            try:
                self.processor, self.stream_manager = await asyncio.wait_for(
                    setup_mcp_sse(
                        servers=self._sse_servers,
                        server_names=self.server_names,
                        namespace=namespace,
                        default_timeout=server_timeout,
                        max_retries=server_max_retries,
                        enable_retries=(
                            server_max_retries > 0
                        ),  # Disable if max_retries is 0
                        oauth_refresh_callback=oauth_refresh_callback,  # NEW
                    ),
                    timeout=self.initialization_timeout,
                )
            except Exception as setup_error:
                error_msg = str(setup_error).lower()
                # Also check the full exception chain for auth errors
                full_error_context = ""
                if hasattr(setup_error, "__cause__") and setup_error.__cause__:
                    full_error_context += str(setup_error.__cause__).lower()
                if hasattr(setup_error, "args"):
                    full_error_context += " ".join(
                        str(arg) for arg in setup_error.args
                    ).lower()

                combined_error = error_msg + " " + full_error_context

                # Check if this is a 401/auth error during setup
                if (
                    "401" in combined_error
                    or "invalid_token" in combined_error
                    or "invalid access token" in combined_error
                    or "unauthorized" in combined_error
                ):
                    logger.warning(
                        f"SSE server setup failed with authentication error: {setup_error}"
                    )
                    logger.info("Attempting to re-authenticate and retry...")

                    # Try OAuth refresh callback to get new tokens
                    try:
                        new_headers = await oauth_refresh_callback()
                    except Exception as callback_error:
                        logger.error(f"OAuth refresh callback failed: {callback_error}")
                        new_headers = None

                    if new_headers:
                        # Retry setup with new authentication
                        logger.info(
                            "Retrying SSE server setup with refreshed authentication..."
                        )
                        self.processor, self.stream_manager = await asyncio.wait_for(
                            setup_mcp_sse(
                                servers=self._sse_servers,
                                server_names=self.server_names,
                                namespace=namespace,
                                default_timeout=self.tool_timeout,
                                oauth_refresh_callback=oauth_refresh_callback,
                            ),
                            timeout=self.initialization_timeout,
                        )
                        logger.info(
                            "✅ SSE server setup succeeded after re-authentication"
                        )
                    else:
                        logger.error(
                            "Re-authentication failed, cannot setup SSE servers. "
                            "Try running 'mcp-cli token delete <server> --is-oauth' to clear tokens."
                        )
                        raise
                else:
                    # Not an auth error, re-raise
                    raise

            return True

        except ImportError as e:
            logger.error(f"SSE transport not available: {e}")
            return False
        except Exception as e:
            logger.error(f"SSE server setup failed: {e}")
            return False

    async def _setup_http_servers(self, namespace: str) -> bool:
        """Setup HTTP servers."""
        try:
            from chuk_tool_processor.mcp.setup_mcp_http_streamable import (
                setup_mcp_http_streamable,
            )

            # Create OAuth refresh callback (NEW)
            oauth_refresh_callback = self._create_oauth_refresh_callback()

            # Extract per-server timeout and max_retries configuration
            server_timeout = self._get_max_server_timeout(self._http_servers)
            server_max_retries = self._get_min_server_max_retries(self._http_servers)

            try:
                self.processor, self.stream_manager = await asyncio.wait_for(
                    setup_mcp_http_streamable(
                        servers=self._http_servers,
                        server_names=self.server_names,
                        namespace=namespace,
                        default_timeout=server_timeout,
                        max_retries=server_max_retries,
                        enable_retries=(
                            server_max_retries > 0
                        ),  # Disable if max_retries is 0
                        oauth_refresh_callback=oauth_refresh_callback,  # NEW
                    ),
                    timeout=self.initialization_timeout,
                )
            except Exception as setup_error:
                error_msg = str(setup_error).lower()
                # Also check the full exception chain for auth errors
                full_error_context = ""
                if hasattr(setup_error, "__cause__") and setup_error.__cause__:
                    full_error_context += str(setup_error.__cause__).lower()
                if hasattr(setup_error, "args"):
                    full_error_context += " ".join(
                        str(arg) for arg in setup_error.args
                    ).lower()

                combined_error = error_msg + " " + full_error_context

                # Check if this is a 401/auth error during setup
                if (
                    "401" in combined_error
                    or "invalid_token" in combined_error
                    or "invalid access token" in combined_error
                    or "unauthorized" in combined_error
                ):
                    logger.warning(
                        f"HTTP server setup failed with authentication error: {setup_error}"
                    )
                    logger.info("Attempting to re-authenticate and retry...")

                    # Try OAuth refresh callback to get new tokens
                    try:
                        new_headers = await oauth_refresh_callback()
                    except Exception as callback_error:
                        logger.error(f"OAuth refresh callback failed: {callback_error}")
                        new_headers = None

                    if new_headers:
                        # Retry setup with new authentication
                        logger.info(
                            "Retrying HTTP server setup with refreshed authentication..."
                        )
                        self.processor, self.stream_manager = await asyncio.wait_for(
                            setup_mcp_http_streamable(
                                servers=self._http_servers,
                                server_names=self.server_names,
                                namespace=namespace,
                                default_timeout=self.tool_timeout,
                                oauth_refresh_callback=oauth_refresh_callback,
                            ),
                            timeout=self.initialization_timeout,
                        )
                        logger.info(
                            "✅ HTTP server setup succeeded after re-authentication"
                        )
                    else:
                        logger.error(
                            "Re-authentication failed, cannot setup HTTP servers. "
                            "Try running 'mcp-cli token delete <server> --is-oauth' to clear tokens."
                        )
                        raise
                else:
                    # Not an auth error, re-raise
                    raise

            # DEBUG: Check if processor has tools and attempt manual registration
            if self.processor:
                logger.info(f"HTTP setup returned processor: {type(self.processor)}")

                # Try to manually trigger tool registration for HTTP servers
                # This is a workaround for HTTP MCP servers not auto-registering tools
                if (
                    hasattr(self.stream_manager, "_clients")
                    and self.stream_manager._clients
                ):
                    logger.info(
                        "Attempting manual tool registration for HTTP MCP servers..."
                    )
                    for client_id, client in enumerate(self.stream_manager._clients):
                        try:
                            if hasattr(client, "tools") and client.tools:
                                logger.info(
                                    f"Client {client_id} has {len(client.tools)} tools available"
                                )
                                # Tools are in the client but need to be registered
                                # This should be handled by chuk-tool-processor but appears to be missing
                        except Exception as e:
                            logger.debug(
                                f"Could not check tools for client {client_id}: {e}"
                            )

            if self.stream_manager:
                logger.info(
                    f"HTTP setup returned stream_manager: {type(self.stream_manager)}"
                )

            return True

        except ImportError as e:
            logger.error(f"HTTP transport not available: {e}")
            return False
        except Exception as e:
            logger.error(f"HTTP server setup failed: {e}")
            return False

    async def _setup_stdio_servers(self, namespace: str) -> bool:
        """Setup STDIO servers."""
        try:
            from chuk_tool_processor.mcp.setup_mcp_stdio import setup_mcp_stdio
            import tempfile

            logger.info(
                f"Setting up STDIO servers with {self.initialization_timeout}s timeout"
            )

            # Write the modified config (with resolved tokens) to a temp file
            # This ensures chuk-tool-processor gets the config with tokens already replaced
            config_to_use = self.config_file
            temp_config_file = None

            if self._config_cache:
                # Create a temporary file with the modified config
                temp_config_file = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                )
                json.dump(self._config_cache, temp_config_file)
                temp_config_file.close()
                config_to_use = temp_config_file.name
                logger.debug(
                    f"Using temporary config file with resolved tokens: {config_to_use}"
                )

            # Try to pass initialization_timeout to setup_mcp_stdio
            # This controls the timeout for the initial connection/handshake
            try:
                self.processor, self.stream_manager = await asyncio.wait_for(
                    setup_mcp_stdio(
                        config_file=config_to_use,
                        servers=self._stdio_servers,
                        server_names=self.server_names,
                        namespace=namespace,
                        default_timeout=self.tool_timeout,
                        initialization_timeout=self.initialization_timeout,  # Pass init timeout
                        enable_caching=True,
                        enable_retries=True,
                        max_retries=2,
                    ),
                    timeout=self.initialization_timeout
                    + 10.0,  # Add buffer for outer timeout
                )
            except TypeError:
                # Fallback if initialization_timeout parameter doesn't exist
                logger.warning(
                    "initialization_timeout not supported by setup_mcp_stdio, using legacy call"
                )
                self.processor, self.stream_manager = await asyncio.wait_for(
                    setup_mcp_stdio(
                        config_file=config_to_use,
                        servers=self._stdio_servers,
                        server_names=self.server_names,
                        namespace=namespace,
                        default_timeout=self.tool_timeout,
                        enable_caching=True,
                        enable_retries=True,
                        max_retries=2,
                    ),
                    timeout=self.initialization_timeout,
                )
            finally:
                # Clean up temp file
                if temp_config_file:
                    try:
                        import os

                        os.unlink(temp_config_file.name)
                    except Exception:
                        pass

            logger.info("STDIO servers initialized successfully")
            return True

        except asyncio.TimeoutError:
            logger.error(
                f"STDIO server initialization timed out after {self.initialization_timeout}s"
            )
            logger.error(
                "This may indicate the server is not responding or is misconfigured"
            )
            return False
        except Exception as e:
            logger.error(f"STDIO server setup failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def _setup_empty_toolset(self) -> bool:
        """Setup an empty tool processor when no servers are configured."""
        try:
            # Create a minimal mock processor that implements the required interface
            class EmptyToolProcessor:
                def __init__(self):
                    self.tools = {}

                async def execute_tool(self, *args, **kwargs):
                    return {"error": "No tools available"}

                def list_tools(self):
                    return []

                def get_tool(self, name):
                    return None

            class EmptyStreamManager:
                def __init__(self):
                    pass

                async def stream(self, *args, **kwargs):
                    yield {"error": "No streaming available"}

                async def close(self):
                    """No-op close method for compatibility."""
                    pass

            # Create minimal processor and stream manager with no tools
            self.processor = EmptyToolProcessor()
            self.stream_manager = EmptyStreamManager()

            logger.info(
                "Initialized with empty tool set - chat mode available without tools"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to setup empty toolset: {e}")
            return False

    async def _setup_common_components(self):
        """Setup components common to all transport types."""
        self._registry = await asyncio.wait_for(
            ToolRegistryProvider.get_registry(), timeout=30.0
        )

        # DEBUG: Check how many tools are in the registry
        try:
            registry_items = await asyncio.wait_for(
                self._registry.list_tools(), timeout=10.0
            )
            logger.info(
                f"Registry has {len(registry_items)} tools registered after initialization"
            )
            if registry_items:
                logger.info(f"Sample tools: {registry_items[:5]}")
            else:
                logger.warning("No tools found in registry after initialization!")
        except Exception as e:
            logger.error(f"Failed to list tools from registry: {e}")

        # Use effective timeout calculated from server configs
        effective_timeout = self._effective_timeout or self.tool_timeout

        strategy = InProcessStrategy(
            self._registry,
            max_concurrency=self.max_concurrency,
            default_timeout=effective_timeout,
        )

        self._executor = ToolExecutor(
            registry=self._registry,
            strategy=strategy,
            default_timeout=effective_timeout,
        )

    async def close(self):
        """Close all resources and connections."""
        errors = []

        # Close stream manager first (handles MCP connections and subprocesses)
        if self.stream_manager:
            try:
                logger.debug("Closing stream manager...")
                # Set a reasonable timeout for cleanup
                await asyncio.wait_for(self.stream_manager.close(), timeout=10.0)
                logger.debug("Stream manager closed successfully")
            except asyncio.TimeoutError:
                logger.warning("Stream manager close timed out after 10s")
                errors.append("Stream manager close timed out")
            except Exception as exc:
                logger.warning(f"Error closing stream manager: {exc}")
                errors.append(f"Stream manager: {exc}")

        # Close executor
        if self._executor:
            try:
                logger.debug("Shutting down executor...")
                await asyncio.wait_for(self._executor.shutdown(), timeout=5.0)
                logger.debug("Executor shutdown successfully")
            except asyncio.TimeoutError:
                logger.warning("Executor shutdown timed out after 5s")
                errors.append("Executor shutdown timed out")
            except Exception as exc:
                logger.warning(f"Error shutting down executor: {exc}")
                errors.append(f"Executor: {exc}")

        if errors:
            logger.warning(f"Cleanup completed with errors: {'; '.join(errors)}")

    # Tool discovery
    async def get_all_tools(self) -> List[ToolInfo]:
        """Return all available tools."""
        if not self._registry:
            logger.warning("get_all_tools called but registry is None")
            return []

        tools = []  # type: ignore[unreachable]
        try:
            logger.debug("get_all_tools: Listing tools from registry")
            registry_items = await asyncio.wait_for(
                self._registry.list_tools(), timeout=30.0
            )
            logger.info(f"get_all_tools: Found {len(registry_items)} tools in registry")

            for ns, name in registry_items:
                try:
                    metadata = await asyncio.wait_for(
                        self._registry.get_metadata(name, ns), timeout=5.0
                    )

                    tools.append(
                        ToolInfo(
                            name=name,
                            namespace=ns,
                            description=metadata.description if metadata else "",
                            parameters=metadata.argument_schema if metadata else {},
                            is_async=metadata.is_async if metadata else False,
                            tags=list(metadata.tags) if metadata else [],
                            supports_streaming=getattr(
                                metadata, "supports_streaming", False
                            )
                            if metadata
                            else False,
                        )
                    )

                except asyncio.TimeoutError:
                    logger.warning(f"Timeout getting metadata for {ns}.{name}")
                except Exception as e:
                    logger.warning(f"Error getting metadata for {ns}.{name}: {e}")

        except Exception as exc:
            logger.error(f"Error discovering tools: {exc}")

        return tools

    async def get_unique_tools(self) -> List[ToolInfo]:
        """Return tools without duplicates from the default namespace."""
        seen = set()
        unique = []

        for tool in await self.get_all_tools():
            if tool.namespace == "default" or tool.name in seen:
                continue
            seen.add(tool.name)
            unique.append(tool)

        return unique

    async def get_tool_by_name(
        self, tool_name: str, namespace: str | None = None
    ) -> Optional[ToolInfo]:
        """Get tool info by name and optional namespace."""
        if not self._registry:
            return None

        if namespace:  # type: ignore[unreachable]
            try:
                metadata = await asyncio.wait_for(
                    self._registry.get_metadata(tool_name, namespace), timeout=5.0
                )
                if metadata:
                    return ToolInfo(
                        name=tool_name,
                        namespace=namespace,
                        description=metadata.description,
                        parameters=metadata.argument_schema,
                        is_async=metadata.is_async,
                        tags=list(metadata.tags),
                        supports_streaming=getattr(
                            metadata, "supports_streaming", False
                        ),
                    )
            except Exception:
                pass

        # Search all non-default namespaces
        for tool in await self.get_all_tools():
            if tool.name == tool_name and tool.namespace != "default":
                return tool

        return None

    def _is_oauth_error(self, error: Union[Exception, str]) -> bool:
        """
        Check if an error is related to OAuth authentication failure.

        Args:
            error: Exception or error string to check

        Returns:
            True if this appears to be an OAuth/authentication error
        """
        # Convert to string if it's an exception
        if isinstance(error, Exception):
            error_msg = str(error).lower()
        else:
            error_msg = str(error).lower() if error else ""

        # Check for common OAuth error indicators
        oauth_indicators = [
            "401",
            "invalid_token",
            "invalid access token",
            "unauthorized",
            "oauth validation failed",
            "expired token",
            "token expired",
        ]

        # Check the main error message
        if any(indicator in error_msg for indicator in oauth_indicators):
            return True

        # Also check the exception chain if it's an Exception object
        if isinstance(error, Exception):
            if hasattr(error, "__cause__") and error.__cause__:
                cause_msg = str(error.__cause__).lower()
                if any(indicator in cause_msg for indicator in oauth_indicators):
                    return True

            # Check exception args
            if hasattr(error, "args"):
                for arg in error.args:
                    if isinstance(arg, str) and any(
                        indicator in arg.lower() for indicator in oauth_indicators
                    ):
                        return True

        return False

    async def _attempt_oauth_reauth_for_namespace(self, namespace: str) -> bool:
        """
        Attempt to re-authenticate OAuth for a given namespace (server).

        Args:
            namespace: The server namespace that needs re-authentication

        Returns:
            True if re-authentication succeeded, False otherwise
        """
        try:
            # Find the server name from the namespace
            server_name = namespace
            if not server_name or server_name == "default":
                logger.debug("Cannot re-authenticate: invalid or default namespace")
                return False

            # Get the server configuration
            server_config = await self._get_server_config_by_name(server_name)
            if not server_config:
                logger.debug(
                    f"Cannot re-authenticate: no config found for {server_name}"
                )
                return False

            # Only attempt OAuth re-auth for HTTP/SSE servers
            if "url" not in server_config:
                logger.debug(
                    f"Cannot re-authenticate: {server_name} is not an HTTP/SSE server"
                )
                return False

            logger.info(
                f"Clearing stored tokens for {server_name} to force re-authentication..."
            )
            self.oauth_handler.clear_tokens(server_name)

            # Delete client registration file if it exists
            config_dir = Path.home() / ".mcp_cli" / "oauth"
            client_file = config_dir / f"{server_name}_client.json"
            if client_file.exists():
                client_file.unlink()
                logger.debug(f"Deleted client registration file: {client_file}")

            # Get new OAuth headers (this will trigger full OAuth flow)
            logger.info(f"Starting OAuth re-authentication for {server_name}...")
            headers = await self.oauth_handler.prepare_server_headers(server_config)

            if not headers or "Authorization" not in headers:
                logger.error(
                    f"Re-authentication failed: no Authorization header returned for {server_name}"
                )
                return False

            logger.info(f"✅ OAuth re-authentication completed for {server_name}")

            # Update the server entry with new auth headers
            server_entry = None
            for entry in self._http_servers + self._sse_servers:
                if entry.get("name") == server_name:
                    server_entry = entry
                    break

            if server_entry:
                # Update api_key
                auth_value = headers.get("Authorization", "")
                if auth_value.startswith("Bearer "):
                    auth_value = auth_value[7:]
                server_entry["api_key"] = auth_value
                logger.debug(f"Updated server_entry api_key for {server_name}")

                # Update transport api_key if it exists
                if self.stream_manager:
                    transport = self.stream_manager.transports.get(server_name)
                    if transport and hasattr(transport, "api_key"):
                        transport.api_key = auth_value
                        logger.debug(f"Updated transport.api_key for {server_name}")

            return True

        except Exception as e:
            logger.error(f"OAuth re-authentication failed for {namespace}: {e}")
            return False

    async def _get_server_config_by_name(self, server_name: str) -> Optional[Any]:
        """
        Get server configuration by server name as a ServerConfig object.

        Args:
            server_name: Name of the server

        Returns:
            ServerConfig object or None if not found
        """
        try:
            from mcp_cli.config.config_manager import initialize_config

            # Use initialize_config to get a proper config manager
            config = initialize_config(Path(self.config_file))
            return config.get_server(server_name)

        except Exception as e:
            logger.error(f"Failed to get server config for {server_name}: {e}")
            return None

    # Tool execution
    async def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any], timeout: Optional[float] = None
    ) -> ToolCallResult:
        """Execute a tool and return the result."""
        if not isinstance(arguments, dict):
            return ToolCallResult(  # type: ignore[unreachable]
                tool_name=tool_name, success=False, error="Arguments must be a dict"
            )

        # Check if tool is enabled
        if not self.tool_filter.is_tool_enabled(tool_name):
            disabled_reason = self.tool_filter.get_disabled_tools().get(
                tool_name, "unknown"
            )
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool disabled ({disabled_reason})",
            )

        # CLEAN: Just look up the tool directly in the registry
        namespace, base_name = await self._find_tool_in_registry(tool_name)

        if not namespace:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' not found in registry",
            )

        logger.debug(
            f"Found tool '{tool_name}' -> namespace='{namespace}', base_name='{base_name}'"
        )

        # Determine the timeout to use for this call
        # Priority: explicit timeout > server-configured timeout > fallback default
        effective_timeout = timeout or self._effective_timeout or self.tool_timeout

        call = ToolCall(
            tool=base_name,
            namespace=namespace,
            arguments=arguments,
        )

        # Try to execute the tool, with OAuth re-authentication retry if needed
        max_oauth_retries = 1
        for attempt in range(max_oauth_retries + 1):
            try:
                if not self._executor:
                    return ToolCallResult(
                        tool_name=tool_name,
                        success=False,
                        error="Tool executor not initialized",
                    )

                results = await self._executor.execute(
                    [call], timeout=effective_timeout
                )

                if not results:
                    logger.error("No results returned from executor")
                    return ToolCallResult(
                        tool_name=tool_name, success=False, error="No result returned"
                    )

                result = results[0]

                # Check for errors in both result.error and result.result['error']
                error_to_check = None
                if result.error:
                    error_to_check = result.error
                elif isinstance(result.result, dict) and "error" in result.result:
                    error_to_check = result.result["error"]

                # Log errors at debug level
                if error_to_check:
                    logger.debug(
                        f"Tool result has error: {error_to_check[:200] if len(error_to_check) > 200 else error_to_check}"
                    )

                # Check if the result contains an OAuth error
                if error_to_check and self._is_oauth_error(error_to_check):
                    if attempt < max_oauth_retries:
                        logger.warning(
                            f"🔐 OAuth error detected in tool result: {error_to_check[:200]}..."
                        )
                        logger.info(
                            f"🔄 Attempting OAuth re-authentication for namespace '{namespace}' (attempt {attempt + 1}/{max_oauth_retries})..."
                        )

                        # Try to re-authenticate
                        if await self._attempt_oauth_reauth_for_namespace(namespace):
                            logger.info(
                                "✅ Re-authentication successful, retrying tool execution..."
                            )
                            continue  # Retry the tool execution
                        else:
                            logger.warning(
                                "Re-authentication failed or not applicable for this server"
                            )
                    else:
                        logger.error(
                            f"OAuth error persists after {max_oauth_retries} retry attempts"
                        )

                return ToolCallResult(
                    tool_name=tool_name,
                    success=not bool(result.error),
                    result=result.result,
                    error=result.error,
                    execution_time=(
                        (result.end_time - result.start_time).total_seconds()
                        if hasattr(result, "end_time") and hasattr(result, "start_time")
                        else None
                    ),
                )
            except Exception as exc:
                logger.error(f"Tool execution exception: {exc}")

                # Check if this is an OAuth error and we haven't retried yet
                if self._is_oauth_error(exc) and attempt < max_oauth_retries:
                    logger.warning(f"OAuth error detected in exception: {exc}")
                    logger.info(
                        f"Attempting OAuth re-authentication for namespace '{namespace}' (attempt {attempt + 1}/{max_oauth_retries})..."
                    )

                    # Try to re-authenticate
                    if await self._attempt_oauth_reauth_for_namespace(namespace):
                        logger.info(
                            "✅ Re-authentication successful, retrying tool execution..."
                        )
                        continue  # Retry the tool execution
                    else:
                        logger.warning(
                            "Re-authentication failed or not applicable for this server"
                        )

                # If not an OAuth error, or retry failed, return the error
                return ToolCallResult(
                    tool_name=tool_name, success=False, error=str(exc)
                )

        # Should not reach here, but just in case
        return ToolCallResult(
            tool_name=tool_name, success=False, error="Max retries exceeded"
        )

    async def _find_tool_in_registry(self, tool_name: str) -> Tuple[str, str]:
        """
        Find a tool in the registry exactly as it exists.
        Resolve namespace for the tool name.

        Returns:
            Tuple of (namespace, tool_name) or ("", "") if not found
        """
        if not self._registry:
            logger.debug("No registry available")
            return "", ""

        try:  # type: ignore[unreachable]
            # Get all available tools from registry
            registry_items = await asyncio.wait_for(
                self._registry.list_tools(), timeout=10.0
            )

            logger.info(
                f"Looking for tool '{tool_name}' in {len(registry_items)} registry entries"
            )

            # Find the tool and its namespace
            for namespace, name in registry_items:
                if name == tool_name:
                    logger.info(f"Found tool '{tool_name}' in namespace '{namespace}'")
                    return namespace, name

            logger.error(f"Tool '{tool_name}' not found in any namespace")
            logger.debug("Available tools:")
            for namespace, name in registry_items[:10]:  # Show first 10 for debugging
                logger.debug(f"  {namespace}/{name}")

            return "", ""

        except Exception as e:
            logger.error(f"Error looking up tool '{tool_name}' in registry: {e}")
            return "", ""

    async def stream_execute_tool(
        self, tool_name: str, arguments: Dict[str, Any], timeout: Optional[float] = None
    ) -> AsyncIterator[ToolResult]:
        """Execute a tool with streaming support."""
        # Check if tool is enabled
        if not self.tool_filter.is_tool_enabled(tool_name):
            from chuk_tool_processor.models.tool_result import ToolResult
            from chuk_tool_processor.models.tool_call import ToolCall

            disabled_reason = self.tool_filter.get_disabled_tools().get(
                tool_name, "unknown"
            )
            dummy_call = ToolCall(tool=tool_name, namespace="", arguments={})
            error_result = ToolResult(
                tool_call=dummy_call,
                result=None,
                error=f"Tool disabled ({disabled_reason})",
            )
            yield error_result
            return

        # CLEAN: Same direct lookup
        namespace, base_name = await self._find_tool_in_registry(tool_name)

        if not namespace:
            dummy_call = ToolCall(tool=tool_name, namespace="", arguments={})
            error_result = ToolResult(
                tool_call=dummy_call,
                result=None,
                error=f"Tool '{tool_name}' not found in registry",
            )
            yield error_result
            return

        call = ToolCall(
            tool=base_name,
            namespace=namespace,
            arguments=arguments,
            timeout=timeout or self._effective_timeout or self.tool_timeout,
        )

        if self._executor:
            async for result in self._executor.stream_execute([call]):
                yield result
        else:
            dummy_call = ToolCall(tool=tool_name, namespace="", arguments={})
            error_result = ToolResult(
                tool_call=dummy_call, result=None, error="Tool executor not initialized"
            )
            yield error_result

    async def process_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
        name_mapping: Dict[str, str],
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ToolResult]:
        """Process tool calls from an LLM."""
        chuk_calls = []
        call_mapping = {}

        for tc in tool_calls:
            if not (tc.get("function") and "name" in tc.get("function", {})):
                continue

            # LLM tool name (possibly sanitized by LLM provider)
            llm_tool_name = tc["function"]["name"]
            tool_call_id = (
                tc.get("id") or f"call_{llm_tool_name}_{uuid.uuid4().hex[:8]}"
            )

            # Get the original tool name from the mapping provided by LLM provider
            # This mapping should handle all sanitization/conversion
            original_tool_name = name_mapping.get(llm_tool_name, llm_tool_name)

            logger.debug(
                f"Tool call: LLM name='{llm_tool_name}' -> Original name='{original_tool_name}'"
            )

            # Check if tool is enabled
            if not self.tool_filter.is_tool_enabled(original_tool_name):
                logger.warning(f"Skipping disabled tool: {original_tool_name}")
                continue

            # Parse arguments
            args_str = tc["function"].get("arguments", "{}")
            try:
                args_dict = (
                    json.loads(args_str)
                    if isinstance(args_str, str) and args_str.strip()
                    else args_str or {}
                )
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool arguments for {llm_tool_name}: {e}")
                args_dict = {}

            # CLEAN: Direct registry lookup using the original name
            namespace, base_name = await self._find_tool_in_registry(original_tool_name)

            # Skip if we can't find the tool
            if not namespace:
                logger.error(f"Tool not found in registry: {original_tool_name}")
                continue

            call = ToolCall(
                tool=base_name,
                namespace=namespace,
                arguments=args_dict,
                metadata={"call_id": tool_call_id, "original_name": original_tool_name},
            )

            chuk_calls.append(call)
            call_mapping[id(call)] = {
                "id": tool_call_id,
                "name": llm_tool_name,
            }  # Use LLM name for conversation

            # Add to conversation history using LLM tool name (for consistency with LLM)
            if conversation_history is not None:
                conversation_history.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": llm_tool_name,  # Use LLM name in conversation
                                    "arguments": json.dumps(args_dict),
                                },
                            }
                        ],
                    }
                )

        # Execute tool calls
        if not self._executor:
            # Return empty results if executor not available
            return []
        results = await self._executor.execute(chuk_calls)

        # Process results
        for result in results:
            call_info = call_mapping.get(
                id(result.tool_call),
                {"id": f"call_{uuid.uuid4().hex[:8]}", "name": result.tool},
            )

            if conversation_history is not None:
                content = (
                    f"Error: {result.error}"
                    if result.error
                    else self.format_tool_response(result.result)
                )
                conversation_history.append(
                    {
                        "role": "tool",
                        "name": call_info["name"],  # Use LLM name for consistency
                        "content": content,
                        "tool_call_id": call_info["id"],
                    }
                )

        typed_results: List[ToolResult] = results
        return typed_results

    # Server helpers
    async def get_server_info(self) -> List[ServerInfo]:
        """Get information about all connected servers."""
        if not self.stream_manager:
            return []

        try:
            if hasattr(self.stream_manager, "get_server_info"):
                server_info_result = self.stream_manager.get_server_info()

                if hasattr(server_info_result, "__await__"):
                    raw_infos = await asyncio.wait_for(server_info_result, timeout=10.0)
                else:
                    raw_infos = server_info_result

                return [
                    ServerInfo(
                        id=raw.get("id", 0),
                        name=raw.get("name", "Unknown"),
                        status=raw.get("status", "Unknown"),
                        tool_count=raw.get("tools", 0),
                        namespace=raw.get("name", "").split("_")[0]
                        if "_" in raw.get("name", "")
                        else raw.get("name", ""),
                        enabled=raw.get("enabled", True),
                        connected=raw.get("connected", False),
                        transport=raw.get("type", "stdio"),
                        capabilities=raw.get("capabilities", {}),
                    )
                    for raw in raw_infos
                ]

        except Exception as exc:
            logger.error(f"Error getting server info: {exc}")

        # Fallback to basic info
        return [
            ServerInfo(
                id=i,
                name=server,
                status="Unknown",
                tool_count=0,
                namespace=server.split("_")[0] if "_" in server else server,
            )
            for i, server in enumerate(self.servers)
        ]

    async def get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Get the server name for a tool."""
        # CLEAN: Just get namespace from registry lookup
        namespace, _ = await self._find_tool_in_registry(tool_name)
        return namespace if namespace else None

    # LLM helpers
    async def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool definitions (validated)."""
        valid_tools, _ = await self.get_adapted_tools_for_llm("openai")
        return valid_tools

    async def get_adapted_tools_for_llm(
        self, provider: str = "openai"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Get tools adapted for the specified LLM provider with validation."""
        # Get raw tools first
        raw_tools, raw_name_mapping = await self._get_raw_adapted_tools_for_llm(
            provider
        )

        # Apply validation and filtering
        valid_tools, invalid_tools = self.tool_filter.filter_tools(raw_tools, provider)

        # Update name mapping to only include valid tools
        valid_tool_names = {
            self.tool_filter._extract_tool_name(tool) for tool in valid_tools
        }
        filtered_name_mapping = {
            k: v
            for k, v in raw_name_mapping.items()
            if any(
                v.endswith(name.split(".")[-1]) or v == name
                for name in valid_tool_names
            )
        }

        # Store validation results
        self.validation_results = {
            "provider": provider,
            "total_tools": len(raw_tools),
            "valid_tools": len(valid_tools),
            "invalid_tools": len(invalid_tools),
            "disabled_tools": self.tool_filter.get_disabled_tools(),
            "validation_errors": [
                {
                    "tool": self.tool_filter._extract_tool_name(tool),
                    "error": tool.get("_validation_error"),
                    "reason": tool.get("_disabled_reason"),
                }
                for tool in invalid_tools
                if tool.get("_validation_error")
            ],
        }
        self.last_validation_provider = provider

        if invalid_tools:
            logger.warning(
                f"Tool validation for {provider}: {len(valid_tools)} valid, {len(invalid_tools)} invalid"
            )
            # Log specific errors for debugging
            for error in self.validation_results["validation_errors"]:
                logger.debug(
                    f"Tool '{error['tool']}' validation error: {error['error']}"
                )
        else:
            logger.info(
                f"Tool validation for {provider}: {len(valid_tools)} valid, 0 invalid"
            )

        return valid_tools, filtered_name_mapping

    async def _get_raw_adapted_tools_for_llm(
        self, provider: str = "openai"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Get raw tools adapted for the specified LLM provider (without validation)."""
        unique_tools = await self.get_unique_tools()

        llm_tools = []
        name_mapping = {}

        logger.info(f"Creating LLM tools for provider '{provider}'")
        logger.info(f"Processing {len(unique_tools)} unique tools:")

        for tool in unique_tools:
            # NO MANIPULATION - use tool name exactly as it exists in registry
            tool_name = tool.name
            logger.info(f"  Tool: namespace='{tool.namespace}', name='{tool.name}'")

            # NO SANITIZATION - pass through tool name as-is
            # The LLM provider handles any necessary name conversion
            name_mapping[tool_name] = (
                tool.name
            )  # Identity mapping - no conversion needed

            llm_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool.description or "",
                        "parameters": tool.parameters or {},
                    },
                }
            )

        logger.info(f"Final name mapping (identity): {name_mapping}")
        return llm_tools, name_mapping

    # ENHANCED: Tool management methods
    def disable_tool(self, tool_name: str, reason: str = "user") -> None:
        """Disable a specific tool."""
        self.tool_filter.disable_tool(tool_name, reason)

    def enable_tool(self, tool_name: str) -> None:
        """Re-enable a specific tool."""
        self.tool_filter.enable_tool(tool_name)

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a tool is enabled."""
        return self.tool_filter.is_tool_enabled(tool_name)

    def get_disabled_tools(self) -> Dict[str, str]:
        """Get all disabled tools with their reasons."""
        return self.tool_filter.get_disabled_tools()

    def set_auto_fix_enabled(self, enabled: bool) -> None:
        """Enable or disable automatic fixing of tool schemas."""
        self.tool_filter.auto_fix_enabled = enabled
        logger.info(f"Auto-fix {'enabled' if enabled else 'disabled'}")

    def is_auto_fix_enabled(self) -> bool:
        """Check if auto-fix is enabled."""
        return self.tool_filter.auto_fix_enabled

    def clear_validation_disabled_tools(self) -> None:
        """Clear all tools disabled due to validation errors."""
        self.tool_filter.clear_validation_disabled()

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get a summary of the last validation run."""
        summary = self.tool_filter.get_validation_summary()
        summary.update(self.validation_results)
        return summary

    async def revalidate_tools(self, provider: str | None = None) -> Dict[str, Any]:
        """
        Re-run validation on all tools.

        Args:
            provider: Provider to validate for (defaults to last used)

        Returns:
            Validation summary
        """
        target_provider = provider or self.last_validation_provider or "openai"

        # Clear validation-disabled tools
        self.clear_validation_disabled_tools()

        # Re-run validation
        _, _ = await self.get_adapted_tools_for_llm(target_provider)

        return self.get_validation_summary()

    async def validate_single_tool(
        self, tool_name: str, provider: str = "openai"
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a single tool by name.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Get the tool definition
        all_tools = await self.get_unique_tools()
        target_tool = None

        for tool in all_tools:
            if tool.name == tool_name or f"{tool.namespace}.{tool.name}" == tool_name:
                target_tool = tool
                break

        if not target_tool:
            return False, f"Tool '{tool_name}' not found"

        # Convert to LLM format
        llm_tools, _ = await self._get_raw_adapted_tools_for_llm(provider)

        # Find the corresponding LLM tool
        llm_tool = None
        for tool_def in llm_tools:
            if self.tool_filter._extract_tool_name(tool_def) == tool_name:
                llm_tool = tool_def
                break

        if not llm_tool:
            return False, f"Tool '{tool_name}' not found in LLM format"

        # Validate
        if provider == "openai":
            return ToolSchemaValidator.validate_openai_schema(llm_tool)
        else:
            return True, None  # Assume valid for other providers

    def get_tool_validation_details(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed validation information for a specific tool."""
        disabled_tools = self.get_disabled_tools()

        if tool_name in disabled_tools:
            reason = disabled_tools[tool_name]

            # Find validation error in results
            validation_error = None
            for error in self.validation_results.get("validation_errors", []):
                if error["tool"] == tool_name:
                    validation_error = error["error"]
                    break

            return {
                "tool_name": tool_name,
                "is_enabled": False,
                "disabled_reason": reason,
                "validation_error": validation_error,
                "can_auto_fix": self.tool_filter.auto_fix_enabled
                and reason == "validation",
            }
        else:
            return {
                "tool_name": tool_name,
                "is_enabled": True,
                "disabled_reason": None,
                "validation_error": None,
                "can_auto_fix": False,
            }

    # Formatting helpers
    @staticmethod
    def format_tool_response(response_content: Union[List[Dict[str, Any]], Any]) -> str:
        """Format tool response content for LLM consumption."""
        if (
            isinstance(response_content, list)
            and response_content
            and isinstance(response_content[0], dict)
        ):
            if all(
                isinstance(item, dict) and item.get("type") == "text"
                for item in response_content
            ):
                return "\n".join(item.get("text", "") for item in response_content)
            try:
                return json.dumps(response_content, indent=2)
            except Exception:
                return str(response_content)
        elif isinstance(response_content, dict):
            try:
                return json.dumps(response_content, indent=2)
            except Exception:
                return str(response_content)
        else:
            return str(response_content)

    # Configuration
    def set_tool_timeout(self, timeout: float) -> None:
        """Update the tool execution timeout."""
        self.tool_timeout = timeout
        if self._executor and hasattr(self._executor.strategy, "default_timeout"):
            self._executor.strategy.default_timeout = timeout

    def get_tool_timeout(self) -> float:
        """Get the current tool timeout value."""
        return self.tool_timeout

    # Resource access (kept for compatibility)
    async def list_prompts(self) -> List[Dict[str, Any]]:
        """Return all prompts from servers."""
        if self.stream_manager and hasattr(self.stream_manager, "list_prompts"):
            try:
                return await asyncio.wait_for(
                    self.stream_manager.list_prompts(), timeout=10.0
                )
            except Exception:
                pass
        return []

    async def list_resources(self) -> List[Dict[str, Any]]:
        """Return all resources from servers."""
        if self.stream_manager and hasattr(self.stream_manager, "list_resources"):
            try:
                return await asyncio.wait_for(
                    self.stream_manager.list_resources(), timeout=10.0
                )
            except Exception:
                pass
        return []

    def get_streams(self):
        """
        Get streams from the stream manager for backward compatibility.

        Returns:
            Generator of (read_stream, write_stream) tuples
        """
        if self.stream_manager:
            return self.stream_manager.get_streams()
        return []


# Global singleton
_tool_manager: Optional[ToolManager] = None


def get_tool_manager() -> Optional[ToolManager]:
    """Get the global tool manager instance."""
    return _tool_manager


def set_tool_manager(manager: ToolManager) -> None:
    """Set the global tool manager instance."""
    global _tool_manager
    _tool_manager = manager
