"""Token management actions for MCP CLI."""

from __future__ import annotations

import json

from chuk_term.ui import output, format_table
from mcp_cli.auth import TokenManager
from mcp_cli.auth import TokenStoreBackend, TokenStoreFactory
from mcp_cli.auth import APIKeyToken, BearerToken, TokenType
from mcp_cli.config.config_manager import get_config
from mcp_cli.constants import NAMESPACE, OAUTH_NAMESPACE
from mcp_cli.commands.models import (
    TokenListParams,
    TokenSetParams,
    TokenDeleteParams,
    TokenClearParams,
    TokenProviderParams,
)


def _get_token_manager() -> TokenManager:
    """Get configured token manager instance with mcp-cli namespace."""
    import os

    # Check for CLI override first
    backend_override = os.environ.get("MCP_CLI_TOKEN_BACKEND")
    if backend_override:
        try:
            backend = TokenStoreBackend(backend_override)
        except (ValueError, KeyError):
            # Invalid backend specified, fall through to config
            backend = None
    else:
        backend = None

    # If no override or invalid override, check config
    if backend is None:
        try:
            config = get_config()
            backend = TokenStoreBackend(config.token_store_backend)
        except Exception:
            backend = TokenStoreBackend.AUTO

    return TokenManager(backend=backend, namespace=NAMESPACE, service_name="mcp-cli")


async def token_list_action_async(params: TokenListParams) -> None:
    """
    List all stored tokens (metadata only, no sensitive data).

    Args:
        params: Token list parameters

    Example:
        >>> params = TokenListParams(namespace="provider", show_oauth=True)
        >>> await token_list_action_async(params)
    """
    try:
        manager = _get_token_manager()

        output.rule("[bold]🔐 Stored Tokens[/bold]", style="primary")

        # Track if we showed any tokens at all
        provider_tokens = {}
        oauth_entries = []

        # Show provider tokens with hierarchical status
        if params.show_providers and (
            params.namespace is None or params.namespace == "provider"
        ):
            from mcp_cli.auth.provider_tokens import (
                list_all_provider_tokens,
            )

            provider_tokens = list_all_provider_tokens(manager)

            if provider_tokens:
                output.print(
                    "\n[bold]Provider API Keys (Stored in Secure Storage):[/bold]"
                )
                provider_table_data = []

                for provider_name, status_info in provider_tokens.items():
                    env_var = status_info["env_var"]

                    # These are all storage tokens
                    status_display = "🔐 storage"

                    # Note if env var also exists (shows hierarchy)
                    if status_info["in_env"]:
                        note = f"(overridden by {env_var})"
                    else:
                        note = "active"

                    provider_table_data.append(
                        {
                            "Provider": provider_name,
                            "Status": status_display,
                            "Env Var": env_var,
                            "Note": note,
                        }
                    )

                provider_table = format_table(
                    provider_table_data,
                    title=None,
                    columns=["Provider", "Status", "Env Var", "Note"],
                )
                output.print_table(provider_table)
                output.info(
                    "💡 Environment variables take precedence over stored tokens"
                )
                output.print()

        # List OAuth tokens - check servers from provided list
        if params.show_oauth and params.server_names:
            # Check each configured server for OAuth tokens
            for server_name in params.server_names:
                # Check if tokens exist for this server
                tokens = manager.load_tokens(server_name)
                if tokens:
                    metadata = {}
                    if tokens.expires_in:
                        import time

                        if tokens.issued_at:
                            metadata["expires_at"] = (
                                tokens.issued_at + tokens.expires_in
                            )
                        else:
                            metadata["expires_at"] = time.time() + tokens.expires_in

                    oauth_entries.append(
                        {
                            "name": server_name,
                            "type": "oauth",
                            "namespace": OAUTH_NAMESPACE,
                            "registered_at": tokens.issued_at
                            if tokens.issued_at
                            else None,
                            "metadata": metadata,
                        }
                    )

            if oauth_entries:
                output.print("\n[bold]OAuth Tokens (Server Authentication):[/bold]")
                oauth_table_data = []

                for entry in oauth_entries:
                    token_name = entry.get("name", "unknown")
                    token_type = entry.get("type", "unknown")

                    # Format created date
                    import time
                    from datetime import datetime

                    registered_at = entry.get("registered_at")
                    created = "-"
                    if registered_at and isinstance(registered_at, (int, float)):
                        dt = datetime.fromtimestamp(registered_at)
                        created = dt.strftime("%Y-%m-%d")

                    # Get expires info from metadata
                    metadata_raw = entry.get("metadata", {})
                    metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
                    expires = metadata.get("expires_at", "-")
                    if expires != "-" and isinstance(expires, (int, float)):
                        exp_dt = datetime.fromtimestamp(expires)
                        # Check if expired
                        if time.time() > expires:
                            expires = f"{exp_dt.strftime('%Y-%m-%d')} ⚠️ Expired"
                        else:
                            expires = exp_dt.strftime("%Y-%m-%d")

                    oauth_table_data.append(
                        {
                            "Server": token_name,
                            "Type": token_type,
                            "Created": created,
                            "Expires": expires,
                        }
                    )

                oauth_table = format_table(
                    oauth_table_data,
                    title=None,
                    columns=["Server", "Type", "Created", "Expires"],
                )
                output.print_table(oauth_table)
                output.info("💡 Use '/token get <server>' to view token details")
                output.print()
        elif params.show_oauth and not params.server_names:
            # Show message when no servers configured
            output.info("No servers configured. OAuth tokens are stored per server.")
            output.print()

        # List tokens from registry (non-provider, non-oauth tokens)
        registry = manager.registry
        registered_tokens = registry.list_tokens(namespace=params.namespace)

        table_data = []
        for entry in registered_tokens:
            token_type = entry.get("type", "unknown")
            token_name = entry.get("name", "unknown")
            token_namespace = entry.get("namespace", "unknown")

            # Skip provider namespace if show_providers is True (already shown above)
            if params.show_providers and token_namespace == "provider":
                continue

            # Skip OAuth namespace if show_oauth is True (already shown above)
            if params.show_oauth and token_namespace == OAUTH_NAMESPACE:
                continue

            # Filter by type
            if token_type == TokenType.BEARER.value and not params.show_bearer:
                continue
            if token_type == TokenType.API_KEY.value and not params.show_api_keys:
                continue

            # Format created date
            import time
            from datetime import datetime

            registered_at = entry.get("registered_at")
            created = "-"
            if registered_at and isinstance(registered_at, (int, float)):
                dt = datetime.fromtimestamp(registered_at)
                created = dt.strftime("%Y-%m-%d")

            # Get expires info from metadata
            metadata_raw = entry.get("metadata", {})
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
            expires = metadata.get("expires_at", "-")
            if expires != "-" and isinstance(expires, (int, float)):
                exp_dt = datetime.fromtimestamp(expires)
                # Check if expired
                if time.time() > expires:
                    expires = f"{exp_dt.strftime('%Y-%m-%d')} ⚠️"
                else:
                    expires = exp_dt.strftime("%Y-%m-%d")

            # Build details (provider, namespace if not generic)
            details = []
            if metadata.get("provider"):
                details.append(f"provider={metadata['provider']}")
            if token_namespace != "generic":
                details.append(f"ns={token_namespace}")

            table_data.append(
                {
                    "Type": token_type,
                    "Name": token_name,
                    "Created": created,
                    "Expires": expires,
                    "Details": ", ".join(details) if details else "-",
                }
            )

        if table_data:
            output.print("\n[bold]Other Tokens:[/bold]")
            table = format_table(
                table_data,
                title=None,
                columns=["Type", "Name", "Created", "Expires", "Details"],
            )
            output.print_table(table)
        elif not provider_tokens and not oauth_entries:
            # Only show "No tokens found" if we truly have no tokens at all
            output.warning("No tokens found.")

        output.print()
        output.tip("💡 Token Management:")
        output.info("  • Store provider key: mcp-cli token set-provider <provider>")
        output.info("  • Store bearer token: mcp-cli token set <name> --type bearer")
        output.info("  • View: mcp-cli token get <name>")
        output.info("  • Delete: mcp-cli token delete <name>")

    except Exception as e:
        output.error(f"Error listing tokens: {e}")
        raise


async def token_set_action_async(params: TokenSetParams) -> None:
    """
    Store a token manually.

    Args:
        params: Token set parameters

    Example:
        >>> params = TokenSetParams(name="my-token", token_type="bearer", value="abc123")
        >>> await token_set_action_async(params)
    """
    try:
        manager = _get_token_manager()
        store = manager.token_store

        # Prompt for value if not provided
        value = params.value
        if value is None:
            from getpass import getpass

            value = getpass(f"Enter token value for '{params.name}': ")

        if not value:
            output.error("Token value is required")
            return

        # Store based on type
        registry = manager.registry

        if params.token_type == "bearer":
            bearer = BearerToken(token=value)
            stored = bearer.to_stored_token(params.name)
            stored.metadata = {"namespace": params.namespace}
            store._store_raw(
                f"{params.namespace}:{params.name}", json.dumps(stored.model_dump())
            )

            # Register in index with expiration if available
            reg_metadata = {}
            if bearer.expires_at:
                reg_metadata["expires_at"] = bearer.expires_at

            registry.register(
                params.name, TokenType.BEARER, params.namespace, metadata=reg_metadata
            )
            output.success(f"Bearer token '{params.name}' stored successfully")

        elif params.token_type == "api-key":
            if not params.provider:
                output.error("Provider name is required for API keys")
                output.hint("Use: --provider <name>")
                return

            api_key = APIKeyToken(provider=params.provider, key=value)
            stored = api_key.to_stored_token(params.name)
            stored.metadata = {"namespace": params.namespace}
            store._store_raw(
                f"{params.namespace}:{params.name}", json.dumps(stored.model_dump())
            )

            # Register in index
            registry.register(
                params.name,
                TokenType.API_KEY,
                params.namespace,
                metadata={"provider": params.provider},
            )
            output.success(
                f"API key '{params.name}' for '{params.provider}' stored successfully"
            )

        elif params.token_type == "generic":
            store.store_generic(params.name, value, params.namespace)

            # Register in index
            registry.register(
                params.name, TokenType.BEARER, params.namespace, metadata={}
            )
            output.success(
                f"Token '{params.name}' stored in namespace '{params.namespace}'"
            )

        else:
            output.error(f"Unknown token type: {params.token_type}")
            output.hint("Valid types: bearer, api-key, generic")

    except Exception as e:
        output.error(f"Error storing token: {e}")
        raise


async def token_get_action_async(name: str, namespace: str = "generic") -> None:
    """
    Get information about a stored token.

    Args:
        name: Token identifier/name
        namespace: Storage namespace
    """
    try:
        manager = _get_token_manager()
        store = manager.token_store

        raw_data = store._retrieve_raw(f"{namespace}:{name}")
        if not raw_data:
            output.warning(f"Token '{name}' not found in namespace '{namespace}'")
            return

        try:
            from mcp_cli.auth import StoredToken

            stored = StoredToken.model_validate(json.loads(raw_data))
            info = stored.get_display_info()

            output.rule(f"[bold]Token: {name}[/bold]", style="primary")
            output.info(f"Type: {stored.token_type.value}")
            output.info(f"Namespace: {namespace}")

            for key, value in info.items():
                if key not in ["name", "type"]:
                    output.info(f"{key}: {value}")

        except Exception as e:
            output.warning(f"Could not parse token data: {e}")

    except Exception as e:
        output.error(f"Error retrieving token: {e}")
        raise


async def token_delete_action_async(params: TokenDeleteParams) -> None:
    """
    Delete a stored token.

    Args:
        params: Token delete parameters

    Example:
        >>> params = TokenDeleteParams(name="my-token", namespace="generic")
        >>> await token_delete_action_async(params)
    """
    try:
        manager = _get_token_manager()
        store = manager.token_store
        registry = manager.registry

        if params.oauth:
            # Delete OAuth token
            if manager.delete_tokens(params.name):
                output.success(f"OAuth token for server '{params.name}' deleted")
            else:
                output.warning(f"OAuth token for server '{params.name}' not found")
            return

        # Delete generic token
        if params.namespace:
            namespaces = [params.namespace]
        else:
            namespaces = ["bearer", "api-key", "provider", "generic"]

        deleted = False
        for ns in namespaces:
            if store.delete_generic(params.name, ns):
                # Unregister from index
                registry.unregister(params.name, ns)
                output.success(f"Token '{params.name}' deleted from namespace '{ns}'")
                deleted = True
                break

        if not deleted:
            output.warning(f"Token '{params.name}' not found")

    except Exception as e:
        output.error(f"Error deleting token: {e}")
        raise


async def token_set_provider_action_async(params: TokenProviderParams) -> None:
    """
    Store a provider API key in secure storage.

    Args:
        params: Token provider parameters

    Example:
        >>> params = TokenProviderParams(provider="openai", api_key="sk-xxx")
        >>> await token_set_provider_action_async(params)
    """
    try:
        from mcp_cli.auth.provider_tokens import (
            set_provider_token,
            get_provider_env_var_name,
        )
        import os

        manager = _get_token_manager()

        # Prompt for api_key if not provided
        api_key = params.api_key
        if api_key is None:
            from getpass import getpass

            api_key = getpass(f"Enter API key for '{params.provider}': ")

        if not api_key:
            output.error("API key is required")
            return

        # Store the token
        if set_provider_token(params.provider, api_key, manager):
            output.success(f"✅ Stored API key for provider '{params.provider}'")

            # Show hierarchy info
            env_var = get_provider_env_var_name(params.provider)
            output.print()
            output.info("📋 Token Hierarchy:")
            output.info(f"  1. Environment variable: {env_var} (highest priority)")
            output.info("  2. Secure storage: 🔐 (currently set)")

            # Check if env var is also set
            if os.environ.get(env_var):
                output.warning(
                    f"\n⚠️  Note: {env_var} is set in environment and will take precedence"
                )
        else:
            output.error(f"Failed to store API key for provider '{params.provider}'")

    except Exception as e:
        output.error(f"Error storing provider token: {e}")
        raise


async def token_get_provider_action_async(params: TokenProviderParams) -> None:
    """
    Get information about a provider's API key.

    Args:
        params: Token provider parameters

    Example:
        >>> params = TokenProviderParams(provider="openai")
        >>> await token_get_provider_action_async(params)
    """
    try:
        from mcp_cli.auth.provider_tokens import (
            check_provider_token_status,
        )

        manager = _get_token_manager()
        status = check_provider_token_status(params.provider, manager)

        output.rule(f"[bold]Provider Token: {params.provider}[/bold]", style="primary")

        # Display status
        if status["has_token"]:
            output.success("✅ API key is configured")
            output.info(f"   Source: {status['source']}")
        else:
            output.warning("❌ No API key configured")

        output.print()
        output.info("Token Status:")
        output.info(
            f"  • Environment variable ({status['env_var']}): {'✅ set' if status['in_env'] else '❌ not set'}"
        )
        output.info(
            f"  • Secure storage: {'✅ set' if status['in_storage'] else '❌ not set'}"
        )

        output.print()
        output.tip("Hierarchy: Environment variables take precedence over storage")

        if not status["has_token"]:
            output.print()
            output.info("To set API key:")
            output.info(
                f"  • Via storage: mcp-cli token set-provider {params.provider}"
            )
            output.info(f"  • Via environment: export {status['env_var']}=your-key")

    except Exception as e:
        output.error(f"Error retrieving provider token info: {e}")
        raise


async def token_delete_provider_action_async(params: TokenProviderParams) -> None:
    """
    Delete a provider API key from secure storage.

    Note: This only removes from storage, not environment variables.

    Args:
        params: Token provider parameters
    """
    try:
        from mcp_cli.auth.provider_tokens import (
            check_provider_token_status,
        )

        provider = params.provider
        manager = _get_token_manager()
        status = check_provider_token_status(provider, manager)

        output.rule(f"[bold]Provider Token: {provider}[/bold]", style="primary")

        # Display status
        if status["has_token"]:
            output.success("✅ API key is configured")
            output.info(f"   Source: {status['source']}")
        else:
            output.warning("❌ No API key configured")

        output.print()
        output.info("Token Status:")
        output.info(
            f"  • Environment variable ({status['env_var']}): {'✅ set' if status['in_env'] else '❌ not set'}"
        )
        output.info(
            f"  • Secure storage: {'✅ set' if status['in_storage'] else '❌ not set'}"
        )

        output.print()
        output.tip("Hierarchy: Environment variables take precedence over storage")

        if not status["has_token"]:
            output.print()
            output.info("To set API key:")
            output.info(
                f"  • Via storage: mcp-cli token set-provider {params.provider}"
            )
            output.info(f"  • Via environment: export {status['env_var']}=your-key")

    except Exception as e:
        output.error(f"Error retrieving provider token info: {e}")
        raise


async def token_clear_action_async(params: TokenClearParams) -> None:
    """
    Clear all stored tokens.

    Args:
        params: Token clear parameters

    Example:
        >>> params = TokenClearParams(namespace="provider", force=True)
        >>> await token_clear_action_async(params)
    """
    try:
        manager = _get_token_manager()
        store = manager.token_store
        registry = manager.registry

        # Confirm before clearing
        if not params.force:
            if params.namespace:
                msg = f"Clear all tokens in namespace '{params.namespace}'?"
            else:
                msg = "Clear ALL tokens from ALL namespaces?"

            from chuk_term.ui.prompts import confirm

            if not confirm(msg):
                output.warning("Cancelled")
                return

        # Get tokens to clear from registry
        tokens_to_clear = registry.list_tokens(namespace=params.namespace)

        if not tokens_to_clear:
            output.warning("No tokens to clear")
            return

        # Clear each token from storage
        count = 0
        for entry in tokens_to_clear:
            token_name = entry.get("name")
            token_namespace = entry.get("namespace")
            if (
                token_name
                and token_namespace
                and store.delete_generic(token_name, token_namespace)
            ):
                count += 1

        # Clear from registry
        if params.namespace:
            registry.clear_namespace(params.namespace)
        else:
            registry.clear_all()

        if count > 0:
            output.success(f"Cleared {count} token(s)")
        else:
            output.warning("No tokens to clear")

    except Exception as e:
        output.error(f"Error clearing tokens: {e}")
        raise


async def token_backends_action_async() -> None:
    """List available token storage backends."""
    import os

    try:
        available = TokenStoreFactory.get_available_backends()

        # Check for CLI override first
        backend_override = os.environ.get("MCP_CLI_TOKEN_BACKEND")
        override_succeeded = False
        if backend_override:
            try:
                detected = TokenStoreBackend(backend_override)
                override_succeeded = True
            except (ValueError, KeyError):
                # Invalid backend specified, use auto-detection
                detected = TokenStoreFactory._detect_backend()
                output.warning(
                    f"Invalid backend '{backend_override}', using auto-detected backend"
                )
        else:
            detected = TokenStoreFactory._detect_backend()

        output.rule("[bold]🔒 Token Storage Backends[/bold]", style="primary")

        all_backends = [
            ("keychain", "macOS Keychain"),
            ("windows", "Windows Credential Manager"),
            ("secretservice", "Linux Secret Service"),
            ("vault", "HashiCorp Vault"),
            ("encrypted", "Encrypted File Storage"),
        ]

        table_data = []
        for backend_id, backend_name in all_backends:
            backend = TokenStoreBackend(backend_id)
            is_available = backend in available
            is_detected = backend == detected

            status = []
            if is_detected:
                status.append("🎯 Auto-detected")
            if is_available:
                status.append("✓")

            table_data.append(
                {
                    "Backend": backend_name,
                    "Available": "✓" if is_available else "✗",
                    "Status": " ".join(status) if status else "-",
                }
            )

        table = format_table(
            table_data, title=None, columns=["Backend", "Available", "Status"]
        )
        output.print_table(table)
        output.print()
        if override_succeeded:
            output.info(
                f"Current backend: {detected.value} (overridden via --token-backend)"
            )
        else:
            output.info(f"Current backend: {detected.value}")

    except Exception as e:
        output.error(f"Error listing backends: {e}")
        raise


# Export action for use in main.py
async def servers_action_async(**kwargs) -> None:
    """Handle token subcommands (placeholder for main.py integration)."""
    # This will be called from main.py with appropriate routing
    pass
