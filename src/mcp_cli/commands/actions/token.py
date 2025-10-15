"""Token management actions for MCP CLI."""

from __future__ import annotations

import json
from typing import Optional

from chuk_term.ui import output, format_table
from mcp_cli.auth.token_manager import TokenManager
from mcp_cli.auth.token_store_factory import TokenStoreBackend, TokenStoreFactory
from mcp_cli.auth.token_types import APIKeyToken, BearerToken, TokenType
from mcp_cli.config.config_manager import get_config


def _get_token_manager() -> TokenManager:
    """Get configured token manager instance."""
    try:
        config = get_config()
        backend = TokenStoreBackend(config.token_store_backend)
    except Exception:
        backend = TokenStoreBackend.AUTO

    return TokenManager(backend=backend)


async def token_list_action_async(
    namespace: Optional[str] = None,
    show_oauth: bool = True,
    show_bearer: bool = True,
    show_api_keys: bool = True,
) -> None:
    """
    List all stored tokens (metadata only, no sensitive data).

    Args:
        namespace: Filter by namespace
        show_oauth: Show OAuth tokens
        show_bearer: Show bearer tokens
        show_api_keys: Show API keys
    """
    try:
        manager = _get_token_manager()
        store = manager.token_store

        output.rule("[bold]🔐 Stored Tokens[/bold]", style="primary")

        # List OAuth tokens note
        if show_oauth:
            output.info(
                "OAuth tokens are stored per server. "
                "Use 'mcp-cli servers' to see OAuth-enabled servers."
            )
            output.print()

        # List tokens from registry
        registry = manager.registry
        registered_tokens = registry.list_tokens(namespace=namespace)

        table_data = []
        for entry in registered_tokens:
            token_type = entry.get("type", "unknown")
            token_name = entry.get("name", "unknown")
            token_namespace = entry.get("namespace", "unknown")

            # Filter by type
            if token_type == TokenType.BEARER.value and not show_bearer:
                continue
            if token_type == TokenType.API_KEY.value and not show_api_keys:
                continue

            # Format created date
            import time
            from datetime import datetime

            registered_at = entry.get("registered_at")
            created = "-"
            if registered_at:
                dt = datetime.fromtimestamp(registered_at)
                created = dt.strftime('%Y-%m-%d')

            # Get expires info from metadata
            metadata = entry.get("metadata", {})
            expires = metadata.get("expires_at", "-")
            if expires != "-" and isinstance(expires, (int, float)):
                exp_dt = datetime.fromtimestamp(expires)
                # Check if expired
                if time.time() > expires:
                    expires = f"{exp_dt.strftime('%Y-%m-%d')} ⚠️"
                else:
                    expires = exp_dt.strftime('%Y-%m-%d')

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
            table = format_table(
                table_data,
                title=None,
                columns=["Type", "Name", "Created", "Expires", "Details"],
            )
            output.print_table(table)
        else:
            output.warning("No tokens found.")

        output.print()
        output.tip("💡 Token Management:")
        output.info("  • Store: mcp-cli token set <name> --type bearer")
        output.info("  • View: mcp-cli token get <name>")
        output.info("  • Delete: mcp-cli token delete <name>")

    except Exception as e:
        output.error(f"Error listing tokens: {e}")
        raise


async def token_set_action_async(
    name: str,
    token_type: str = "bearer",
    value: Optional[str] = None,
    provider: Optional[str] = None,
    namespace: str = "generic",
) -> None:
    """
    Store a token manually.

    Args:
        name: Token identifier/name
        token_type: Token type (bearer, api-key, generic)
        value: Token value (will prompt if not provided)
        provider: Provider name (for API keys)
        namespace: Storage namespace
    """
    try:
        manager = _get_token_manager()
        store = manager.token_store

        # Prompt for value if not provided
        if value is None:
            from getpass import getpass

            value = getpass(f"Enter token value for '{name}': ")

        if not value:
            output.error("Token value is required")
            return

        # Store based on type
        registry = manager.registry

        if token_type == "bearer":
            bearer = BearerToken(token=value)
            stored = bearer.to_stored_token(name)
            stored.metadata = {"namespace": namespace}
            store._store_raw(f"{namespace}:{name}", json.dumps(stored.to_dict()))

            # Register in index with expiration if available
            reg_metadata = {}
            if bearer.expires_at:
                reg_metadata["expires_at"] = bearer.expires_at

            registry.register(name, TokenType.BEARER, namespace, metadata=reg_metadata)
            output.success(f"Bearer token '{name}' stored successfully")

        elif token_type == "api-key":
            if not provider:
                output.error("Provider name is required for API keys")
                output.hint("Use: --provider <name>")
                return

            api_key = APIKeyToken(provider=provider, key=value)
            stored = api_key.to_stored_token(name)
            stored.metadata = {"namespace": namespace}
            store._store_raw(f"{namespace}:{name}", json.dumps(stored.to_dict()))

            # Register in index
            registry.register(
                name, TokenType.API_KEY, namespace, metadata={"provider": provider}
            )
            output.success(
                f"API key '{name}' for '{provider}' stored successfully"
            )

        elif token_type == "generic":
            store.store_generic(name, value, namespace)

            # Register in index
            registry.register(name, TokenType.BEARER, namespace, metadata={})
            output.success(
                f"Token '{name}' stored in namespace '{namespace}'"
            )

        else:
            output.error(f"Unknown token type: {token_type}")
            output.hint("Valid types: bearer, api-key, generic")

    except Exception as e:
        output.error(f"Error storing token: {e}")
        raise


async def token_get_action_async(
    name: str, namespace: str = "generic"
) -> None:
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
            output.warning(
                f"Token '{name}' not found in namespace '{namespace}'"
            )
            return

        try:
            from mcp_cli.auth.token_types import StoredToken

            stored = StoredToken.from_dict(json.loads(raw_data))
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


async def token_delete_action_async(
    name: str, namespace: Optional[str] = None, oauth: bool = False
) -> None:
    """
    Delete a stored token.

    Args:
        name: Token identifier/name
        namespace: Storage namespace (tries common namespaces if not specified)
        oauth: Delete OAuth token for server
    """
    try:
        manager = _get_token_manager()
        store = manager.token_store
        registry = manager.registry

        if oauth:
            # Delete OAuth token
            if manager.delete_tokens(name):
                output.success(f"OAuth token for server '{name}' deleted")
            else:
                output.warning(f"OAuth token for server '{name}' not found")
            return

        # Delete generic token
        if namespace:
            namespaces = [namespace]
        else:
            namespaces = ["bearer", "api-key", "provider", "generic"]

        deleted = False
        for ns in namespaces:
            if store.delete_generic(name, ns):
                # Unregister from index
                registry.unregister(name, ns)
                output.success(
                    f"Token '{name}' deleted from namespace '{ns}'"
                )
                deleted = True
                break

        if not deleted:
            output.warning(f"Token '{name}' not found")

    except Exception as e:
        output.error(f"Error deleting token: {e}")
        raise


async def token_clear_action_async(
    namespace: Optional[str] = None, force: bool = False
) -> None:
    """
    Clear all stored tokens.

    Args:
        namespace: Clear only tokens in this namespace
        force: Skip confirmation prompt
    """
    try:
        manager = _get_token_manager()
        store = manager.token_store
        registry = manager.registry

        # Confirm before clearing
        if not force:
            if namespace:
                msg = f"Clear all tokens in namespace '{namespace}'?"
            else:
                msg = "Clear ALL tokens from ALL namespaces?"

            from chuk_term.ui.prompts import confirm

            if not confirm(msg):
                output.warning("Cancelled")
                return

        # Get tokens to clear from registry
        tokens_to_clear = registry.list_tokens(namespace=namespace)

        if not tokens_to_clear:
            output.warning("No tokens to clear")
            return

        # Clear each token from storage
        count = 0
        for entry in tokens_to_clear:
            token_name = entry.get("name")
            token_namespace = entry.get("namespace")
            if store.delete_generic(token_name, token_namespace):
                count += 1

        # Clear from registry
        if namespace:
            registry.clear_namespace(namespace)
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
    try:
        available = TokenStoreFactory.get_available_backends()
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
        output.info(f"Current backend: {detected.value}")

    except Exception as e:
        output.error(f"Error listing backends: {e}")
        raise


# Export action for use in main.py
async def servers_action_async(**kwargs) -> None:
    """Handle token subcommands (placeholder for main.py integration)."""
    # This will be called from main.py with appropriate routing
    pass
