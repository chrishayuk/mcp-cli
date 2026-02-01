# tests/tools/test_config_loader_extended.py
"""Extended tests for ConfigLoader to achieve >90% coverage.

Covers missing lines: 160-192.
These lines are in _resolve_token_placeholders and handle the new
${TOKEN:namespace:name} format for bearer/api-key tokens.
"""

import json
import pytest
from unittest.mock import patch

from mcp_cli.tools.config_loader import (
    ConfigLoader,
    TOKEN_ENV_PREFIX,
    TOKEN_ENV_SUFFIX,
    TOKEN_PLACEHOLDER_PREFIX,
    TOKEN_PLACEHOLDER_SUFFIX,
)


# ────────────────────────────────────────────────────────────────────
# ${TOKEN:namespace:name} format - lines 160-194
# ────────────────────────────────────────────────────────────────────


class TestResolveNewTokenFormat:
    """Test _resolve_token_placeholders with ${TOKEN:namespace:name} format."""

    def test_token_env_format_resolved_with_token_key(self, tmp_path):
        """Lines 160-184: ${TOKEN:ns:name} resolved when stored token has 'token' key."""
        config = {
            "mcpServers": {
                "server": {
                    "env": {
                        "API_KEY": "${TOKEN:bearer:my_api}",
                    }
                }
            }
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        loader = ConfigLoader(str(config_path), ["server"])

        # Mock stored token with 'token' key in data
        stored_token_json = json.dumps(
            {
                "token_type": "bearer",
                "name": "my_api",
                "data": {"token": "my_secret_token_value"},
            }
        )

        with patch.object(
            loader._token_store, "_retrieve_raw", return_value=stored_token_json
        ):
            loaded = loader.load()

        assert (
            loaded["mcpServers"]["server"]["env"]["API_KEY"] == "my_secret_token_value"
        )

    def test_token_env_format_resolved_with_access_token_key(self, tmp_path):
        """Lines 177-179: ${TOKEN:ns:name} resolved via 'access_token' fallback key."""
        config = {
            "mcpServers": {
                "server": {
                    "headers": {
                        "X-Api-Key": "${TOKEN:api_key:service_key}",
                    }
                }
            }
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        loader = ConfigLoader(str(config_path), ["server"])

        # Token data uses 'access_token' instead of 'token'
        stored_token_json = json.dumps(
            {
                "token_type": "api_key",
                "name": "service_key",
                "data": {"access_token": "access_tok_123"},
            }
        )

        with patch.object(
            loader._token_store, "_retrieve_raw", return_value=stored_token_json
        ):
            loaded = loader.load()

        assert (
            loaded["mcpServers"]["server"]["headers"]["X-Api-Key"] == "access_tok_123"
        )

    def test_token_env_format_no_token_value_in_data(self, tmp_path):
        """Lines 185-188: stored token data has neither 'token' nor 'access_token'."""
        config = {"mcpServers": {"server": {"env": {"KEY": "${TOKEN:ns:name}"}}}}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        loader = ConfigLoader(str(config_path), ["server"])

        # Token data has neither 'token' nor 'access_token'
        stored_token_json = json.dumps(
            {
                "token_type": "bearer",
                "name": "name",
                "data": {"some_other_field": "value"},
            }
        )

        with patch.object(
            loader._token_store, "_retrieve_raw", return_value=stored_token_json
        ):
            loaded = loader.load()

        # Should keep the placeholder since token value couldn't be extracted
        assert loaded["mcpServers"]["server"]["env"]["KEY"] == "${TOKEN:ns:name}"

    def test_token_env_format_no_raw_data_found(self, tmp_path):
        """Lines 189-190: token store returns None (token not found)."""
        config = {
            "mcpServers": {"server": {"env": {"KEY": "${TOKEN:ns:missing_token}"}}}
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        loader = ConfigLoader(str(config_path), ["server"])

        with patch.object(loader._token_store, "_retrieve_raw", return_value=None):
            loaded = loader.load()

        # Placeholder kept when token not found
        assert (
            loaded["mcpServers"]["server"]["env"]["KEY"] == "${TOKEN:ns:missing_token}"
        )

    def test_token_env_format_exception_during_lookup(self, tmp_path):
        """Lines 191-194: exception during token lookup is caught and logged."""
        config = {"mcpServers": {"server": {"env": {"KEY": "${TOKEN:ns:bad_token}"}}}}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        loader = ConfigLoader(str(config_path), ["server"])

        with patch.object(
            loader._token_store,
            "_retrieve_raw",
            side_effect=RuntimeError("keychain error"),
        ):
            loaded = loader.load()

        # Placeholder kept on error
        assert loaded["mcpServers"]["server"]["env"]["KEY"] == "${TOKEN:ns:bad_token}"

    def test_token_env_format_with_empty_data(self, tmp_path):
        """Lines 176-178: stored token has data=None."""
        config = {"mcpServers": {"server": {"env": {"KEY": "${TOKEN:ns:no_data}"}}}}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        loader = ConfigLoader(str(config_path), ["server"])

        # Token with data=None
        stored_token_json = json.dumps(
            {
                "token_type": "bearer",
                "name": "no_data",
                "data": None,
            }
        )

        with patch.object(
            loader._token_store, "_retrieve_raw", return_value=stored_token_json
        ):
            loaded = loader.load()

        # Placeholder kept since data is None
        assert loaded["mcpServers"]["server"]["env"]["KEY"] == "${TOKEN:ns:no_data}"

    def test_token_env_format_insufficient_parts(self, tmp_path):
        """Lines 162: ${TOKEN:only_one_part} has fewer than 2 parts after split."""
        config = {"mcpServers": {"server": {"env": {"KEY": "${TOKEN:single_part}"}}}}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        loader = ConfigLoader(str(config_path), ["server"])

        loaded = loader.load()

        # Should keep placeholder since split produces only 1 part
        assert loaded["mcpServers"]["server"]["env"]["KEY"] == "${TOKEN:single_part}"


# ────────────────────────────────────────────────────────────────────
# ${TOKEN:namespace:name} in nested structures
# ────────────────────────────────────────────────────────────────────


class TestResolveNewTokenFormatNested:
    """Test ${TOKEN:ns:name} resolution in nested dicts and lists."""

    def test_token_env_format_in_nested_dict(self, tmp_path):
        """Token in deeply nested dict is resolved."""
        config = {
            "mcpServers": {
                "server": {"nested": {"deep": {"api_key": "${TOKEN:bearer:deep_key}"}}}
            }
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        loader = ConfigLoader(str(config_path), ["server"])

        stored_token_json = json.dumps(
            {
                "token_type": "bearer",
                "name": "deep_key",
                "data": {"token": "deep_token_value"},
            }
        )

        with patch.object(
            loader._token_store, "_retrieve_raw", return_value=stored_token_json
        ):
            loaded = loader.load()

        assert (
            loaded["mcpServers"]["server"]["nested"]["deep"]["api_key"]
            == "deep_token_value"
        )

    def test_token_env_format_in_list(self, tmp_path):
        """Token in list values is resolved."""
        config = {
            "mcpServers": {
                "server": {
                    "tokens": [
                        "${TOKEN:ns:token1}",
                        "${TOKEN:ns:token2}",
                    ]
                }
            }
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        loader = ConfigLoader(str(config_path), ["server"])

        stored_token_json = json.dumps(
            {
                "token_type": "bearer",
                "name": "token1",
                "data": {"token": "resolved_value"},
            }
        )

        with patch.object(
            loader._token_store, "_retrieve_raw", return_value=stored_token_json
        ):
            loaded = loader.load()

        assert loaded["mcpServers"]["server"]["tokens"] == [
            "resolved_value",
            "resolved_value",
        ]

    def test_mixed_token_formats(self, tmp_path):
        """Mix of legacy {{token:provider}} and new ${TOKEN:ns:name} formats."""
        config = {
            "mcpServers": {
                "server": {
                    "headers": {
                        "Authorization": "{{token:github}}",
                        "X-Api-Key": "${TOKEN:api_key:my_key}",
                    }
                }
            }
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        loader = ConfigLoader(str(config_path), ["server"])

        # Mock retrieve_raw to return different tokens based on key
        def mock_retrieve_raw(key):
            if key == "oauth:github":
                return json.dumps(
                    {
                        "token_type": "oauth",
                        "name": "github",
                        "data": {"access_token": "github_token"},
                    }
                )
            elif key == "api_key:my_key":
                return json.dumps(
                    {
                        "token_type": "api_key",
                        "name": "my_key",
                        "data": {"token": "api_key_value"},
                    }
                )
            return None

        with patch.object(
            loader._token_store, "_retrieve_raw", side_effect=mock_retrieve_raw
        ):
            loaded = loader.load()

        assert (
            loaded["mcpServers"]["server"]["headers"]["Authorization"]
            == "Bearer github_token"
        )
        assert loaded["mcpServers"]["server"]["headers"]["X-Api-Key"] == "api_key_value"


# ────────────────────────────────────────────────────────────────────
# ${TOKEN:namespace:name} via load_async
# ────────────────────────────────────────────────────────────────────


class TestResolveNewTokenFormatAsync:
    """Test ${TOKEN:ns:name} resolution through async load path."""

    @pytest.mark.asyncio
    async def test_token_env_format_resolved_via_load_async(self, tmp_path):
        """Token resolution works through load_async as well."""
        config = {
            "mcpServers": {"server": {"env": {"SECRET": "${TOKEN:bearer:async_key}"}}}
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        loader = ConfigLoader(str(config_path), ["server"])

        stored_token_json = json.dumps(
            {
                "token_type": "bearer",
                "name": "async_key",
                "data": {"token": "async_token_value"},
            }
        )

        with patch.object(
            loader._token_store, "_retrieve_raw", return_value=stored_token_json
        ):
            loaded = await loader.load_async()

        assert loaded["mcpServers"]["server"]["env"]["SECRET"] == "async_token_value"


# ────────────────────────────────────────────────────────────────────
# Token format constants verification
# ────────────────────────────────────────────────────────────────────


class TestTokenFormatConstants:
    """Verify token format constants are correct."""

    def test_legacy_token_constants(self):
        assert TOKEN_PLACEHOLDER_PREFIX == "{{token:"
        assert TOKEN_PLACEHOLDER_SUFFIX == "}}"

    def test_new_token_constants(self):
        assert TOKEN_ENV_PREFIX == "${TOKEN:"
        assert TOKEN_ENV_SUFFIX == "}"

    def test_token_env_format_string_construction(self):
        """Verify the format string ${TOKEN:ns:name} is parsed correctly."""
        value = "${TOKEN:my_namespace:my_name}"
        inner = value[len(TOKEN_ENV_PREFIX) : -len(TOKEN_ENV_SUFFIX)]
        parts = inner.split(":")
        assert parts[0] == "my_namespace"
        assert parts[1] == "my_name"


# ────────────────────────────────────────────────────────────────────
# Edge cases: non-mcpServers config
# ────────────────────────────────────────────────────────────────────


class TestConfigWithoutMcpServers:
    """Test config files without mcpServers key."""

    def test_no_mcp_servers_key_skips_resolution(self, tmp_path):
        """Config without mcpServers key does not error during token resolution."""
        config = {"otherConfig": {"key": "value"}}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        loader = ConfigLoader(str(config_path), [])
        loaded = loader.load()

        assert "otherConfig" in loaded
        assert "mcpServers" not in loaded

    def test_detect_server_types_no_mcp_servers(self, tmp_path):
        """detect_server_types with config that has no mcpServers key."""
        config = {"other": "data"}
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        loader = ConfigLoader(str(config_path), ["some_server"])
        loaded = loader.load()
        loader.detect_server_types(loaded)

        assert loader.http_servers == []
        assert loader.sse_servers == []
        assert loader.stdio_servers == []


# ────────────────────────────────────────────────────────────────────
# Token with extra colon parts
# ────────────────────────────────────────────────────────────────────


class TestTokenExtraColonParts:
    """Test ${TOKEN:ns:name} where name itself contains colons."""

    def test_token_with_extra_parts(self, tmp_path):
        """${TOKEN:ns:name:extra} - only first two parts matter."""
        config = {
            "mcpServers": {"server": {"env": {"KEY": "${TOKEN:ns:name:extra_info}"}}}
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))

        loader = ConfigLoader(str(config_path), ["server"])

        # The key used for lookup should be "ns:name" (first two parts)
        stored_token_json = json.dumps(
            {
                "token_type": "bearer",
                "name": "name",
                "data": {"token": "extra_parts_token"},
            }
        )

        with patch.object(
            loader._token_store, "_retrieve_raw", return_value=stored_token_json
        ):
            loaded = loader.load()

        assert loaded["mcpServers"]["server"]["env"]["KEY"] == "extra_parts_token"
