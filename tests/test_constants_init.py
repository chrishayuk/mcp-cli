# tests/test_constants_init.py
"""Tests for mcp_cli.constants backwards-compatibility re-exports."""


class TestConstantsReExports:
    """Verify that importing mcp_cli.constants re-exports from mcp_cli.config."""

    def test_module_imports_successfully(self):
        """Simply importing the module should cover the import statements."""
        import mcp_cli.constants  # noqa: F401

    # -- Application constants --------------------------------------------------

    def test_app_name_exported(self):
        from mcp_cli.constants import APP_NAME

        assert isinstance(APP_NAME, str) and len(APP_NAME) > 0

    def test_app_version_exported(self):
        from mcp_cli.constants import APP_VERSION

        assert isinstance(APP_VERSION, str)

    def test_namespace_exported(self):
        from mcp_cli.constants import NAMESPACE

        assert isinstance(NAMESPACE, str)

    def test_generic_namespace_exported(self):
        from mcp_cli.constants import GENERIC_NAMESPACE

        assert isinstance(GENERIC_NAMESPACE, str)

    def test_oauth_namespace_exported(self):
        from mcp_cli.constants import OAUTH_NAMESPACE

        assert isinstance(OAUTH_NAMESPACE, str)

    def test_provider_namespace_exported(self):
        from mcp_cli.constants import PROVIDER_NAMESPACE

        assert isinstance(PROVIDER_NAMESPACE, str)

    # -- Timeouts ---------------------------------------------------------------

    def test_timeout_constants(self):
        from mcp_cli.constants import (
            DEFAULT_HTTP_CONNECT_TIMEOUT,
            DEFAULT_HTTP_REQUEST_TIMEOUT,
            DISCOVERY_TIMEOUT,
            REFRESH_TIMEOUT,
            SHUTDOWN_TIMEOUT,
        )

        for val in (
            DEFAULT_HTTP_CONNECT_TIMEOUT,
            DEFAULT_HTTP_REQUEST_TIMEOUT,
            DISCOVERY_TIMEOUT,
            REFRESH_TIMEOUT,
            SHUTDOWN_TIMEOUT,
        ):
            assert isinstance(val, (int, float))

    # -- Platforms --------------------------------------------------------------

    def test_platform_constants(self):
        from mcp_cli.constants import PLATFORM_DARWIN, PLATFORM_LINUX, PLATFORM_WINDOWS

        assert PLATFORM_DARWIN == "darwin"
        assert PLATFORM_LINUX == "linux"
        assert PLATFORM_WINDOWS == "win32"

    # -- Providers --------------------------------------------------------------

    def test_provider_constants(self):
        from mcp_cli.constants import (
            PROVIDER_ANTHROPIC,
            PROVIDER_DEEPSEEK,
            PROVIDER_GROQ,
            PROVIDER_OLLAMA,
            PROVIDER_OPENAI,
            PROVIDER_XAI,
            SUPPORTED_PROVIDERS,
        )

        assert isinstance(SUPPORTED_PROVIDERS, (list, tuple, set, frozenset))
        assert PROVIDER_OPENAI in SUPPORTED_PROVIDERS
        for p in (
            PROVIDER_ANTHROPIC,
            PROVIDER_DEEPSEEK,
            PROVIDER_GROQ,
            PROVIDER_OLLAMA,
            PROVIDER_OPENAI,
            PROVIDER_XAI,
        ):
            assert isinstance(p, str)

    # -- JSON types -------------------------------------------------------------

    def test_json_type_constants(self):
        from mcp_cli.constants import (
            JSON_TYPE_ARRAY,
            JSON_TYPE_BOOLEAN,
            JSON_TYPE_INTEGER,
            JSON_TYPE_NULL,
            JSON_TYPE_NUMBER,
            JSON_TYPE_OBJECT,
            JSON_TYPE_STRING,
            JSON_TYPES,
        )

        assert isinstance(JSON_TYPES, (list, tuple, set, frozenset))
        for jt in (
            JSON_TYPE_ARRAY,
            JSON_TYPE_BOOLEAN,
            JSON_TYPE_INTEGER,
            JSON_TYPE_NULL,
            JSON_TYPE_NUMBER,
            JSON_TYPE_OBJECT,
            JSON_TYPE_STRING,
        ):
            assert isinstance(jt, str)

    # -- Enums ------------------------------------------------------------------

    def test_enum_exports(self):
        from mcp_cli.constants import (
            ConversationAction,
            OutputFormat,
            ServerAction,
            ServerStatus,
            ThemeAction,
            TokenAction,
            TokenNamespace,
            ToolAction,
        )

        # Each should be an enum class
        import enum

        for cls in (
            ConversationAction,
            OutputFormat,
            ServerAction,
            ServerStatus,
            ThemeAction,
            TokenAction,
            TokenNamespace,
            ToolAction,
        ):
            assert issubclass(cls, enum.Enum)

    # -- Environment helpers ----------------------------------------------------

    def test_env_helpers_exported(self):
        from mcp_cli.constants import (
            get_env,
            get_env_bool,
            get_env_float,
            get_env_int,
            get_env_list,
            is_set,
            set_env,
            unset_env,
        )

        assert callable(get_env)
        assert callable(get_env_bool)
        assert callable(get_env_float)
        assert callable(get_env_int)
        assert callable(get_env_list)
        assert callable(is_set)
        assert callable(set_env)
        assert callable(unset_env)

    # -- __all__ ----------------------------------------------------------------

    def test_all_is_defined(self):
        import mcp_cli.constants as mod

        assert hasattr(mod, "__all__")
        assert isinstance(mod.__all__, list)
        assert len(mod.__all__) > 0

    def test_all_entries_are_importable(self):
        import mcp_cli.constants as mod

        for name in mod.__all__:
            assert hasattr(mod, name), f"{name} listed in __all__ but not found"
