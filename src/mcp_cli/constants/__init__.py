"""Constants module - all enums, constants, and magic string replacements."""

from importlib.metadata import PackageNotFoundError, version

from mcp_cli.constants.env_vars import (
    EnvVar,
    get_env,
    get_env_bool,
    get_env_float,
    get_env_int,
    get_env_list,
    is_set,
    set_env,
    unset_env,
)
from mcp_cli.constants.enums import (
    ConversationAction,
    ServerAction,
    ServerStatus,
    ThemeAction,
    ToolAction,
    TokenAction,
)
from mcp_cli.constants.json_schema import (
    JSON_TYPE_ARRAY,
    JSON_TYPE_BOOLEAN,
    JSON_TYPE_INTEGER,
    JSON_TYPE_NULL,
    JSON_TYPE_NUMBER,
    JSON_TYPE_OBJECT,
    JSON_TYPE_STRING,
    JSON_TYPES,
)
from mcp_cli.constants.platforms import (
    PLATFORM_DARWIN,
    PLATFORM_LINUX,
    PLATFORM_WINDOWS,
)
from mcp_cli.constants.providers import (
    PROVIDER_ANTHROPIC,
    PROVIDER_DEEPSEEK,
    PROVIDER_GROQ,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    PROVIDER_XAI,
    SUPPORTED_PROVIDERS,
)
from mcp_cli.constants.timeouts import (
    DEFAULT_HTTP_CONNECT_TIMEOUT,
    DEFAULT_HTTP_REQUEST_TIMEOUT,
    DISCOVERY_TIMEOUT,
    REFRESH_TIMEOUT,
    SHUTDOWN_TIMEOUT,
)

# Application constants
NAMESPACE = "mcp-cli"
OAUTH_NAMESPACE = NAMESPACE
PROVIDER_NAMESPACE = "provider"
GENERIC_NAMESPACE = "generic"
APP_NAME = "mcp-cli"

# Get version from package metadata
try:
    APP_VERSION = version("mcp-cli")
except PackageNotFoundError:
    APP_VERSION = "0.0.0-dev"

__all__ = [
    # Environment variables
    "EnvVar",
    "get_env",
    "set_env",
    "unset_env",
    "is_set",
    "get_env_int",
    "get_env_float",
    "get_env_bool",
    "get_env_list",
    # Enums
    "ServerStatus",
    "ConversationAction",
    "TokenAction",
    "ServerAction",
    "ToolAction",
    "ThemeAction",
    # Timeouts
    "DISCOVERY_TIMEOUT",
    "REFRESH_TIMEOUT",
    "SHUTDOWN_TIMEOUT",
    "DEFAULT_HTTP_CONNECT_TIMEOUT",
    "DEFAULT_HTTP_REQUEST_TIMEOUT",
    # Providers
    "PROVIDER_OLLAMA",
    "PROVIDER_OPENAI",
    "PROVIDER_ANTHROPIC",
    "PROVIDER_GROQ",
    "PROVIDER_DEEPSEEK",
    "PROVIDER_XAI",
    "SUPPORTED_PROVIDERS",
    # Platforms
    "PLATFORM_WINDOWS",
    "PLATFORM_DARWIN",
    "PLATFORM_LINUX",
    # JSON Schema types
    "JSON_TYPE_STRING",
    "JSON_TYPE_NUMBER",
    "JSON_TYPE_INTEGER",
    "JSON_TYPE_BOOLEAN",
    "JSON_TYPE_ARRAY",
    "JSON_TYPE_OBJECT",
    "JSON_TYPE_NULL",
    "JSON_TYPES",
    # App constants
    "NAMESPACE",
    "OAUTH_NAMESPACE",
    "PROVIDER_NAMESPACE",
    "GENERIC_NAMESPACE",
    "APP_NAME",
    "APP_VERSION",
]
