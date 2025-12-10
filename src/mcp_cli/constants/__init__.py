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

# Application constants (duplicated from old constants.py to avoid circular import)
# TODO: Consolidate with old constants.py file
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
    # App constants
    "NAMESPACE",
    "OAUTH_NAMESPACE",
    "PROVIDER_NAMESPACE",
    "GENERIC_NAMESPACE",
    "APP_NAME",
    "APP_VERSION",
]
