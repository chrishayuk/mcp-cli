# environment.py
import os
import sys

# Default environment variables to inherit
DEFAULT_INHERITED_ENV_VARS = (
    ["HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER"]
    if sys.platform != "win32"
    else [
        "APPDATA",
        "HOMEDRIVE",
        "HOMEPATH",
        "LOCALAPPDATA",
        "PATH",
        "PROCESSOR_ARCHITECTURE",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "USERNAME",
        "USERPROFILE",
    ]
)


def get_default_environment() -> dict[str, str]:
    """
    Retrieve a dictionary of default environment variables to inherit.
    """

    # get the current environment
    env = {
        key: value
        for key in DEFAULT_INHERITED_ENV_VARS
        if (value := os.environ.get(key)) and not value.startswith("()")
    }

    # return the dictionary
    return env
