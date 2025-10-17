# mcp_cli/logging_config.py
"""
Centralized logging configuration for MCP CLI.
"""

import logging
import os
import sys


def setup_logging(
    level: str = "WARNING",
    quiet: bool = False,
    verbose: bool = False,
    format_style: str = "simple",
) -> None:
    """
    Configure centralized logging for MCP CLI and all dependencies.

    Args:
        level: Base logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        quiet: If True, suppress most output except errors
        verbose: If True, enable debug logging
        format_style: "simple", "detailed", or "json"
    """
    # Determine effective log level
    if quiet:
        log_level = logging.ERROR
    elif verbose:
        log_level = logging.DEBUG
    else:
        # Parse string level
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {level}")
        log_level = numeric_level

    # Set environment variable that chuk components respect
    os.environ["CHUK_LOG_LEVEL"] = logging.getLevelName(log_level)

    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure format
    if format_style == "json":
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"message": "%(message)s", "logger": "%(name)s"}'
        )
    elif format_style == "detailed":
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s"
        )
    else:  # simple
        formatter = logging.Formatter("%(levelname)-8s %(message)s")

    # Create console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    # Configure root logger
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # CRITICAL: Set the root logger's level for ALL loggers
    # This ensures that even if child loggers have their own handlers,
    # they won't emit logs below this level
    logging.root.setLevel(log_level)

    # Silence noisy third-party loggers unless in debug mode
    if log_level > logging.DEBUG:
        # Silence common third-party library loggers
        third_party_loggers = [
            "urllib3",
            "requests",
            "httpx",
            "asyncio",
        ]

        for logger_name in third_party_loggers:
            logging.getLogger(logger_name).setLevel(logging.ERROR)

        # Aggressively silence chuk framework components
        # These components set up their own handlers, so we need to remove them
        chuk_loggers = [
            "chuk_tool_processor",
            "chuk_mcp",
            "chuk_mcp_runtime",
            "chuk_sessions",
            "chuk_artifacts",
            "chuk_llm",
        ]

        for logger_name in chuk_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.ERROR)
            logger.propagate = False
            # Remove all existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            # Add null handler to prevent any output
            logger.addHandler(logging.NullHandler())

    # Set mcp_cli loggers to appropriate level
    logging.getLogger("mcp_cli").setLevel(log_level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name."""
    return logging.getLogger(f"mcp_cli.{name}")


# Convenience function for common use case
def setup_quiet_logging() -> None:
    """Set up minimal logging for production use."""
    setup_logging(quiet=True)


def setup_verbose_logging() -> None:
    """Set up detailed logging for debugging."""
    setup_logging(verbose=True, format_style="detailed")


def setup_clean_logging() -> None:
    """Set up clean logging that suppresses MCP server noise but shows warnings."""
    setup_logging(level="WARNING", quiet=False, verbose=False)


def configure_mcp_server_logging(suppress: bool = True) -> None:
    """
    Configure logging for chuk framework components.

    Args:
        suppress: If True, suppress INFO/DEBUG logs. If False, allow all.
    """
    # Generic chuk framework loggers
    framework_loggers = [
        "chuk_mcp_runtime",
        "chuk_sessions",
        "chuk_artifacts",
    ]

    target_level = logging.CRITICAL if suppress else logging.INFO

    for logger_name in framework_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(target_level)
        if suppress:
            logger.propagate = False
            if not logger.handlers:
                logger.addHandler(logging.NullHandler())


def setup_silent_mcp_environment() -> None:
    """Set up environment variables to silence subprocesses before they start."""
    # Create a Python startup script to suppress logging
    from tempfile import NamedTemporaryFile

    startup_script_content = """
import logging
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Set root logger to ERROR before any other code runs
# This affects all loggers by default
root = logging.getLogger()
if not root.handlers:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)-8s %(message)s")
root.setLevel(logging.ERROR)
"""

    # Create temporary startup script
    with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(startup_script_content)
        startup_script_path = f.name

    # Set generic environment variables for quiet logging
    silent_env_vars = {
        # Python startup script - this runs before any other Python code
        "PYTHONSTARTUP": startup_script_path,
        # Python logging configuration
        "PYTHONWARNINGS": "ignore",
        "PYTHONIOENCODING": "utf-8",
        # General logging levels
        "LOG_LEVEL": "ERROR",
        "LOGGING_LEVEL": "ERROR",
        # Disable various verbosity flags
        "VERBOSE": "0",
        "DEBUG": "0",
        "QUIET": "1",
    }

    for key, value in silent_env_vars.items():
        os.environ[key] = value
