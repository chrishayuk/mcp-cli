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

    # Silence noisy third-party loggers unless in debug mode
    if log_level > logging.DEBUG:
        # Silence chuk components unless we need debug info
        logging.getLogger("chuk_tool_processor").setLevel(logging.WARNING)
        logging.getLogger("chuk_mcp").setLevel(logging.WARNING)
        logging.getLogger("chuk_llm").setLevel(logging.WARNING)

        # ENHANCED: More aggressive silencing of MCP server loggers
        mcp_server_loggers = [
            "chuk_mcp_runtime.tools.artifacts",
            "chuk_mcp_runtime.entry",
            "chuk_mcp_runtime.server",
            "chuk_sessions.session_manager",
            "chuk_artifacts.store",
            "chuk_mcp_runtime.tools.session",
            "chuk_mcp_runtime",
            "chuk_sessions",
            "chuk_artifacts",
        ]

        for logger_name in mcp_server_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL)
            # Also try setting propagate to False to prevent any logging
            logger.propagate = False
            # Add a null handler to prevent any output
            if not logger.handlers:
                logger.addHandler(logging.NullHandler())

        # Silence other common noisy loggers
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)

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
    Specifically configure MCP server-related logging.

    Args:
        suppress: If True, suppress INFO/DEBUG from MCP servers. If False, allow all.
    """
    mcp_server_loggers = [
        "chuk_mcp_runtime.tools.artifacts",
        "chuk_mcp_runtime.entry",
        "chuk_mcp_runtime.server",
        "chuk_sessions.session_manager",
        "chuk_artifacts.store",
        "chuk_mcp_runtime.tools.session",
        "chuk_mcp_runtime",
        "chuk_sessions",
        "chuk_artifacts",
    ]

    target_level = logging.CRITICAL if suppress else logging.INFO

    for logger_name in mcp_server_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(target_level)
        if suppress:
            logger.propagate = False
            if not logger.handlers:
                logger.addHandler(logging.NullHandler())


def setup_silent_mcp_environment() -> None:
    """Set up environment variables to silence MCP servers before they start."""
    # Create a Python startup script to suppress logging
    from tempfile import NamedTemporaryFile

    startup_script_content = """
import logging
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress specific loggers immediately on Python startup
for logger_name in [
    "chuk_mcp_runtime", "chuk_mcp_runtime.tools.artifacts", "chuk_mcp_runtime.entry",
    "chuk_mcp_runtime.server", "chuk_sessions", "chuk_sessions.session_manager", 
    "chuk_artifacts", "chuk_artifacts.store", "chuk_mcp_runtime.tools.session"
]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    logger.addHandler(logging.NullHandler())

# Set root logger to ERROR if not configured
root = logging.getLogger()
if not root.handlers:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)-8s %(message)s")
"""

    # Create temporary startup script
    with NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(startup_script_content)
        startup_script_path = f.name

    # Set multiple environment variables that might be respected by MCP servers
    silent_env_vars = {
        # Python startup script - this runs before any other Python code
        "PYTHONSTARTUP": startup_script_path,
        # Python logging configuration
        "PYTHONWARNINGS": "ignore",  # Suppress Python warnings
        "PYTHONIOENCODING": "utf-8",  # Ensure proper encoding
        # General logging levels
        "LOG_LEVEL": "ERROR",
        "LOGGING_LEVEL": "ERROR",
        "MCP_LOG_LEVEL": "ERROR",
        # chuk-specific logging levels
        "CHUK_LOG_LEVEL": "ERROR",
        "CHUK_MCP_LOG_LEVEL": "ERROR",
        "CHUK_MCP_RUNTIME_LOG_LEVEL": "ERROR",
        "CHUK_SESSIONS_LOG_LEVEL": "ERROR",
        "CHUK_ARTIFACTS_LOG_LEVEL": "ERROR",
        # Disable various verbosity flags
        "VERBOSE": "0",
        "DEBUG": "0",
        "QUIET": "1",
        # Python specific settings to reduce noise
        "PYTHONPATH_LOGGING_LEVEL": "ERROR",
        "PYTHON_LOGGING_LEVEL": "ERROR",
        # Try some common environment variables for subprocess logging
        "SUBPROCESS_LOG_LEVEL": "ERROR",
        "CHILD_PROCESS_LOG_LEVEL": "ERROR",
    }

    for key, value in silent_env_vars.items():
        os.environ[key] = value
