# mcp_cli/config/logging.py
"""
Centralized logging configuration for MCP CLI.

Includes secret redaction (always active) and optional file logging
with rotation.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


# ── Secret redaction ─────────────────────────────────────────────────────────

# Patterns that match sensitive values in log messages
_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Bearer tokens: "Bearer eyJ..." or "Bearer sk-..."
    (re.compile(r"(Bearer\s+)\S+", re.IGNORECASE), r"\1[REDACTED]"),
    # API keys: sk-... (OpenAI style)
    (re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"), "[REDACTED_API_KEY]"),
    # Generic "api_key=..." or "api-key:..." values
    (
        re.compile(r"(api[_-]?key\s*[=:]\s*)['\"]?\S+['\"]?", re.IGNORECASE),
        r"\1[REDACTED]",
    ),
    # OAuth access tokens in JSON-ish contexts: "access_token": "..."
    (
        re.compile(r'("access_token"\s*:\s*")[^"]+(")', re.IGNORECASE),
        r"\1[REDACTED]\2",
    ),
    # Authorization headers: "Authorization: Basic xyz" or "Authorization=token"
    (
        re.compile(r"(Authorization\s*[=:]\s*)\S+(?:\s+\S+)?", re.IGNORECASE),
        r"\1[REDACTED]",
    ),
]


class SecretRedactingFilter(logging.Filter):
    """Logging filter that redacts secrets from log messages.

    Catches Bearer tokens, API keys (sk-...), OAuth access tokens,
    and Authorization header values. Always active on all handlers.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern, replacement in _SECRET_PATTERNS:
                record.msg = pattern.sub(replacement, record.msg)
        # Also redact formatted args if they've been interpolated
        if record.args:
            formatted = record.getMessage()
            for pattern, replacement in _SECRET_PATTERNS:
                formatted = pattern.sub(replacement, formatted)
            record.msg = formatted
            record.args = None
        return True


# Module-level singleton so callers can add it to custom handlers
secret_filter = SecretRedactingFilter()


# ── Core setup ───────────────────────────────────────────────────────────────


def setup_logging(
    level: str = "WARNING",
    quiet: bool = False,
    verbose: bool = False,
    format_style: str = "simple",
    log_file: str | None = None,
) -> None:
    """
    Configure centralized logging for MCP CLI and all dependencies.

    Args:
        level: Base logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        quiet: If True, suppress most output except errors
        verbose: If True, enable debug logging
        format_style: "simple", "detailed", or "json"
        log_file: Optional file path for rotating file log (DEBUG level).
                  Expands ~ and creates parent directories automatically.
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

    # Create console handler (with redaction)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    console_handler.addFilter(secret_filter)

    # Configure root logger
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # CRITICAL: Set the root logger's level for ALL loggers
    # This ensures that even if child loggers have their own handlers,
    # they won't emit logs below this level
    logging.root.setLevel(log_level)

    # Optional file logging
    if log_file:
        _add_file_handler(root_logger, log_file)

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


def _add_file_handler(root_logger: logging.Logger, log_file: str) -> None:
    """Add a rotating file handler with JSON format and secret redaction."""
    from mcp_cli.config.defaults import DEFAULT_LOG_MAX_BYTES, DEFAULT_LOG_BACKUP_COUNT

    path = Path(log_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    file_formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
        '"logger": "%(name)s", "line": %(lineno)d, "message": "%(message)s"}'
    )

    file_handler = RotatingFileHandler(
        str(path),
        maxBytes=DEFAULT_LOG_MAX_BYTES,
        backupCount=DEFAULT_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(secret_filter)

    # File handler always logs at DEBUG so root must accept DEBUG too
    if root_logger.level > logging.DEBUG:
        root_logger.setLevel(logging.DEBUG)

    root_logger.addHandler(file_handler)


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
