# mcp_cli/logging_config.py
"""
Centralized logging configuration for MCP CLI.
"""
import logging
import os
import sys
from typing import Optional

def setup_logging(
    level: str = "WARNING",
    quiet: bool = False,
    verbose: bool = False,
    format_style: str = "simple"
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
            raise ValueError(f'Invalid log level: {level}')
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
        
        # Silence other common noisy loggers
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
    
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