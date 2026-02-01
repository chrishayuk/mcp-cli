# tests/config/test_logging.py
"""Tests for logging configuration."""

from __future__ import annotations

import logging
import os

import pytest

from mcp_cli.config.logging import (
    configure_mcp_server_logging,
    get_logger,
    setup_clean_logging,
    setup_logging,
    setup_quiet_logging,
    setup_silent_mcp_environment,
    setup_verbose_logging,
)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_default(self) -> None:
        """Test setup_logging with default parameters."""
        setup_logging()
        root = logging.getLogger()
        assert root.level == logging.WARNING

    def test_setup_logging_quiet_mode(self) -> None:
        """Test setup_logging with quiet=True."""
        setup_logging(quiet=True)
        root = logging.getLogger()
        assert root.level == logging.ERROR

    def test_setup_logging_verbose_mode(self) -> None:
        """Test setup_logging with verbose=True."""
        setup_logging(verbose=True)
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_setup_logging_invalid_level(self) -> None:
        """Test setup_logging with invalid level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging(level="INVALID_LEVEL")

    def test_setup_logging_json_format(self) -> None:
        """Test setup_logging with JSON format."""
        setup_logging(format_style="json")
        root = logging.getLogger()
        assert len(root.handlers) > 0
        # Check formatter contains JSON structure
        handler = root.handlers[0]
        assert handler.formatter is not None
        assert "timestamp" in handler.formatter._fmt

    def test_setup_logging_detailed_format(self) -> None:
        """Test setup_logging with detailed format."""
        setup_logging(format_style="detailed")
        root = logging.getLogger()
        assert len(root.handlers) > 0
        handler = root.handlers[0]
        assert handler.formatter is not None
        assert "%(asctime)s" in handler.formatter._fmt

    def test_setup_logging_simple_format(self) -> None:
        """Test setup_logging with simple format (default)."""
        setup_logging(format_style="simple")
        root = logging.getLogger()
        assert len(root.handlers) > 0
        handler = root.handlers[0]
        assert handler.formatter is not None
        assert "%(levelname)" in handler.formatter._fmt

    def test_setup_logging_sets_chuk_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that setup_logging sets CHUK_LOG_LEVEL env var."""
        monkeypatch.delenv("CHUK_LOG_LEVEL", raising=False)
        setup_logging(level="INFO")
        assert os.environ.get("CHUK_LOG_LEVEL") == "INFO"


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_logger(self) -> None:
        """Test get_logger returns a logger with mcp_cli prefix."""
        logger = get_logger("test_module")
        assert logger.name == "mcp_cli.test_module"

    def test_get_logger_different_names(self) -> None:
        """Test get_logger returns different loggers for different names."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        assert logger1.name != logger2.name


class TestConvenienceFunctions:
    """Tests for convenience logging functions."""

    def test_setup_quiet_logging(self) -> None:
        """Test setup_quiet_logging sets ERROR level."""
        setup_quiet_logging()
        root = logging.getLogger()
        assert root.level == logging.ERROR

    def test_setup_verbose_logging(self) -> None:
        """Test setup_verbose_logging sets DEBUG level with detailed format."""
        setup_verbose_logging()
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_setup_clean_logging(self) -> None:
        """Test setup_clean_logging sets WARNING level."""
        setup_clean_logging()
        root = logging.getLogger()
        assert root.level == logging.WARNING


class TestConfigureMCPServerLogging:
    """Tests for configure_mcp_server_logging function."""

    def test_configure_mcp_server_logging_suppress(self) -> None:
        """Test configure_mcp_server_logging with suppress=True."""
        configure_mcp_server_logging(suppress=True)
        # Check that framework loggers are set to CRITICAL
        logger = logging.getLogger("chuk_mcp_runtime")
        assert logger.level == logging.CRITICAL
        assert logger.propagate is False

    def test_configure_mcp_server_logging_no_suppress(self) -> None:
        """Test configure_mcp_server_logging with suppress=False."""
        configure_mcp_server_logging(suppress=False)
        # Check that framework loggers are set to INFO
        logger = logging.getLogger("chuk_mcp_runtime")
        assert logger.level == logging.INFO


class TestSetupSilentMCPEnvironment:
    """Tests for setup_silent_mcp_environment function."""

    def test_setup_silent_mcp_environment_sets_env_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that setup_silent_mcp_environment sets required env vars."""
        # Clear any existing env vars
        for var in ["LOG_LEVEL", "VERBOSE", "DEBUG", "QUIET"]:
            monkeypatch.delenv(var, raising=False)

        setup_silent_mcp_environment()

        # Check environment variables are set
        assert os.environ.get("LOG_LEVEL") == "ERROR"
        assert os.environ.get("VERBOSE") == "0"
        assert os.environ.get("DEBUG") == "0"
        assert os.environ.get("QUIET") == "1"

    def test_setup_silent_mcp_environment_creates_startup_script(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that setup_silent_mcp_environment creates PYTHONSTARTUP script."""
        monkeypatch.delenv("PYTHONSTARTUP", raising=False)

        setup_silent_mcp_environment()

        # Check PYTHONSTARTUP is set and file exists
        startup_path = os.environ.get("PYTHONSTARTUP")
        assert startup_path is not None
        assert os.path.exists(startup_path)
