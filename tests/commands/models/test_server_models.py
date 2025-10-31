"""Tests for server command models."""

import pytest
from pydantic import ValidationError
from mcp_cli.commands.models.server import (
    ServerActionParams,
    ServerStatusInfo,
    ServerPerformanceInfo,
)


class TestServerActionParams:
    """Test ServerActionParams model."""

    def test_default_params(self):
        """Test default parameter values."""
        params = ServerActionParams()

        assert params.args == []
        assert not params.detailed
        assert not params.show_capabilities
        assert not params.show_transport
        assert params.output_format == "table"
        assert not params.ping_servers

    def test_custom_params(self):
        """Test custom parameter values."""
        params = ServerActionParams(
            args=["list", "servers"],
            detailed=True,
            show_capabilities=True,
            show_transport=True,
            output_format="json",
            ping_servers=True,
        )

        assert params.args == ["list", "servers"]
        assert params.detailed
        assert params.show_capabilities
        assert params.show_transport
        assert params.output_format == "json"
        assert params.ping_servers

    def test_invalid_output_format(self):
        """Test validation of invalid output format."""
        with pytest.raises(ValidationError) as exc_info:
            ServerActionParams(output_format="invalid")

        error = exc_info.value.errors()[0]
        assert "output_format" in error["loc"]

    def test_valid_output_formats(self):
        """Test valid output formats."""
        for fmt in ["table", "json"]:
            params = ServerActionParams(output_format=fmt)
            assert params.output_format == fmt


class TestServerStatusInfo:
    """Test ServerStatusInfo model."""

    def test_creation(self):
        """Test creating status info."""
        status = ServerStatusInfo(
            icon="✅", status="Connected", reason="Server is online"
        )

        assert status.icon == "✅"
        assert status.status == "Connected"
        assert status.reason == "Server is online"

    def test_required_fields(self):
        """Test that all fields are required."""
        with pytest.raises(ValidationError):
            ServerStatusInfo()

    def test_min_length_validation(self):
        """Test minimum length validation."""
        with pytest.raises(ValidationError) as exc_info:
            ServerStatusInfo(icon="", status="Connected", reason="Online")

        # Should fail on empty icon
        error = exc_info.value.errors()[0]
        assert "icon" in error["loc"]


class TestServerPerformanceInfo:
    """Test ServerPerformanceInfo model."""

    def test_creation(self):
        """Test creating performance info."""
        perf = ServerPerformanceInfo(icon="🚀", latency="25.5ms", ping_ms=25.5)

        assert perf.icon == "🚀"
        assert perf.latency == "25.5ms"
        assert perf.ping_ms == 25.5

    def test_optional_ping_ms(self):
        """Test that ping_ms is optional."""
        perf = ServerPerformanceInfo(icon="🚀", latency="N/A")

        assert perf.icon == "🚀"
        assert perf.latency == "N/A"
        assert perf.ping_ms is None

    def test_negative_ping_validation(self):
        """Test validation of negative ping time."""
        with pytest.raises(ValidationError) as exc_info:
            ServerPerformanceInfo(icon="🚀", latency="25.5ms", ping_ms=-10.0)

        error = exc_info.value.errors()[0]
        assert "ping_ms" in error["loc"]

    def test_zero_ping(self):
        """Test that zero ping is valid."""
        perf = ServerPerformanceInfo(icon="🚀", latency="0ms", ping_ms=0.0)

        assert perf.ping_ms == 0.0

    def test_required_fields(self):
        """Test that icon and latency are required."""
        with pytest.raises(ValidationError):
            ServerPerformanceInfo(ping_ms=25.5)
