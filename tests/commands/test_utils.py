"""Tests for commands utility functions."""

from mcp_cli.commands.utils import (
    format_capabilities,
    format_performance,
    get_server_icon,
    human_size,
)


class TestFormatCapabilities:
    """Tests for format_capabilities function."""

    def test_no_capabilities(self):
        """Test with empty capabilities."""
        assert format_capabilities({}) == "None"

    def test_tools_only(self):
        """Test with only tools capability."""
        caps = {"tools": True}
        assert format_capabilities(caps) == "Tools"

    def test_prompts_only(self):
        """Test with only prompts capability."""
        caps = {"prompts": True}
        assert format_capabilities(caps) == "Prompts"

    def test_resources_only(self):
        """Test with only resources capability."""
        caps = {"resources": True}
        assert format_capabilities(caps) == "Resources"

    def test_all_standard_capabilities(self):
        """Test with all standard capabilities."""
        caps = {"tools": True, "prompts": True, "resources": True}
        result = format_capabilities(caps)
        assert "Tools" in result
        assert "Prompts" in result
        assert "Resources" in result

    def test_experimental_events(self):
        """Test with experimental events capability."""
        caps = {"experimental": {"events": True}}
        assert format_capabilities(caps) == "Events*"

    def test_experimental_streaming(self):
        """Test with experimental streaming capability."""
        caps = {"experimental": {"streaming": True}}
        assert format_capabilities(caps) == "Streaming*"

    def test_mixed_standard_and_experimental(self):
        """Test with both standard and experimental capabilities."""
        caps = {
            "tools": True,
            "prompts": True,
            "experimental": {"events": True, "streaming": True},
        }
        result = format_capabilities(caps)
        assert "Tools" in result
        assert "Prompts" in result
        assert "Events*" in result
        assert "Streaming*" in result

    def test_false_capabilities(self):
        """Test with explicitly false capabilities."""
        caps = {
            "tools": False,
            "prompts": False,
            "resources": False,
            "experimental": {"events": False, "streaming": False},
        }
        assert format_capabilities(caps) == "None"


class TestFormatPerformance:
    """Tests for format_performance function."""

    def test_none_ping(self):
        """Test with None ping value."""
        icon, text = format_performance(None)
        assert icon == "❓"
        assert text == "Unknown"

    def test_very_fast_ping(self):
        """Test with very fast ping (< 10ms)."""
        icon, text = format_performance(5.5)
        assert icon == "🚀"
        assert "5.5ms" in text

    def test_fast_ping(self):
        """Test with fast ping (10-50ms)."""
        icon, text = format_performance(25.5)
        assert icon == "✅"
        assert "25.5ms" in text

    def test_moderate_ping(self):
        """Test with moderate ping (50-100ms)."""
        icon, text = format_performance(75.0)
        assert icon == "⚠️"
        assert "75.0ms" in text

    def test_slow_ping(self):
        """Test with slow ping (>= 100ms)."""
        icon, text = format_performance(150.0)
        assert icon == "🔴"
        assert "150.0ms" in text

    def test_edge_case_10ms(self):
        """Test edge case at 10ms boundary."""
        icon, text = format_performance(10.0)
        assert icon == "✅"
        assert "10.0ms" in text

    def test_edge_case_50ms(self):
        """Test edge case at 50ms boundary."""
        icon, text = format_performance(50.0)
        assert icon == "⚠️"
        assert "50.0ms" in text

    def test_edge_case_100ms(self):
        """Test edge case at 100ms boundary."""
        icon, text = format_performance(100.0)
        assert icon == "🔴"
        assert "100.0ms" in text


class TestGetServerIcon:
    """Tests for get_server_icon function."""

    def test_full_featured_server(self):
        """Test server with both resources and prompts."""
        caps = {"resources": True, "prompts": True}
        assert get_server_icon(caps, 5) == "🎯"

    def test_resource_capable_server(self):
        """Test server with only resources."""
        caps = {"resources": True, "prompts": False}
        assert get_server_icon(caps, 5) == "📁"

    def test_prompt_capable_server(self):
        """Test server with only prompts."""
        caps = {"resources": False, "prompts": True}
        assert get_server_icon(caps, 5) == "💬"

    def test_tool_heavy_server(self):
        """Test server with many tools (> 15)."""
        caps = {}
        assert get_server_icon(caps, 20) == "🔧"

    def test_basic_tool_server(self):
        """Test server with some tools (1-15)."""
        caps = {}
        assert get_server_icon(caps, 5) == "⚙️"

    def test_minimal_server(self):
        """Test server with no tools."""
        caps = {}
        assert get_server_icon(caps, 0) == "📦"

    def test_edge_case_16_tools(self):
        """Test edge case with exactly 16 tools."""
        caps = {}
        assert get_server_icon(caps, 16) == "🔧"

    def test_edge_case_15_tools(self):
        """Test edge case with exactly 15 tools."""
        caps = {}
        assert get_server_icon(caps, 15) == "⚙️"

    def test_resources_overrides_tool_count(self):
        """Test that resources capability overrides tool count."""
        caps = {"resources": True}
        assert get_server_icon(caps, 20) == "📁"

    def test_prompts_overrides_tool_count(self):
        """Test that prompts capability overrides tool count."""
        caps = {"prompts": True}
        assert get_server_icon(caps, 20) == "💬"


class TestHumanSize:
    """Tests for human_size function."""

    def test_none_size(self):
        """Test with None size."""
        assert human_size(None) == "-"

    def test_negative_size(self):
        """Test with negative size."""
        assert human_size(-100) == "-"

    def test_bytes(self):
        """Test size in bytes."""
        assert human_size(500) == "500 B"

    def test_kilobytes(self):
        """Test size in kilobytes."""
        assert human_size(1024) == "1 KB"
        assert human_size(2048) == "2 KB"

    def test_megabytes(self):
        """Test size in megabytes."""
        assert human_size(1048576) == "1 MB"
        assert human_size(2097152) == "2 MB"

    def test_gigabytes(self):
        """Test size in gigabytes."""
        assert human_size(1073741824) == "1 GB"
        assert human_size(2147483648) == "2 GB"

    def test_terabytes(self):
        """Test size in terabytes."""
        assert human_size(1099511627776) == "1.0 TB"
        assert human_size(2199023255552) == "2.0 TB"

    def test_edge_case_1023_bytes(self):
        """Test edge case just below 1 KB."""
        assert human_size(1023) == "1023 B"

    def test_fractional_kb(self):
        """Test fractional kilobytes."""
        assert human_size(1536) == "2 KB"  # 1.5 KB rounds to 2

    def test_fractional_mb(self):
        """Test fractional megabytes."""
        assert human_size(1572864) == "2 MB"  # 1.5 MB rounds to 2

    def test_zero_size(self):
        """Test with zero size."""
        assert human_size(0) == "0 B"


class TestModuleExports:
    """Test module exports."""

    def test_all_exports(self):
        """Test that __all__ contains expected functions."""
        from mcp_cli.commands import utils

        assert hasattr(utils, "__all__")
        assert "format_capabilities" in utils.__all__
        assert "format_performance" in utils.__all__
        assert "get_server_icon" in utils.__all__
        assert "human_size" in utils.__all__
        assert len(utils.__all__) == 4
