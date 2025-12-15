# tests/chat/state/test_cache.py
"""Tests for ResultCache."""

import pytest

from mcp_cli.chat.state.cache import ResultCache


class TestResultCache:
    """Tests for ResultCache."""

    @pytest.fixture
    def cache(self):
        return ResultCache()

    def test_put_and_get(self, cache):
        """Test caching and retrieving a result."""
        cache.put("sqrt", {"x": 18}, 4.2426)
        cached = cache.get("sqrt", {"x": 18})
        assert cached is not None
        assert cached.result == 4.2426

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        cached = cache.get("sqrt", {"x": 18})
        assert cached is None

    def test_duplicate_increments_count(self, cache):
        """Test that duplicate calls increment count."""
        cache.put("sqrt", {"x": 18}, 4.2426)
        cached1 = cache.get("sqrt", {"x": 18})
        assert cached1.call_count == 2  # put + get

        cached2 = cache.get("sqrt", {"x": 18})
        assert cached2.call_count == 3

    def test_duplicate_count_tracking(self, cache):
        """Test duplicate count is tracked."""
        cache.put("sqrt", {"x": 18}, 4.2426)
        assert cache.duplicate_count == 0

        cache.get("sqrt", {"x": 18})
        assert cache.duplicate_count == 1

    def test_store_variable(self, cache):
        """Test storing named variables."""
        var = cache.store_variable("sigma", 5.5, units="units/day")
        assert var.name == "sigma"
        assert var.value == 5.5
        assert var.units == "units/day"

    def test_get_variable(self, cache):
        """Test retrieving stored variables."""
        cache.store_variable("sigma", 5.5)
        var = cache.get_variable("sigma")
        assert var is not None
        assert var.value == 5.5

    def test_get_variable_missing(self, cache):
        """Test getting non-existent variable returns None."""
        var = cache.get_variable("nonexistent")
        assert var is None

    def test_format_state_empty(self, cache):
        """Test format_state with empty cache."""
        state = cache.format_state()
        assert state == ""

    def test_format_state_with_results(self, cache):
        """Test format_state with cached results."""
        cache.put("sqrt", {"x": 18}, 4.2426)
        cache.put("multiply", {"a": 2, "b": 3}, 6)
        state = cache.format_state()
        assert "sqrt" in state or "multiply" in state

    def test_reset(self, cache):
        """Test reset clears all state."""
        cache.put("sqrt", {"x": 18}, 4.2426)
        cache.store_variable("sigma", 5.5)
        cache.reset()

        assert cache.get("sqrt", {"x": 18}) is None
        assert cache.get_variable("sigma") is None
        assert cache.duplicate_count == 0

    def test_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = ResultCache(max_size=3)
        cache.put("tool1", {"x": 1}, 1)
        cache.put("tool2", {"x": 2}, 2)
        cache.put("tool3", {"x": 3}, 3)
        cache.put("tool4", {"x": 4}, 4)

        # First entry should be evicted
        assert cache.get("tool1", {"x": 1}) is None
        assert cache.get("tool4", {"x": 4}) is not None

    def test_get_stats(self, cache):
        """Test get_stats returns correct info."""
        cache.put("sqrt", {"x": 18}, 4.2426)
        cache.store_variable("sigma", 5.5)

        stats = cache.get_stats()
        assert stats["total_cached"] == 1
        assert stats["total_variables"] == 1

    def test_format_duplicate_message(self, cache):
        """Test format_duplicate_message."""
        cache.put("sqrt", {"x": 18}, 4.2426)
        msg = cache.format_duplicate_message("sqrt", {"x": 18})
        assert "sqrt" in msg
        assert "4.2426" in msg or "cached" in msg.lower()
