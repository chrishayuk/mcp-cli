"""Tests for streaming display Pydantic models."""

import pytest
import time
from mcp_cli.display.models import (
    ContentType,
    StreamingChunk,
    StreamingPhase,
    StreamingState,
    DisplayUpdate,
)


class TestStreamingChunk:
    """Tests for StreamingChunk model."""

    def test_from_raw_chunk_chuk_llm_format(self):
        """Test parsing chuk-llm format chunks."""
        raw = {
            "response": "Hello world",
            "tool_calls": [{"id": "1", "name": "test"}],
            "reasoning_content": "Thinking...",
        }

        chunk = StreamingChunk.from_raw_chunk(raw)

        assert chunk.content == "Hello world"
        assert chunk.tool_calls == [{"id": "1", "name": "test"}]
        assert chunk.reasoning_content == "Thinking..."
        assert chunk.finish_reason is None

    def test_from_raw_chunk_openai_format(self):
        """Test parsing OpenAI format chunks."""
        raw = {
            "choices": [
                {
                    "delta": {"content": "Hello"},
                    "finish_reason": "stop",
                }
            ]
        }

        chunk = StreamingChunk.from_raw_chunk(raw)

        assert chunk.content == "Hello"
        assert chunk.finish_reason == "stop"

    def test_from_raw_chunk_deepseek_reasoning(self):
        """Test parsing DeepSeek reasoning content from delta."""
        raw = {
            "choices": [
                {
                    "delta": {"reasoning_content": "Let me analyze this..."},
                    "finish_reason": None,
                }
            ]
        }

        chunk = StreamingChunk.from_raw_chunk(raw)

        assert chunk.reasoning_content == "Let me analyze this..."
        assert chunk.content is None

    def test_from_raw_chunk_deepseek_mixed(self):
        """Test parsing DeepSeek chunk with both reasoning and content."""
        raw = {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "Thinking...",
                        "content": "Answer",
                    },
                }
            ]
        }

        chunk = StreamingChunk.from_raw_chunk(raw)

        assert chunk.reasoning_content == "Thinking..."
        assert chunk.content == "Answer"

    def test_from_raw_chunk_simple_format(self):
        """Test parsing simple content format."""
        raw = {"content": "Test content", "finish_reason": "length"}

        chunk = StreamingChunk.from_raw_chunk(raw)

        assert chunk.content == "Test content"
        assert chunk.finish_reason == "length"

    def test_from_raw_chunk_empty(self):
        """Test parsing empty chunk."""
        chunk = StreamingChunk.from_raw_chunk({})

        assert chunk.content is None
        assert chunk.tool_calls is None
        assert chunk.reasoning_content is None

    def test_chunk_is_immutable(self):
        """Test that chunks are frozen."""
        chunk = StreamingChunk(content="test")

        with pytest.raises(Exception):  # Pydantic ValidationError
            chunk.content = "modified"


class TestStreamingState:
    """Tests for StreamingState model."""

    def test_initial_state(self):
        """Test initial state creation."""
        state = StreamingState()

        assert state.accumulated_content == ""
        assert state.chunks_received == 0
        assert state.phase == StreamingPhase.INITIALIZING
        assert state.detected_type == ContentType.UNKNOWN
        assert not state.interrupted
        assert state.is_active
        assert not state.is_complete

    def test_add_text_chunk(self):
        """Test adding text chunk."""
        state = StreamingState()
        chunk = StreamingChunk(content="Hello ")

        state.add_chunk(chunk)

        assert state.accumulated_content == "Hello "
        assert state.chunks_received == 1
        assert state.phase == StreamingPhase.RECEIVING

    def test_add_multiple_chunks(self):
        """Test adding multiple chunks."""
        state = StreamingState()

        state.add_chunk(StreamingChunk(content="Hello "))
        state.add_chunk(StreamingChunk(content="world"))
        state.add_chunk(StreamingChunk(content="!"))

        assert state.accumulated_content == "Hello world!"
        assert state.chunks_received == 3

    def test_detect_code_content_type(self):
        """Test code content type detection."""
        state = StreamingState()

        state.add_chunk(StreamingChunk(content="def hello():"))

        assert state.detected_type == ContentType.CODE

    def test_detect_markdown_content_type(self):
        """Test markdown content type detection."""
        state = StreamingState()

        state.add_chunk(StreamingChunk(content="## Heading"))

        assert state.detected_type == ContentType.MARKDOWN

    def test_detect_markdown_table_content_type(self):
        """Test markdown table detection."""
        state = StreamingState()

        table = "| Col 1 | Col 2 |\n|-------|-------|"
        state.add_chunk(StreamingChunk(content=table))

        assert state.detected_type == ContentType.MARKDOWN_TABLE

    def test_detect_json_content_type(self):
        """Test JSON content type detection."""
        state = StreamingState()

        state.add_chunk(StreamingChunk(content='{"key": "value"}'))

        assert state.detected_type == ContentType.JSON

    def test_detect_sql_content_type(self):
        """Test SQL content type detection."""
        state = StreamingState()

        state.add_chunk(StreamingChunk(content="SELECT * FROM users"))

        assert state.detected_type == ContentType.SQL

    def test_content_type_locks_after_detection(self):
        """Test that content type doesn't change after detection."""
        state = StreamingState()

        state.add_chunk(StreamingChunk(content="def hello():"))
        assert state.detected_type == ContentType.CODE

        # Add markdown content - type should stay CODE
        state.add_chunk(StreamingChunk(content="## Heading"))
        assert state.detected_type == ContentType.CODE

    def test_reasoning_content(self):
        """Test reasoning content accumulation."""
        state = StreamingState()

        chunk = StreamingChunk(content="Answer", reasoning_content="Let me think...")
        state.add_chunk(chunk)

        assert state.reasoning_content == "Let me think..."

    def test_finish_reason(self):
        """Test finish reason capture."""
        state = StreamingState()

        chunk = StreamingChunk(content="Done", finish_reason="stop")
        state.add_chunk(chunk)

        assert state.finish_reason == "stop"

    def test_complete_normal(self):
        """Test normal completion."""
        state = StreamingState()
        state.add_chunk(StreamingChunk(content="Test"))

        state.complete()

        assert state.phase == StreamingPhase.COMPLETED
        assert not state.interrupted
        assert state.is_complete
        assert not state.is_active
        assert state.end_time is not None

    def test_complete_interrupted(self):
        """Test interrupted completion."""
        state = StreamingState()
        state.add_chunk(StreamingChunk(content="Test"))

        state.complete(interrupted=True)

        assert state.phase == StreamingPhase.INTERRUPTED
        assert state.interrupted
        assert state.is_complete

    def test_mark_error(self):
        """Test error state."""
        state = StreamingState()

        state.mark_error()

        assert state.phase == StreamingPhase.ERROR
        assert state.is_complete
        assert state.end_time is not None

    def test_elapsed_time(self):
        """Test elapsed time calculation."""
        state = StreamingState()

        # Add small delay
        time.sleep(0.1)

        assert state.elapsed_time >= 0.1

    def test_elapsed_time_after_completion(self):
        """Test elapsed time uses end_time after completion."""
        state = StreamingState()
        time.sleep(0.1)
        state.complete()

        # Elapsed time should be frozen
        elapsed1 = state.elapsed_time
        time.sleep(0.1)
        elapsed2 = state.elapsed_time

        # Should be nearly identical (end_time is set)
        assert abs(elapsed1 - elapsed2) < 0.01

    def test_content_length(self):
        """Test content length property."""
        state = StreamingState()

        state.add_chunk(StreamingChunk(content="Hello"))
        assert state.content_length == 5

        state.add_chunk(StreamingChunk(content=" world"))
        assert state.content_length == 11


class TestDisplayUpdate:
    """Tests for DisplayUpdate model."""

    def test_from_state(self):
        """Test creating update from state."""
        state = StreamingState()
        state.add_chunk(StreamingChunk(content="def hello():"))

        update = DisplayUpdate.from_state(state)

        assert update.content == "def hello():"
        assert update.content_type == ContentType.CODE
        assert update.phase == StreamingPhase.RECEIVING
        assert update.chunks_received == 1
        assert update.show_spinner is True

    def test_from_completed_state(self):
        """Test update from completed state."""
        state = StreamingState()
        state.add_chunk(StreamingChunk(content="Done"))
        state.complete()

        update = DisplayUpdate.from_state(state)

        assert update.phase == StreamingPhase.COMPLETED
        assert update.show_spinner is False

    def test_update_with_reasoning(self):
        """Test update includes reasoning content."""
        state = StreamingState()
        chunk = StreamingChunk(content="Answer", reasoning_content="Thinking...")
        state.add_chunk(chunk)

        update = DisplayUpdate.from_state(state)

        assert update.reasoning_content == "Thinking..."

    def test_update_is_immutable(self):
        """Test that updates are frozen."""
        state = StreamingState()
        update = DisplayUpdate.from_state(state)

        with pytest.raises(Exception):  # Pydantic ValidationError
            update.content = "modified"


class TestContentTypeDetection:
    """Tests for content type detection edge cases."""

    def test_code_with_backticks(self):
        """Test code blocks with backticks."""
        state = StreamingState()
        state.add_chunk(StreamingChunk(content="Here's code:\n```python\n"))

        assert state.detected_type == ContentType.CODE

    def test_javascript_function(self):
        """Test JavaScript function detection."""
        state = StreamingState()
        state.add_chunk(StreamingChunk(content="function test() {"))

        assert state.detected_type == ContentType.CODE

    def test_html_markup(self):
        """Test HTML detection."""
        state = StreamingState()
        state.add_chunk(StreamingChunk(content="<html><body>"))

        assert state.detected_type == ContentType.MARKUP

    def test_json_array(self):
        """Test JSON array detection."""
        state = StreamingState()
        state.add_chunk(StreamingChunk(content="[1, 2, 3]"))

        assert state.detected_type == ContentType.JSON

    def test_plain_text(self):
        """Test plain text detection."""
        state = StreamingState()
        state.add_chunk(StreamingChunk(content="Just some regular text."))

        assert state.detected_type == ContentType.TEXT


class TestStreamingBufferCaps:
    """Tests for StreamingState buffer caps and stall detection."""

    def test_content_cap_enforced(self):
        """Content stops accumulating when cap is reached."""
        state = StreamingState(max_accumulated_chars=100)

        # Add content up to the cap
        state.add_chunk(StreamingChunk(content="A" * 80))
        assert len(state.accumulated_content) == 80
        assert not state.content_capped

        # This chunk would exceed the cap
        state.add_chunk(StreamingChunk(content="B" * 30))
        assert state.content_capped
        assert "buffer limit reached" in state.accumulated_content
        # Content should NOT have the B's appended
        assert "B" * 30 not in state.accumulated_content

    def test_content_after_cap_ignored(self):
        """Chunks after cap are silently dropped."""
        state = StreamingState(max_accumulated_chars=50)
        state.add_chunk(StreamingChunk(content="A" * 60))
        assert state.content_capped

        # More content should be ignored
        state.add_chunk(StreamingChunk(content="SHOULD_NOT_APPEAR"))
        assert "SHOULD_NOT_APPEAR" not in state.accumulated_content

    def test_content_cap_zero_disables(self):
        """max_accumulated_chars=0 means unlimited."""
        state = StreamingState(max_accumulated_chars=0)
        big_content = "X" * 10_000_000
        state.add_chunk(StreamingChunk(content=big_content))
        assert len(state.accumulated_content) == 10_000_000
        assert not state.content_capped

    def test_reasoning_cap_enforced(self):
        """Reasoning content respects the cap."""
        state = StreamingState(max_accumulated_chars=100)
        state.add_chunk(StreamingChunk(reasoning_content="R" * 80))
        assert len(state.reasoning_content) == 80

        # This would exceed cap
        state.add_chunk(StreamingChunk(reasoning_content="S" * 30))
        # Reasoning should NOT have grown
        assert "S" * 30 not in state.reasoning_content

    def test_reasoning_cap_zero_disables(self):
        """max_accumulated_chars=0 means unlimited for reasoning too."""
        state = StreamingState(max_accumulated_chars=0)
        state.add_chunk(StreamingChunk(reasoning_content="R" * 100_000))
        assert len(state.reasoning_content) == 100_000

    def test_stall_detection_max_chunks(self):
        """finish_reason set when max_chunks reached."""
        state = StreamingState(max_chunks=3)
        state.add_chunk(StreamingChunk(content="a"))
        state.add_chunk(StreamingChunk(content="b"))
        assert state.finish_reason is None

        state.add_chunk(StreamingChunk(content="c"))
        assert state.finish_reason == "max_chunks_reached"

    def test_stall_detection_does_not_override_finish_reason(self):
        """If finish_reason already set, stall detection preserves it."""
        state = StreamingState(max_chunks=2)
        state.add_chunk(StreamingChunk(content="a", finish_reason="stop"))
        state.add_chunk(StreamingChunk(content="b"))
        # finish_reason should remain "stop", not overwritten
        assert state.finish_reason == "stop"

    def test_stall_detection_zero_disables(self):
        """max_chunks=0 means no stall detection."""
        state = StreamingState(max_chunks=0)
        for _ in range(1000):
            state.add_chunk(StreamingChunk(content="x"))
        assert state.finish_reason is None

    def test_default_caps(self):
        """Default values match expected constants."""
        state = StreamingState()
        assert state.max_accumulated_chars == 1_048_576
        assert state.max_chunks == 50_000
        assert not state.content_capped

    def test_chunks_still_counted_when_capped(self):
        """chunks_received keeps incrementing even after cap."""
        state = StreamingState(max_accumulated_chars=10)
        state.add_chunk(StreamingChunk(content="A" * 20))
        assert state.content_capped
        state.add_chunk(StreamingChunk(content="B"))
        state.add_chunk(StreamingChunk(content="C"))
        assert state.chunks_received == 3
