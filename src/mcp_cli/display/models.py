"""Pydantic models for streaming display state.

This module defines type-safe models for all streaming display operations,
eliminating dictionary-based state management and magic strings.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ChunkField(str, Enum):
    """Field names for raw streaming chunks from various providers."""

    # Top-level fields
    RESPONSE = "response"
    CONTENT = "content"
    TEXT = "text"
    CHOICES = "choices"
    TOOL_CALLS = "tool_calls"
    REASONING_CONTENT = "reasoning_content"
    FINISH_REASON = "finish_reason"

    # Delta fields (OpenAI/DeepSeek format)
    DELTA = "delta"


class ContentType(str, Enum):
    """Detected content type for appropriate rendering."""

    TEXT = "text"
    CODE = "code"
    MARKDOWN = "markdown"
    MARKDOWN_TABLE = "markdown_table"
    JSON = "json"
    SQL = "sql"
    MARKUP = "markup"
    UNKNOWN = "unknown"


class StreamingPhase(str, Enum):
    """Current phase of streaming operation."""

    INITIALIZING = "initializing"
    RECEIVING = "receiving"
    PROCESSING = "processing"
    COMPLETING = "completing"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    ERROR = "error"


class StreamingChunk(BaseModel):
    """A single chunk received from the streaming LLM response.

    This normalizes various chunk formats from different providers
    into a consistent structure.
    """

    content: str | None = Field(default=None, description="Text content in this chunk")
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Tool call data in this chunk"
    )
    reasoning_content: str | None = Field(
        default=None, description="Reasoning/thinking content (DeepSeek Reasoner)"
    )
    finish_reason: str | None = Field(
        default=None, description="Reason for completion if final chunk"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional provider-specific metadata"
    )

    model_config = {"frozen": True}

    @classmethod
    def from_raw_chunk(cls, chunk: dict[str, Any]) -> StreamingChunk:
        """Parse a raw chunk from various LLM providers into normalized format.

        Supports multiple formats:
        - chuk-llm format: {'response': str, 'tool_calls': list, ...}
        - OpenAI format: {'choices': [{'delta': {...}}], ...}
        - Direct format: {'content': str, ...}
        """
        # Extract content from various formats
        content = None
        if ChunkField.RESPONSE in chunk:
            content = chunk[ChunkField.RESPONSE]
        elif ChunkField.CONTENT in chunk:
            content = chunk[ChunkField.CONTENT]
        elif ChunkField.TEXT in chunk:
            content = chunk[ChunkField.TEXT]
        elif ChunkField.DELTA in chunk:
            delta = chunk[ChunkField.DELTA]
            if isinstance(delta, dict):
                content = delta.get(ChunkField.CONTENT)
        elif ChunkField.CHOICES in chunk and chunk[ChunkField.CHOICES]:
            choice = chunk[ChunkField.CHOICES][0]
            if ChunkField.DELTA in choice:
                content = choice[ChunkField.DELTA].get(ChunkField.CONTENT)

        # Extract tool calls
        tool_calls = chunk.get(ChunkField.TOOL_CALLS)

        # Extract reasoning content (DeepSeek sends in delta)
        reasoning = None
        if ChunkField.REASONING_CONTENT in chunk:
            reasoning = chunk[ChunkField.REASONING_CONTENT]
        elif ChunkField.CHOICES in chunk and chunk[ChunkField.CHOICES]:
            choice = chunk[ChunkField.CHOICES][0]
            if ChunkField.DELTA in choice:
                delta = choice[ChunkField.DELTA]
                if isinstance(delta, dict):
                    reasoning = delta.get(ChunkField.REASONING_CONTENT)

        # Extract finish reason
        finish_reason = chunk.get(ChunkField.FINISH_REASON)
        if (
            not finish_reason
            and ChunkField.CHOICES in chunk
            and chunk[ChunkField.CHOICES]
        ):
            finish_reason = chunk[ChunkField.CHOICES][0].get(ChunkField.FINISH_REASON)

        return cls(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning,
            finish_reason=finish_reason,
            metadata=chunk,
        )


class StreamingState(BaseModel):
    """Complete state of an active streaming operation.

    This replaces scattered state variables with a single, type-safe model.
    """

    # Content accumulation
    accumulated_content: str = Field(
        default="", description="All content received so far"
    )
    reasoning_content: str = Field(
        default="", description="Reasoning/thinking content accumulated"
    )

    # Chunk tracking
    chunks_received: int = Field(default=0, description="Total chunks processed")
    last_chunk_time: float = Field(
        default_factory=time.time, description="Timestamp of last chunk"
    )

    # Content detection
    detected_type: ContentType = Field(
        default=ContentType.UNKNOWN, description="Detected content type"
    )

    # Phase tracking
    phase: StreamingPhase = Field(
        default=StreamingPhase.INITIALIZING, description="Current streaming phase"
    )

    # Timing
    start_time: float = Field(
        default_factory=time.time, description="When streaming started"
    )
    end_time: float | None = Field(default=None, description="When streaming completed")

    # Completion
    finish_reason: str | None = Field(default=None, description="Why streaming ended")
    interrupted: bool = Field(default=False, description="Whether user interrupted")

    model_config = {"frozen": False}

    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time since streaming started."""
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    @property
    def content_length(self) -> int:
        """Total characters in accumulated content."""
        return len(self.accumulated_content)

    @property
    def is_active(self) -> bool:
        """Whether streaming is currently active."""
        return self.phase in {
            StreamingPhase.INITIALIZING,
            StreamingPhase.RECEIVING,
            StreamingPhase.PROCESSING,
        }

    @property
    def is_complete(self) -> bool:
        """Whether streaming has finished (successfully or not)."""
        return self.phase in {
            StreamingPhase.COMPLETED,
            StreamingPhase.INTERRUPTED,
            StreamingPhase.ERROR,
        }

    def add_chunk(self, chunk: StreamingChunk) -> None:
        """Process a new chunk and update state."""
        self.chunks_received += 1
        self.last_chunk_time = time.time()

        if chunk.content:
            self.accumulated_content += chunk.content
            self._update_content_type(chunk.content)

        if chunk.reasoning_content:
            # Accumulate reasoning content (DeepSeek streams it in chunks)
            # Add space if we're appending to existing content to avoid word concatenation
            if self.reasoning_content and not self.reasoning_content.endswith(" "):
                # Check if the new chunk starts with punctuation or space
                if chunk.reasoning_content and not chunk.reasoning_content[0] in " .,!?;:":
                    self.reasoning_content += " "
            self.reasoning_content += chunk.reasoning_content

        if chunk.finish_reason:
            self.finish_reason = chunk.finish_reason

        # Update phase if receiving
        if self.phase == StreamingPhase.INITIALIZING:
            self.phase = StreamingPhase.RECEIVING

    def _update_content_type(self, new_content: str) -> None:
        """Detect and update content type based on accumulated content."""
        if self.detected_type != ContentType.UNKNOWN:
            return  # Already detected

        full_content = self.accumulated_content

        # Detection logic
        if "```" in full_content:
            self.detected_type = ContentType.CODE
        elif self._is_markdown_table(full_content):
            self.detected_type = ContentType.MARKDOWN_TABLE
        elif "##" in full_content or "###" in full_content:
            self.detected_type = ContentType.MARKDOWN
        elif any(
            x in full_content
            for x in ["def ", "function ", "class ", "import ", "const ", "let "]
        ):
            self.detected_type = ContentType.CODE
        elif any(
            x in full_content.upper()
            for x in ["CREATE TABLE", "SELECT", "INSERT", "UPDATE"]
        ):
            self.detected_type = ContentType.SQL
        elif any(x in full_content for x in ["<html>", "<div>", "<span>", "<?xml"]):
            self.detected_type = ContentType.MARKUP
        elif full_content.strip().startswith(("{", "[")):
            self.detected_type = ContentType.JSON
        else:
            self.detected_type = ContentType.TEXT

    def _is_markdown_table(self, text: str) -> bool:
        """Check if text contains a markdown table."""
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "|" in line and i + 1 < len(lines):
                next_line = lines[i + 1]
                # Check for separator line like |---|---|
                if "|" in next_line and "-" in next_line:
                    return True
        return False

    def complete(self, interrupted: bool = False) -> None:
        """Mark streaming as complete."""
        self.end_time = time.time()
        self.interrupted = interrupted
        if interrupted:
            self.phase = StreamingPhase.INTERRUPTED
        else:
            self.phase = StreamingPhase.COMPLETED

    def mark_error(self) -> None:
        """Mark streaming as errored."""
        self.end_time = time.time()
        self.phase = StreamingPhase.ERROR


class DisplayUpdate(BaseModel):
    """An update to send to the display system.

    This represents a single display update operation.
    """

    content: str = Field(description="Content to display")
    content_type: ContentType = Field(
        default=ContentType.TEXT, description="Type of content"
    )
    phase: StreamingPhase = Field(description="Current streaming phase")
    chunks_received: int = Field(description="Chunks received so far")
    elapsed_time: float = Field(description="Elapsed time in seconds")
    reasoning_content: str | None = Field(
        default=None, description="Reasoning content to display"
    )
    show_spinner: bool = Field(
        default=True, description="Whether to show spinner animation"
    )

    model_config = {"frozen": True}

    @classmethod
    def from_state(cls, state: StreamingState) -> DisplayUpdate:
        """Create a display update from current streaming state."""
        return cls(
            content=state.accumulated_content,
            content_type=state.detected_type,
            phase=state.phase,
            chunks_received=state.chunks_received,
            elapsed_time=state.elapsed_time,
            reasoning_content=state.reasoning_content
            if state.reasoning_content
            else None,
            show_spinner=state.is_active,
        )
