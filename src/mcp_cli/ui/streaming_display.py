# src/mcp_cli/ui/streaming_display.py
"""
Compact streaming display components for MCP-CLI.

Provides content-aware streaming display with dynamic phase messages,
content type detection, and smooth progressive rendering.
"""

import time
from typing import Generator, List, Optional
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown


def tokenize_text(text: str) -> Generator[str, None, None]:
    """Generate tokens to simulate LLM streaming."""
    # Simple word-based tokenization
    words = []
    current_word = ""

    for char in text:
        if char in " \n\t":
            if current_word:
                words.append(current_word)
                current_word = ""
            words.append(char)
        else:
            current_word += char

    if current_word:
        words.append(current_word)

    # Yield words in small groups for smoother streaming
    buffer = ""
    for word in words:
        buffer += word
        if len(buffer) > 15 or word == "\n":
            yield buffer
            buffer = ""

    if buffer:
        yield buffer


class CompactStreamingDisplay:
    """Compact streaming display that shows progress for any content type."""

    def __init__(self, title: str = "🤖 Assistant", mode: str = "response"):
        self.title = title
        self.mode = mode  # response, tool, thinking, etc.
        self.first_lines: List[str] = []  # Store the first few lines
        self.current_line = ""
        self.total_chars = 0
        self.total_lines = 0
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_index = 0
        self.preview_captured = False
        self.max_preview_lines = 4
        self.detected_type: Optional[str] = None
        self.content = ""  # Store full content

    def detect_content_type(self, text: str):
        """Detect what type of content is being generated."""
        if self.detected_type:
            return self.detected_type

        # Check for various content indicators
        if "```" in text:
            self.detected_type = "code"
        elif self._is_markdown_table(text):
            self.detected_type = "markdown_table"
        elif "##" in text or "###" in text:
            self.detected_type = "markdown"
        elif (
            "def " in text
            or "function " in text
            or "class " in text
            or "import " in text
        ):
            self.detected_type = "code"
        elif any(x in text for x in ["CREATE TABLE", "SELECT", "INSERT", "UPDATE"]):
            self.detected_type = "query"
        elif any(x in text for x in ["<html>", "<div>", "<span>", "<?xml"]):
            self.detected_type = "markup"
        elif text.strip().startswith("{") or text.strip().startswith("["):
            self.detected_type = "json"
        else:
            self.detected_type = "text"

        return self.detected_type

    def _is_markdown_table(self, text: str) -> bool:
        """Check if text contains a markdown table."""
        lines = text.split("\n")
        pipe_lines = [line for line in lines if "|" in line]

        # Need at least 2 lines with pipes (header + separator)
        if len(pipe_lines) < 2:
            return False

        # Check for separator line with dashes
        for line in pipe_lines:
            if "|" in line and "-" in line:
                # Count pipes and dashes
                if line.count("|") >= 2 and line.count("-") >= 3:
                    return True

        return False

    def get_phase_message(self):
        """Get appropriate phase message based on mode and progress."""
        if self.mode == "tool":
            phases = [
                (0, "Preparing tool"),
                (100, "Executing tool"),
                (500, "Processing results"),
                (1000, "Formatting output"),
                (2000, "Completing execution"),
            ]
        elif self.mode == "thinking":
            phases = [
                (0, "Thinking"),
                (100, "Analyzing request"),
                (300, "Formulating approach"),
                (600, "Organizing thoughts"),
                (1000, "Preparing response"),
            ]
        else:  # response mode
            if self.detected_type == "code":
                phases = [
                    (0, "Starting"),
                    (50, "Writing code"),
                    (500, "Adding implementation"),
                    (1000, "Adding documentation"),
                    (2000, "Finalizing code"),
                ]
            elif self.detected_type in ["table", "markdown_table"]:
                phases = [
                    (0, "Starting"),
                    (50, "Creating table"),
                    (200, "Adding rows"),
                    (500, "Formatting data"),
                    (1000, "Completing table"),
                ]
            elif self.detected_type == "markdown":
                phases = [
                    (0, "Starting"),
                    (50, "Writing content"),
                    (200, "Adding sections"),
                    (500, "Formatting text"),
                    (1000, "Completing document"),
                ]
            elif self.detected_type == "query":
                phases = [
                    (0, "Starting"),
                    (50, "Writing query"),
                    (200, "Adding conditions"),
                    (500, "Optimizing query"),
                    (1000, "Completing query"),
                ]
            else:  # generic text
                phases = [
                    (0, "Starting"),
                    (50, "Generating response"),
                    (200, "Adding details"),
                    (500, "Elaborating"),
                    (1000, "Completing response"),
                ]

        # Find the appropriate phase based on character count
        for min_chars, message in reversed(phases):
            if self.total_chars >= min_chars:
                return message

        return phases[0][1]  # Default to first phase

    def add_content(self, text: str):
        """Process and store content."""
        self.total_chars += len(text)
        self.content += text

        # Detect content type
        if not self.detected_type and len(self.content) > 10:
            self.detect_content_type(self.content)

        # Only capture the first few lines for preview
        if not self.preview_captured:
            for char in text:
                if char == "\n":
                    if self.current_line.strip():
                        self.first_lines.append(self.current_line)
                        self.total_lines += 1
                    self.current_line = ""

                    # Stop capturing after we have enough lines
                    if len(self.first_lines) >= self.max_preview_lines:
                        self.preview_captured = True
                        break
                else:
                    self.current_line += char
                    if len(self.current_line) > 70:
                        self.first_lines.append(self.current_line[:70])
                        self.current_line = self.current_line[70:]

                        if len(self.first_lines) >= self.max_preview_lines:
                            self.preview_captured = True
                            break
        else:
            # Just count lines after preview is captured
            self.total_lines += text.count("\n")

    def get_panel(self, elapsed: float) -> Panel:
        """Get the compact display panel."""
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_frames)
        spinner = self.spinner_frames[self.spinner_index]

        # Build display parts
        display_parts = []

        # Status line with spinner and dynamic phase
        phase = self.get_phase_message()
        status = f"{spinner} {phase}..."
        display_parts.append(Text(status, style="yellow"))

        # Info line with statistics
        if self.detected_type and self.detected_type != "text":
            info = (
                f"⎿  {self.total_chars:,} chars • {elapsed:.1f}s • {self.detected_type}"
            )
        else:
            info = f"⎿  {self.total_chars:,} chars • {elapsed:.1f}s"
        display_parts.append(Text(info, style="dim"))
        display_parts.append(Text(""))  # Empty line

        # Show the first few lines (preview)
        if self.first_lines:
            display_parts.append(Text("   Preview:", style="dim italic"))
            for i, line in enumerate(self.first_lines[:3]):
                line = line.strip()
                if line:
                    # Style based on detected type
                    if self.detected_type == "code" and (
                        line.startswith("def ") or line.startswith("class ")
                    ):
                        style = "green"
                    elif self.detected_type == "markdown" and line.startswith("#"):
                        style = "bold cyan"
                    elif self.detected_type == "table" and "|" in line:
                        style = "blue"
                    else:
                        style = "dim cyan"

                    if len(line) > 55:
                        display_parts.append(Text(f"   {line[:52]}...", style=style))
                    else:
                        display_parts.append(Text(f"   {line}", style=style))

            # Add continuation indicator
            if self.total_chars > 200:
                display_parts.append(
                    Text("   ... generating content", style="dim italic")
                )
        else:
            # Just show a simple cursor while starting
            display_parts.append(Text("   ▌", style="yellow blink"))

        return Panel(
            Group(*display_parts),
            title=self.title,
            border_style="yellow",
            height=10,  # Fixed height for stability
            expand=False,
        )

    def get_final_panel(self, elapsed: float) -> Panel:
        """Get the final formatted panel with full content."""
        # Check if this is primarily a markdown table
        has_markdown_table = self._is_markdown_table(self.content)

        # Determine how to render the content
        should_render_markdown = False

        # For markdown tables with mixed content, we need special handling
        if has_markdown_table:
            # Check if it's JUST a table or has other markdown content
            lines = self.content.split("\n")
            non_table_lines = [
                line
                for line in lines
                if "|" not in line and line.strip() and not line.strip().startswith("-")
            ]

            # If there's significant non-table content with markdown, render as markdown
            has_other_markdown = any(
                "##" in line or "```" in line or "**" in line
                for line in non_table_lines
            )

            if has_other_markdown:
                # Mixed content - try markdown but be ready to fall back
                should_render_markdown = True
            else:
                # Mostly table - use plain text to preserve formatting
                should_render_markdown = False
        elif "```" in self.content:  # Code blocks
            should_render_markdown = True
        elif "##" in self.content or "###" in self.content:  # Headers
            should_render_markdown = True
        elif self.detected_type == "markdown":
            should_render_markdown = True

        # Try to render as markdown if appropriate
        from typing import Union

        content_display: Union[Markdown, Text]
        if should_render_markdown:
            try:
                content_display = Markdown(self.content)
            except Exception:
                # Fallback to text if markdown rendering fails
                content_display = Text(self.content, overflow="fold")
        else:
            # Use text with overflow handling for tables
            content_display = Text(self.content, overflow="fold")

        # Create panel
        return Panel(
            content_display,
            title=self.title,
            subtitle=f"Response time: {elapsed:.2f}s",
            subtitle_align="right",
            border_style="green",
            expand=True,  # Keep normal expansion
        )


class StreamingContext:
    """Context manager for streaming display."""

    def __init__(
        self,
        console,
        title: str = "🤖 Assistant",
        mode: str = "response",
        refresh_per_second: int = 8,
        transient: bool = True,
    ):
        self.console = console
        self.display = CompactStreamingDisplay(title=title, mode=mode)
        self.refresh_per_second = refresh_per_second
        self.transient = transient
        self.live = None
        self.start_time = time.time()

    def __enter__(self):
        """Start the live display."""
        self.live = Live(
            self.display.get_panel(0),
            console=self.console,
            refresh_per_second=self.refresh_per_second,
            transient=self.transient,
        )
        self.live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up the live display and show final panel."""
        if self.live:
            self.live.__exit__(exc_type, exc_val, exc_tb)

            # Show final panel if we have content
            if self.display.content:
                elapsed = time.time() - self.start_time
                final_panel = self.display.get_final_panel(elapsed)
                self.console.print(final_panel)

    def update(self, content: str):
        """Update the streaming display with new content."""
        self.display.add_content(content)
        elapsed = time.time() - self.start_time
        if self.live:
            self.live.update(self.display.get_panel(elapsed))  # type: ignore[unreachable]

    @property
    def content(self):
        """Get the accumulated content."""
        return self.display.content
