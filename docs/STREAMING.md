# MCP-CLI Streaming Display

## Overview

MCP-CLI now features an enhanced compact streaming display system that provides content-aware, real-time feedback during LLM response generation.

## Features

### Content-Aware Display
- Automatically detects content type (code, markdown, tables, queries, JSON, etc.)
- Adapts phase messages based on detected content
- Provides syntax-appropriate preview styling

### Dynamic Progress Indicators
- Smooth spinner animation
- Real-time character and time statistics
- Progressive phase messages that update based on response length
- Content type indicator in status line

### Modes
- **Response Mode**: For assistant responses with content-type detection
- **Tool Mode**: For tool execution with appropriate status messages
- **Thinking Mode**: For analysis and reasoning steps

## Usage

### Simple Example

```python
from mcp_cli.ui.streaming_display import StreamingContext
from rich.console import Console

console = Console()

# Use as a context manager
with StreamingContext(console=console, title="🤖 Assistant") as ctx:
    for chunk in response_chunks:
        ctx.update(chunk)
```

### With Custom Mode

```python
# Tool execution mode
with StreamingContext(
    console=console,
    title="⚙️ Database Query",
    mode="tool",
    refresh_per_second=8,
    transient=True
) as ctx:
    for chunk in tool_output:
        ctx.update(chunk)
```

## Integration with MCP-CLI

The streaming display is integrated into:
- `StreamingHandler`: Handles async streaming from LLM providers
- `UIManager`: Manages overall UI state during streaming
- `ConversationProcessor`: Orchestrates streaming during chat

## Architecture

### Core Components

1. **CompactStreamingDisplay**: Main display class
   - Content type detection
   - Preview generation
   - Progress tracking
   - Panel rendering

2. **StreamingContext**: Context manager wrapper
   - Manages `rich.Live` display
   - Handles cleanup and final panel display
   - Provides simple update interface

3. **Helper Functions**:
   - `tokenize_text()`: Converts text to streamable tokens

### Display Flow

1. Content arrives in chunks
2. Display detects content type from early chunks
3. Preview is captured from first few lines
4. Progress panel updates with each chunk
5. Final panel renders with full formatted content

## Examples

See the `examples/` directory for complete demos:
- `streaming_simple.py`: Basic usage examples
- `mcp_cli_streaming_demo_v2.py`: Comprehensive feature showcase
- `mcp_cli_clean_demo_new.py`: Full integration example

## Performance

- Optimized refresh rate (8 Hz) for smooth updates without flicker
- Transient display that cleanly replaces with final panel
- Minimal overhead with efficient content accumulation
- Fixed panel height prevents terminal jumping