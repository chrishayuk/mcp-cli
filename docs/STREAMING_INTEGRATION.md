# MCP-CLI Streaming Integration Guide

## Overview

MCP-CLI now features a sophisticated streaming display system that provides real-time feedback during LLM responses with content-aware formatting and progress indication.

## Architecture

### Core Components

1. **`src/mcp_cli/ui/streaming_display.py`**
   - `CompactStreamingDisplay`: Main display class with content detection
   - `StreamingContext`: Context manager for easy integration
   - `tokenize_text()`: Helper for simulating streaming

2. **`src/mcp_cli/chat/streaming_handler.py`**
   - `StreamingResponseHandler`: Manages async streaming from LLM providers
   - Integrates with `StreamingContext` for display
   - Handles both accumulated and delta responses

3. **`src/mcp_cli/chat/conversation.py`**
   - Uses `StreamingResponseHandler` for streaming completions
   - Falls back to regular completions when streaming unavailable

## How It Works

### Streaming Flow

1. User sends a message in chat mode
2. `ConversationProcessor` checks if the LLM client supports streaming
3. If supported, creates a `StreamingResponseHandler` instance
4. Handler creates a `StreamingContext` for display
5. As chunks arrive from the LLM:
   - Content is accumulated
   - Display updates with progress indicator
   - Content type is detected (code, tables, markdown, etc.)
   - Preview lines are shown
6. When streaming completes, final formatted panel is displayed

### Content Detection

The system automatically detects:
- **Code**: Syntax highlighting for code blocks
- **Tables**: Markdown table formatting
- **Queries**: SQL/database query detection
- **JSON**: Structured data formatting
- **Markdown**: Headers and formatting

### Display Modes

- **Response Mode**: For assistant responses
- **Tool Mode**: For tool execution feedback
- **Thinking Mode**: For reasoning/analysis display

## Usage Examples

### Basic Chat

```bash
# With default Ollama provider
mcp-cli --server sqlite

# With OpenAI
source .env  # Load OPENAI_API_KEY
mcp-cli --server sqlite --provider openai --model gpt-4o-mini

# With multiple servers
mcp-cli --server sqlite,echo --provider openai --model gpt-4o-mini
```

### Example Demos

Run the provided examples to see streaming in action:

```bash
# Simple streaming examples
uv run examples/streaming_simple.py

# Complete feature demo
uv run examples/streaming_complete_demo.py

# MCP-CLI integration demo
uv run examples/mcp_cli_working_demo.py

# Full showcase with all content types
uv run examples/streaming_showcase_full.py
```

## Tool Integration

### How Tools Work with Streaming

1. **Initial Response Streaming**: When the LLM decides to use a tool, it streams its reasoning
2. **Tool Detection**: Tool calls are extracted from streaming chunks
3. **Display Finalization**: Streaming display is properly closed before tool execution
4. **Tool Execution**: Tools are invoked with their own progress display
5. **Result Streaming**: After tools complete, the final response streams with results

### Tool Display Modes

- **Thinking Mode**: Used when LLM is deciding which tools to use
- **Tool Mode**: Shows progress during tool execution
- **Response Mode**: Final response after tool results

### Markdown Table Support

The streaming display now properly detects and renders markdown tables:
- Automatic detection of pipe-delimited tables
- Proper formatting with Rich's Markdown renderer
- Works in both streamed responses and tool results

## Features

### Real-time Progress
- Animated spinner during streaming
- Character count and elapsed time
- Dynamic phase messages based on content length

### Content-Aware Display
- Detects content type from early chunks
- Adapts formatting accordingly
- Shows preview of first few lines
- Appropriate styling for different content types

### Clean UI
- Transient display during streaming
- Replaces with final formatted panel
- Fixed height prevents terminal jumping
- Smooth refresh rate (8-10 Hz)

## Provider Support

### Tested Providers
- ✅ **Ollama**: Full streaming support
- ✅ **OpenAI**: Full streaming support (requires API key)
- ✅ **Mock/Test**: Used in demos

### Response Formats
The handler supports both:
- **Accumulated responses**: Each chunk contains the full text so far
- **Delta responses**: Each chunk contains only new text (OpenAI style)

## Configuration

### Environment Variables
```bash
# For OpenAI
OPENAI_API_KEY=your-api-key-here

# For Anthropic
ANTHROPIC_API_KEY=your-api-key-here
```

### Streaming Parameters
- `refresh_per_second`: Display refresh rate (default: 8)
- `transient`: Clear display after completion (default: true)
- `mode`: Display mode (response/tool/thinking)

## Troubleshooting

### Streaming Not Working
1. Check if your provider supports streaming
2. Ensure the model supports streaming
3. Verify API credentials are set correctly

### Display Issues
1. Ensure terminal supports ANSI escape codes
2. Check terminal width is sufficient
3. Try different terminal emulators

### Performance
- Streaming adds minimal overhead
- Network latency is the primary factor
- Display updates are throttled for efficiency

## Development

### Adding New Content Types

Edit `src/mcp_cli/ui/streaming_display.py`:

```python
def detect_content_type(self, text: str):
    # Add your detection logic
    if "your_pattern" in text:
        self.detected_type = "your_type"
```

### Custom Phase Messages

```python
def get_phase_message(self):
    if self.detected_type == "your_type":
        phases = [
            (0, "Starting your process"),
            (100, "Processing"),
            (500, "Finalizing")
        ]
```

## API Reference

### StreamingContext

```python
from mcp_cli.ui.streaming_display import StreamingContext

with StreamingContext(
    console=console,           # Rich console instance
    title="🤖 Assistant",      # Panel title
    mode="response",           # Display mode
    refresh_per_second=8,      # Refresh rate
    transient=True            # Clear after completion
) as ctx:
    for chunk in response_chunks:
        ctx.update(chunk)      # Update display with new content
```

### StreamingResponseHandler

```python
from mcp_cli.chat.streaming_handler import StreamingResponseHandler

handler = StreamingResponseHandler(console)
result = await handler.stream_response(
    client=llm_client,
    messages=conversation_history,
    tools=available_tools
)
```

## Known Limitations

### Tool Streaming
- Tool invocations appear immediately when detected in stream
- Tool results are not streamed (shown as complete blocks)
- Multiple concurrent tools show sequentially

### Markdown Rendering
- Complex nested tables may not render perfectly
- Some markdown extensions not supported
- Code blocks within tables need special handling

## Future Enhancements

- [x] Markdown table rendering support
- [x] Tool execution integration
- [ ] Support for images and multimedia
- [ ] Customizable themes
- [ ] Export streaming sessions
- [ ] Pause/resume streaming
- [ ] Speed controls
- [ ] Token counting display
- [ ] Streaming tool results
- [ ] Progress bars for long operations