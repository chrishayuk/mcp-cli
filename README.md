# MCP CLI - Model Context Protocol Command Line Interface

A powerful, feature-rich command-line interface for interacting with Model Context Protocol servers. This client enables seamless communication with LLMs through integration with the [CHUK Tool Processor](https://github.com/chrishayuk/chuk-tool-processor) and [CHUK-LLM](https://github.com/chrishayuk/chuk-llm), providing tool usage, conversation management, and multiple operational modes.

## 🔄 Architecture Overview

The MCP CLI is built on a modular architecture with clean separation of concerns:

- **[CHUK Tool Processor](https://github.com/chrishayuk/chuk-tool-processor)**: Async-native tool execution and MCP server communication
- **[CHUK-LLM](https://github.com/chrishayuk/chuk-llm)**: Unified LLM provider configuration and client management
- **MCP CLI**: Rich user interface and command orchestration (this project)

## 🌟 Features

### Multiple Operational Modes
- **Chat Mode**: Conversational interface with streaming responses and automated tool usage
- **Interactive Mode**: Command-driven shell interface for direct server operations
- **Command Mode**: Unix-friendly mode for scriptable automation and pipelines
- **Direct Commands**: Run individual commands without entering interactive mode

### Advanced Chat Interface
- **Streaming Responses**: Real-time response generation with live UI updates
- **Concurrent Tool Execution**: Execute multiple tools simultaneously while preserving conversation order
- **Smart Interruption**: Interrupt streaming responses or tool execution with Ctrl+C
- **Performance Metrics**: Response timing, words/second, and execution statistics
- **Rich Formatting**: Markdown rendering, syntax highlighting, and progress indicators

### Comprehensive Provider Support
- **OpenAI**: GPT models (`gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, etc.)
- **Anthropic**: Claude models (`claude-3-opus`, `claude-3-sonnet`, `claude-3-haiku`)
- **Ollama**: Local models (`llama3.2`, `qwen2.5-coder`, `deepseek-coder`, etc.)
- **Custom Providers**: Extensible architecture for additional providers
- **Dynamic Switching**: Change providers and models mid-conversation

### Robust Tool System
- **Automatic Discovery**: Server-provided tools are automatically detected and catalogued
- **Provider Adaptation**: Tool names are automatically sanitized for provider compatibility
- **Concurrent Execution**: Multiple tools can run simultaneously with proper coordination
- **Rich Progress Display**: Real-time progress indicators and execution timing
- **Tool History**: Complete audit trail of all tool executions
- **Streaming Tool Calls**: Support for tools that return streaming data

### Advanced Configuration Management
- **Environment Integration**: API keys and settings via environment variables
- **File-based Config**: YAML and JSON configuration files
- **User Preferences**: Persistent settings for active providers and models
- **Validation & Diagnostics**: Built-in provider health checks and configuration validation

### Enhanced User Experience
- **Cross-Platform Support**: Windows, macOS, and Linux with platform-specific optimizations
- **Rich Console Output**: Colorful, formatted output with automatic fallbacks
- **Command Completion**: Context-aware tab completion for all interfaces
- **Comprehensive Help**: Detailed help system with examples and usage patterns
- **Graceful Error Handling**: User-friendly error messages with troubleshooting hints

## 📋 Prerequisites

- **Python 3.11 or higher**
- **API Keys** (as needed):
  - OpenAI: `OPENAI_API_KEY` environment variable
  - Anthropic: `ANTHROPIC_API_KEY` environment variable
  - Custom providers: Provider-specific configuration
- **Local Services** (as needed):
  - Ollama: Local installation for Ollama models
- **MCP Servers**: Server configuration file (default: `server_config.json`)

## 🚀 Installation

### Using UVX
To install uxx, use the following instructions:

https://docs.astral.sh/uv/getting-started/installation/

Once installed you can test it works using:

```bash
uvx mcp-cli --help
```

or use interactive mode

```bash
uvx mcp-cli interactive
```

### Install from Source

1. **Clone the repository**:
```bash
git clone https://github.com/chrishayuk/mcp-cli
cd mcp-cli  
```

2. **Install the package**:
```bash
pip install -e "."
```

3. **Verify installation**:
```bash
mcp-cli --help
```

### Using UV (Recommended)

UV provides faster dependency resolution and better environment management:

```bash
# Install UV if not already installed
pip install uv

# Install dependencies
uv sync --reinstall

# Run with UV
uv run mcp-cli --help
```

## 🧰 Global Configuration

### Command-line Arguments

Global options available for all modes and commands:

- `--server`: Specify server(s) to connect to (comma-separated)
- `--config-file`: Path to server configuration file (default: `server_config.json`)
- `--provider`: LLM provider (`openai`, `anthropic`, `ollama`, etc.)
- `--model`: Specific model to use (provider-dependent)
- `--disable-filesystem`: Disable filesystem access (default: enabled)
- `--api-base`: Override API endpoint URL
- `--api-key`: Override API key
- `--verbose`: Enable detailed logging
- `--quiet`: Suppress non-essential output

### Environment Variables

```bash
export LLM_PROVIDER=openai              # Default provider
export LLM_MODEL=gpt-4o-mini           # Default model
export OPENAI_API_KEY=sk-...           # OpenAI API key
export ANTHROPIC_API_KEY=sk-ant-...    # Anthropic API key
export MCP_TOOL_TIMEOUT=120            # Tool execution timeout (seconds)
```

## 🌐 Available Modes

### 1. Chat Mode (Default)

Provides a natural language interface with streaming responses and automatic tool usage:

```bash
# Default mode (no subcommand needed)
mcp-cli --server sqlite

# Explicit chat mode
mcp-cli chat --server sqlite

# With specific provider and model
mcp-cli chat --server sqlite --provider anthropic --model claude-3-sonnet

# With custom configuration
mcp-cli chat --server sqlite --provider openai --api-key sk-... --model gpt-4o
```

### 2. Interactive Mode

Command-driven shell interface for direct server operations:

```bash
mcp-cli interactive --server sqlite

# With provider selection
mcp-cli interactive --server sqlite --provider ollama --model llama3.2
```

### 3. Command Mode

Unix-friendly interface for automation and scripting:

```bash
# Process text with LLM
mcp-cli cmd --server sqlite --prompt "Analyze this data" --input data.txt

# Execute tools directly
mcp-cli cmd --server sqlite --tool list_tables --output tables.json

# Pipeline-friendly processing
echo "SELECT * FROM users LIMIT 5" | mcp-cli cmd --server sqlite --tool read_query --input -
```

### 4. Direct Commands

Execute individual commands without entering interactive mode:

```bash
# List available tools
mcp-cli tools --server sqlite

# Show provider configuration
mcp-cli provider list

# Ping servers
mcp-cli ping --server sqlite

# List resources
mcp-cli resources --server sqlite
```

## 🤖 Using Chat Mode

Chat mode provides the most advanced interface with streaming responses and intelligent tool usage.

### Starting Chat Mode

```bash
# Simple startup
mcp-cli --server sqlite

# Multiple servers
mcp-cli --server sqlite,filesystem

# Specific provider configuration
mcp-cli --server sqlite --provider anthropic --model claude-3-opus
```

### Chat Commands (Slash Commands)

#### Provider & Model Management
```bash
/provider                           # Show current configuration
/provider list                      # List all providers
/provider config                    # Show detailed configuration
/provider diagnostic               # Test provider connectivity
/provider set openai api_key sk-... # Configure provider settings
/provider anthropic                # Switch to Anthropic
/provider openai gpt-4o            # Switch provider and model

/model                             # Show current model
/model gpt-4o                      # Switch to specific model
/models                            # List available models
```

#### Tool Management
```bash
/tools                             # List available tools
/tools --all                       # Show detailed tool information
/tools --raw                       # Show raw JSON definitions
/tools call                        # Interactive tool execution

/toolhistory                       # Show tool execution history
/th -n 5                          # Last 5 tool calls
/th 3                             # Details for call #3
/th --json                        # Full history as JSON
```

#### Conversation Management
```bash
/conversation                      # Show conversation history
/ch -n 10                         # Last 10 messages
/ch 5                             # Details for message #5
/ch --json                        # Full history as JSON

/save conversation.json            # Save conversation to file
/compact                          # Summarize conversation
/clear                            # Clear conversation history
/cls                              # Clear screen only
```

#### Session Control
```bash
/verbose                          # Toggle verbose/compact display
/interrupt                        # Stop running operations
/servers                          # List connected servers
/help                            # Show all commands
/help tools                       # Help for specific command
/exit                            # Exit chat mode
```

### Chat Features

#### Streaming Responses
- Real-time text generation with live updates
- Performance metrics (words/second, response time)
- Graceful interruption with Ctrl+C
- Progressive markdown rendering

#### Tool Execution
- Automatic tool discovery and usage
- Concurrent execution with progress indicators
- Verbose and compact display modes
- Complete execution history and timing

#### Provider Integration
- Seamless switching between providers
- Model-specific optimizations
- API key and endpoint management
- Health monitoring and diagnostics

## 🖥️ Using Interactive Mode

Interactive mode provides a command shell for direct server interaction.

### Starting Interactive Mode

```bash
mcp-cli interactive --server sqlite
```

### Interactive Commands

```bash
help                              # Show available commands
exit                              # Exit interactive mode
clear                             # Clear terminal

# Provider management
provider                          # Show current provider
provider list                     # List providers
provider anthropic                # Switch provider

# Tool operations
tools                             # List tools
tools --all                       # Detailed tool info
tools call                        # Interactive tool execution

# Server operations
servers                           # List servers
ping                              # Ping all servers
resources                         # List resources
prompts                           # List prompts
```

## 📄 Using Command Mode

Command mode provides Unix-friendly automation capabilities.

### Command Mode Options

```bash
--input FILE                      # Input file (- for stdin)
--output FILE                     # Output file (- for stdout)
--prompt TEXT                     # Prompt template
--tool TOOL                       # Execute specific tool
--tool-args JSON                  # Tool arguments as JSON
--system-prompt TEXT              # Custom system prompt
--raw                             # Raw output without formatting
--single-turn                     # Disable multi-turn conversation
--max-turns N                     # Maximum conversation turns
```

### Examples

```bash
# Text processing
echo "Analyze this data" | mcp-cli cmd --server sqlite --input - --output analysis.txt

# Tool execution
mcp-cli cmd --server sqlite --tool list_tables --raw

# Complex queries
mcp-cli cmd --server sqlite --tool read_query --tool-args '{"query": "SELECT COUNT(*) FROM users"}'

# Batch processing with GNU Parallel
ls *.txt | parallel mcp-cli cmd --server sqlite --input {} --output {}.summary --prompt "Summarize: {{input}}"
```

## 🔧 Provider Configuration

### Automatic Configuration

The CLI automatically manages provider configurations using the CHUK-LLM library:

```bash
# Configure a provider
mcp-cli provider set openai api_key sk-your-key-here
mcp-cli provider set anthropic api_base https://api.anthropic.com

# Test configuration
mcp-cli provider diagnostic openai

# List available models
mcp-cli provider list
```

### Manual Configuration

The `chuk_llm` library looks for configuration files in a particular order.
First a file specified by the `CHUK_LLM_CONFIG` environment variable.
Then, in the current working directory a file like `chuk_llm.yaml`, `providers.yaml`, `llm_config.yaml` or `config/chuk_llm.yaml`.

But ideally a user wide configuration should be added to `~/.chuk_llm/config.yaml`:

```yaml
openai:
  api_base: https://api.openai.com/v1
  default_model: gpt-4o-mini

anthropic:
  api_base: https://api.anthropic.com
  default_model: claude-3-sonnet

ollama:
  api_base: http://localhost:11434
  default_model: llama3.2
```

API keys are stored securely in `~/.chuk_llm/.env`:

```bash
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## 📂 Server Configuration

Create a `server_config.json` file with your MCP server configurations:

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "python",
      "args": ["-m", "mcp_server.sqlite_server"],
      "env": {
        "DATABASE_PATH": "database.db"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/files"],
      "env": {}
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-brave-api-key"
      }
    }
  }
}
```

## 📈 Advanced Usage Examples

### Multi-Provider Workflow

```bash
# Start with OpenAI
mcp-cli chat --server sqlite --provider openai --model gpt-4o

# In chat, switch to Anthropic for reasoning tasks
> /provider anthropic claude-3-opus

# Switch to Ollama for local processing
> /provider ollama llama3.2

# Compare responses across providers
> /provider openai
> What's the capital of France?
> /provider anthropic  
> What's the capital of France?
```

### Complex Tool Workflows

```bash
# Database analysis workflow
> List all tables in the database
[Tool: list_tables] → products, customers, orders

> Show me the schema for the products table
[Tool: describe_table] → id, name, price, category, stock

> Find the top 10 most expensive products
[Tool: read_query] → SELECT name, price FROM products ORDER BY price DESC LIMIT 10

> Export this data to a CSV file
[Tool: write_file] → Saved to expensive_products.csv
```

### Automation and Scripting

```bash
# Batch data processing
for file in data/*.csv; do
  mcp-cli cmd --server sqlite \
    --tool analyze_data \
    --tool-args "{\"file_path\": \"$file\"}" \
    --output "results/$(basename "$file" .csv)_analysis.json"
done

# Pipeline processing
cat input.txt | \
  mcp-cli cmd --server sqlite --prompt "Extract key entities" --input - | \
  mcp-cli cmd --server sqlite --prompt "Categorize these entities" --input - > output.txt
```

### Performance Monitoring

```bash
# Enable verbose mode for detailed timing
> /verbose

# Monitor tool execution times
> /toolhistory
Tool Call History (15 calls)
#  | Tool        | Arguments                    | Time
1  | list_tables | {}                          | 0.12s
2  | read_query  | {"query": "SELECT..."}      | 0.45s
...

# Check provider performance
> /provider diagnostic
Provider Diagnostics
Provider   | Status      | Response Time | Features
openai     | ✅ Ready    | 234ms        | 📡🔧👁️
anthropic  | ✅ Ready    | 187ms        | 📡🔧
ollama     | ✅ Ready    | 56ms         | 📡🔧
```

## 🔍 Troubleshooting

### Common Issues

1. **"Missing argument 'KWARGS'" error**:
   ```bash
   # Use equals sign format
   mcp-cli chat --server=sqlite --provider=openai
   
   # Or add double dash
   mcp-cli chat -- --server sqlite --provider openai
   ```

2. **Provider not found**:
   ```bash
   mcp-cli provider diagnostic
   mcp-cli provider set <provider> api_key <your-key>
   ```

3. **Tool execution timeout**:
   ```bash
   export MCP_TOOL_TIMEOUT=300  # 5 minutes
   ```

4. **Connection issues**:
   ```bash
   mcp-cli ping --server <server-name>
   mcp-cli servers
   ```

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
mcp-cli --verbose chat --server sqlite
mcp-cli --log-level DEBUG interactive --server sqlite
```

## 🔒 Security Considerations

- **API Keys**: Stored securely in environment variables or protected files
- **File Access**: Filesystem access can be disabled with `--disable-filesystem`
- **Tool Validation**: All tool calls are validated before execution
- **Timeout Protection**: Configurable timeouts prevent hanging operations
- **Server Isolation**: Each server runs in its own process

## 🚀 Performance Features

- **Concurrent Tool Execution**: Multiple tools can run simultaneously
- **Streaming Responses**: Real-time response generation
- **Connection Pooling**: Efficient reuse of client connections
- **Caching**: Tool metadata and provider configurations are cached
- **Async Architecture**: Non-blocking operations throughout

## 📦 Dependencies

Core dependencies are organized into feature groups:

- **cli**: Rich terminal UI, command completion, provider integrations
- **dev**: Development tools, testing utilities, linting
- **chuk-tool-processor**: Core tool execution and MCP communication
- **chuk-llm**: Unified LLM provider management

Install with specific features:
```bash
pip install "mcp-cli[cli]"        # Basic CLI features
pip install "mcp-cli[cli,dev]"    # CLI with development tools
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/chrishayuk/mcp-cli
cd mcp-cli
pip install -e ".[cli,dev]"
pre-commit install
```

### Running Tests

```bash
pytest
pytest --cov=mcp_cli --cov-report=html
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[CHUK Tool Processor](https://github.com/chrishayuk/chuk-tool-processor)** - Async-native tool execution
- **[CHUK-LLM](https://github.com/chrishayuk/chuk-llm)** - Unified LLM provider management
- **[Rich](https://github.com/Textualize/rich)** - Beautiful terminal formatting
- **[Typer](https://typer.tiangolo.com/)** - CLI framework
- **[Prompt Toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit)** - Interactive input

## 🔗 Related Projects

- **[Model Context Protocol](https://modelcontextprotocol.io/)** - Core protocol specification
- **[MCP Servers](https://github.com/modelcontextprotocol/servers)** - Official MCP server implementations
- **[CHUK Tool Processor](https://github.com/chrishayuk/chuk-tool-processor)** - Tool execution engine
- **[CHUK-LLM](https://github.com/chrishayuk/chuk-llm)** - LLM provider abstraction
