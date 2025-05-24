# MCP CLI - Model Context Protocol Command Line Interface

A powerful, feature-rich command-line interface for interacting with Model Context Protocol servers. This client enables seamless communication with LLMs through integration with the [CHUK-MCP protocol library](https://github.com/chrishayuk/chuk-mcp) which is a pyodide compatible pure python protocol implementation of MCP, supporting tool usage, conversation management, and multiple operational modes.

## 🔄 Protocol Implementation

The core protocol implementation has been moved to a separate package at:
**[https://github.com/chrishayuk/chuk-mcp](https://github.com/chrishayuk/chuk-mcp)**

This CLI is built on top of the protocol library, focusing on providing a rich user experience while the protocol library handles the communication layer.

## 🌟 Features

- **Multiple Operational Modes**:
  - **Chat Mode**: Conversational interface with direct LLM interaction and automated tool usage
  - **Interactive Mode**: Command-driven interface for direct server operations
  - **Command Mode**: Unix-friendly mode for scriptable automation and pipelines
  - **Direct Commands**: Run individual commands without entering interactive mode

- **Multi-Provider Support**:
  - OpenAI integration (`gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo`, etc.)
  - Ollama integration (`llama3.2`, `qwen2.5-coder`, etc.)
  - Anthropic integration (`claude-3-opus`, `claude-3-sonnet`, etc.)
  - Extensible architecture for additional providers

- **Provider and Model Management**:
  - Configure multiple LLM providers (API keys, endpoints, default models)
  - Switch between providers and models during sessions
  - Command-line arguments for provider/model selection
  - Interactive commands for provider configuration

- **Robust Tool System**:
  - Automatic discovery of server-provided tools
  - Server-aware tool execution
  - Tool call history tracking and analysis
  - Support for complex, multi-step tool chains

- **Advanced Conversation Management**:
  - Complete conversation history tracking
  - Filtering and viewing specific message ranges
  - JSON export capabilities for debugging or analysis
  - Conversation compaction for reduced token usage

- **Rich User Experience**:
  - Command completion with context-aware suggestions
  - Colorful, formatted console output
  - Progress indicators for long-running operations
  - Detailed help and documentation

- **Resilient Resource Management**:
  - Proper cleanup of asyncio resources
  - Graceful error handling
  - Clean terminal restoration
  - Support for multiple simultaneous server connections

## 📋 Prerequisites

- Python 3.11 or higher
- For OpenAI: Valid API key in `OPENAI_API_KEY` environment variable
- For Anthropic: Valid API key in `ANTHROPIC_API_KEY` environment variable
- For Ollama: Local Ollama installation
- Server configuration file (default: `server_config.json`)
- [CHUK-MCP](https://github.com/chrishayuk/chuk-mcp) protocol library

## 🚀 Installation

### Install from Source

1. Clone the repository:

```bash
git clone https://github.com/chrishayuk/mcp-cli
cd mcp-cli  
```

2. Install the package with development dependencies:

```bash
pip install -e ".[cli,dev]"
```

3. Run the CLI:

```bash
mcp-cli --help
```

### Using UV (Alternative Installation)

If you prefer using UV for dependency management:

```bash
# Install UV if not already installed
pip install uv

# Install dependencies
uv sync --reinstall

# Run using UV
uv run mcp-cli --help
```

## 🧰 Global Command-line Arguments

Global options available for all modes and commands:

- `--server`: Specify the server(s) to connect to (comma-separated for multiple)
- `--config-file`: Path to server configuration file (default: `server_config.json`)
- `--provider`: LLM provider to use (`openai`, `anthropic`, `ollama`, default: `openai`)
- `--model`: Specific model to use (provider-dependent defaults)
- `--disable-filesystem`: Disable filesystem access (default: true)

### CLI Argument Format Issue

You might encounter a "Missing argument 'KWARGS'" error when running various commands. This is due to how the CLI parser is configured. To resolve this, use one of these approaches:

1. Use the equals sign format for all arguments:
   ```bash
   mcp-cli tools call --server=sqlite
   mcp-cli chat --server=sqlite --provider=ollama --model=llama3.2
   ```

2. Add a double-dash (`--`) after the command and before arguments:
   ```bash
   mcp-cli tools call -- --server sqlite
   mcp-cli chat -- --server sqlite --provider ollama --model llama3.2
   ```

3. When using uv and multiple extra parameters, follow the 2nd step but add an empty string at the end:
   ```
   uv run mcp-cli chat -- --server sqlite --provider ollama --model llama3.2 ""
   ```

These format issues apply to all commands (chat, interactive, tools, etc.) and are due to how the argument parser interprets positional vs. named arguments.

## 🌐 Available Modes

### 1. Chat Mode

Chat mode provides a natural language interface for interacting with LLMs, where the model can automatically use available tools:

```bash
# Default (makes chat the default when no other command is specified)
uv run mcp-cli

# Explicit chat mode
uv run mcp-cli chat --server sqlite

# With specific provider and model
uv run mcp-cli chat --server sqlite --provider openai --model gpt-4o
```

### 2. Interactive Mode

Interactive mode provides a command-driven shell interface for direct server operations:

```bash
uv run mcp-cli interactive --server sqlite
```

### 3. Command Mode (Cmd)

Command mode provides a Unix-friendly interface for automation and pipeline integration:

```bash
uv run mcp-cli cmd --server sqlite [options]
```

### 4. Direct Commands

Run individual commands without entering an interactive mode:

```bash
# List available tools
uv run mcp-cli tools list {} --server sqlite

# Call a specific tool
uv run mcp-cli tools call {} --server sqlite
```

## 🤖 Using Chat Mode

Chat mode provides a conversational interface with the LLM, automatically using available tools when needed.

### Starting Chat Mode

```bash
# Default with {} for KWARGS
uv run mcp-cli --server sqlite

# Explicit chat mode with {}
uv run mcp-cli chat --server sqlite

# With specific provider and model
uv run mcp-cli chat --server sqlite --provider openai --model gpt-4o
```

```bash
# Note: Be careful with the command syntax
# Correct format without any KWARGS parameter
uv run mcp-cli chat --server sqlite --provider ollama --model llama3.2

# Or if you encounter the "Missing argument 'KWARGS'" error, try:
uv run mcp-cli chat --server=sqlite --provider=ollama --model=llama3.2
```

### Chat Commands

In chat mode, use these slash commands:

#### General Commands
- `/help`: Show available commands
- `/help <command>`: Show detailed help for a specific command
- `/quickhelp` or `/qh`: Display a quick reference of common commands
- `exit` or `quit`: Exit chat mode

#### Provider and Model Commands
- `/provider` or `/p`: Display or manage LLM providers
  - `/provider`: Show current provider and model
  - `/provider list`: List all configured providers
  - `/provider config`: Show detailed provider configuration
  - `/provider set <name> <key> <value>`: Set a provider configuration value
  - `/provider <name>`: Switch to a different provider
- `/model` or `/m`: Display or change the current model
  - `/model`: Show current model
  - `/model <name>`: Switch to a different model

#### Tool Commands
- `/tools`: Display all available tools with their server information
  - `/tools --all`: Show detailed tool information including parameters
  - `/tools --raw`: Show raw tool definitions
- `/toolhistory` or `/th`: Show history of tool calls in the current session
  - `/th <N>`: Show details for a specific tool call
  - `/th -n 5`: Show only the last 5 tool calls
  - `/th --json`: Show tool calls in JSON format

#### Conversation Commands
- `/conversation` or `/ch`: Show the conversation history
  - `/ch <N>`: Show a specific message from history
  - `/ch -n 5`: Show only the last 5 messages
  - `/ch <N> --json`: Show a specific message in JSON format
  - `/ch --json`: View the entire conversation history in raw JSON format
- `/save <filename>`: Save conversation history to a JSON file
- `/compact`: Condense conversation history into a summary

#### Display Commands
- `/cls`: Clear the screen while keeping conversation history
- `/clear`: Clear both the screen and conversation history
- `/verbose` or `/v`: Toggle between verbose and compact tool display modes

#### Control Commands
- `/interrupt`, `/stop`, or `/cancel`: Interrupt running tool execution
- `/servers`: List connected servers and their status

## 🖥️ Using Interactive Mode

Interactive mode provides a command-driven shell interface for direct server interaction.

### Starting Interactive Mode

```bash
# Using {} to satisfy KWARGS requirement
mcp-cli interactive {} --server sqlite
```

### Interactive Commands

In interactive mode, use these commands:

- `help`: Show available commands
- `exit` or `quit` or `q`: Exit interactive mode
- `clear` or `cls`: Clear the terminal screen
- `servers` or `srv`: List connected servers with their status
- `provider` or `p`: Manage LLM providers
  - `provider`: Show current provider and model
  - `provider list`: List all configured providers
  - `provider config`: Show detailed provider configuration
  - `provider set <name> <key> <value>`: Set a provider configuration value
  - `provider <name>`: Switch to a different provider
- `model` or `m`: Display or change the current model
  - `model`: Show current model
  - `model <name>`: Switch to a different model
- `tools` or `t`: List available tools or call one interactively
  - `tools --all`: Show detailed tool information
  - `tools --raw`: Show raw JSON definitions
  - `tools call`: Launch the interactive tool-call UI
- `resources` or `res`: List available resources from all servers
- `prompts` or `p`: List available prompts from all servers
- `ping`: Ping connected servers (optionally filter by index/name)

## 📄 Using Command Mode (Cmd)

Command mode provides a Unix-friendly interface for automation and pipeline integration.

### Starting Command Mode

```bash
# Using {} to satisfy KWARGS requirement
mcp-cli cmd {} --server sqlite [options]
```

### Command Mode Options

- `--input`: Input file path (use `-` for stdin)
- `--output`: Output file path (use `-` for stdout, default)
- `--prompt`: Prompt template (use `{{input}}` as placeholder for input)
- `--raw`: Output raw text without formatting
- `--tool`: Directly call a specific tool
- `--tool-args`: JSON arguments for tool call
- `--system-prompt`: Custom system prompt
- `--verbose`: Enable verbose logging
- `--provider`: Specify LLM provider
- `--model`: Specify model to use

### Command Mode Examples

Process content with LLM:

```bash
# Summarize a document (with {} for KWARGS)
uv run mcp-cli cmd --server sqlite --input document.md --prompt "Summarize this: {{input}}" --output summary.md

# Process stdin and output to stdout
cat document.md | mcp-cli cmd {} --server sqlite --input - --prompt "Extract key points: {{input}}"

# Use a specific provider and model
uv run mcp-cli cmd {} --server sqlite --input document.md --prompt "Summarize: {{input}}" --provider anthropic --model claude-3-opus
```

Call tools directly:

```bash
# List database tables
uv run mcp-cli cmd {} --server sqlite --tool list_tables --raw

# Run a SQL query
uv run mcp-cli cmd {} --server sqlite --tool read_query --tool-args '{"query": "SELECT COUNT(*) FROM users"}'
```

Batch processing:

```bash
# Process multiple files with GNU Parallel
ls *.md | parallel mcp-cli cmd --server sqlite --input {} --output {}.summary.md --prompt "Summarize: {{input}}"
```

## 🔧 Direct CLI Commands

Run individual commands without entering interactive mode:

### Provider Commands

```bash
# Show current provider configuration
mcp-cli provider show

# List all configured providers
mcp-cli provider list

# Show detailed provider configuration
mcp-cli provider config

# Set a configuration value
mcp-cli provider set <provider_name> <key> <value>
# Example: mcp-cli provider set openai api_key "sk-..."
```

### Tools Commands

```bash
# List all tools (using {} to satisfy KWARGS requirement)
uv run mcp-cli tools list {} --server sqlite

# Show detailed tool information
uv run mcp-cli tools list {} --server sqlite --all

# Show raw tool definitions
uv run mcp-cli tools list {} --server sqlite --raw

# Call a specific tool interactively
uv run mcp-cli tools call {} --server sqlite
```

### Resources and Prompts Commands

```bash
# List available resources
uv run mcp-cli resources list {} --server sqlite

# List available prompts
uv run mcp-cli prompts list {} --server sqlite
```

### Server Commands

```bash
# Ping all servers
uv run mcp-cli ping {} --server sqlite

# Ping specific server(s)
uv run mcp-cli ping {} --server sqlite,another-server
```

## 📂 Server Configuration

Create a `server_config.json` file with your server configurations:

```json
{
  "mcpServers": {
    "sqlite": {
      "command": "python",
      "args": ["-m", "mcp_server.sqlite_server"],
      "env": {
        "DATABASE_PATH": "your_database.db"
      }
    },
    "another-server": {
      "command": "python",
      "args": ["-m", "another_server_module"],
      "env": {}
    }
  }
}
```

## 🔐 Provider Configuration

Provider configurations are stored with these key settings:

- `api_key`: API key for authentication
- `api_base`: Base URL for API requests
- `default_model`: Default model to use with this provider
- Other provider-specific settings

### Environment Variables

You can also set the default provider and model using environment variables:

```bash
export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
```

### Configuration Example

The provider configuration is typically stored in a JSON file and looks like:

```json
{
  "openai": {
    "api_key": "sk-...",
    "api_base": "https://api.openai.com/v1",
    "default_model": "gpt-4o-mini"
  },
  "anthropic": {
    "api_key": "sk-...",
    "api_base": "https://api.anthropic.com",
    "default_model": "claude-3-opus"
  },
  "ollama": {
    "api_base": "http://localhost:11434",
    "default_model": "llama3.2"
  }
}
```

## 📈 Advanced Usage Examples

### Provider and Model Selection

You can change providers or models during a session:

```
# In chat mode
> /provider
Current provider: openai
Current model: gpt-4o-mini
To change provider: /provider <provider_name>

> /provider list
Available Providers
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Provider  ┃ Default Model  ┃ API Base                        ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ openai    │ gpt-4o-mini    │ https://api.openai.com/v1       │
│ anthropic │ claude-3-opus  │ https://api.anthropic.com       │
│ ollama    │ llama3.2       │ http://localhost:11434          │
└───────────┴────────────────┴─────────────────────────────────┘

> /provider anthropic
Switched to provider: anthropic with model: claude-3-opus
LLM client updated successfully

> /model claude-3-sonnet
Switched to model: claude-3-sonnet
```

### Working with Tools in Chat Mode

In chat mode, simply ask questions that require tool usage, and the LLM will automatically call the appropriate tools:

```
You: What tables are available in the database?
[Tool Call: list_tables]
Assistant: There's one table in the database named products. How would you like to proceed?

You: Select top 10 products ordered by price in descending order
[Tool Call: read_query]
Assistant: Here are the top 10 products ordered by price in descending order:
  1 Mini Drone - $299.99
  2 Smart Watch - $199.99
  3 Portable SSD - $179.99
  ...
```

### Using Conversation Management

The MCP CLI provides powerful conversation history management:

```
> /conversation
Conversation History (12 messages)
# | Role      | Content
1 | system    | You are an intelligent assistant capable of using t...
2 | user      | What tables are available in the database?
3 | assistant | Let me check for you.
...

> /save conversation.json
Conversation saved to conversation.json

> /compact
Conversation history compacted with summary.
```

## 🛠️ Implementation Details

The provider configuration is managed by the `ProviderConfig` class, which:
- Loads/saves configuration from a local file
- Manages active provider and model settings
- Provides helper methods for retrieving configuration values

The LLM client is created using the `get_llm_client` function, which instantiates the appropriate client based on the provider and model settings.

## 📦 Dependencies

The CLI is organized with optional dependency groups:

- **cli**: Rich terminal UI, command completion, and provider integrations
- **dev**: Development tools and testing utilities
- **wasm**: (Reserved for future WebAssembly support)
- **chuk-mcp**: Protocol implementation library (core dependency)

Install with specific extras using:
```bash
pip install "mcp-cli[cli]"     # Basic CLI features
pip install "mcp-cli[cli,dev]" # CLI with development tools
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Rich](https://github.com/Textualize/rich) for beautiful terminal formatting
- [Typer](https://typer.tiangolo.com/) for CLI argument parsing
- [Prompt Toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) for interactive input
- [CHUK-MCP](https://github.com/chrishayuk/chuk-mcp) for the core protocol implementation
