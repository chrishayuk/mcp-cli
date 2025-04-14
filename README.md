# MCP CLI - Model Context Provider Command Line Interface
A powerful, feature-rich command-line interface for interacting with Model Context Provider servers. This client enables seamless communication with LLMs through integration with the [CHUK-MCP protocol library](https://github.com/chrishayuk/chuk-mcp) which is a pyodide compatible pure python protocol implementation of MCP, supporting tool usage, conversation management, and multiple operational modes.

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
  - watsonx.ai integration (`ibm/granite-3-8b-instruct`, `mistralai/mistral-large`, etc.)
  - Ollama integration (`llama3.2`, `qwen2.5-coder`, etc.)
  - Extensible architecture for additional providers

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
- For watsonx.ai: Valid API key (`WATSONX_API_KEY`), valid project ID (`WATSONX_PROJECT_ID`), endpoint URL (`WATSONX_ENDPOINT_URL`)
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

## 🧰 Command-line Arguments

Global options available for all commands:

- `--server`: Specify the server(s) to connect to (comma-separated for multiple)
- `--config-file`: Path to server configuration file (default: `server_config.json`)
- `--provider`: LLM provider to use (`openai`, `watsonx` or `ollama`, default: `openai`)
- `--model`: Specific model to use (provider-dependent defaults)
- `--disable-filesystem`: Disable filesystem access (default: true)

## 🤖 Using Chat Mode

Chat mode provides a conversational interface with the LLM, automatically using available tools when needed:

```bash
mcp-cli chat --server sqlite
```

With specific provider and model:

```bash
mcp-cli chat --server sqlite --provider openai --model gpt-4o
```

```bash
mcp-cli chat --server sqlite --provider ollama --model llama3.2
```

### Chat Commands

In chat mode, use these slash commands:

#### General Commands
- `/help`: Show available commands
- `/help <command>`: Show detailed help for a specific command
- `/quickhelp` or `/qh`: Display a quick reference of common commands
- `exit` or `quit`: Exit chat mode

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
- `/provider <n>`: Change the current LLM provider 
- `/model <n>`: Change the current LLM model
- `/servers`: List connected servers and their status

## 🖥️ Using Interactive Mode

Interactive mode provides a command-line interface with slash commands for direct server interaction:

```bash
mcp-cli interactive --server sqlite
```

### Interactive Commands

In interactive mode, use these commands:

- `/ping`: Check if server is responsive
- `/prompts`: List available prompts
- `/tools`: List available tools
- `/tools-all`: Show detailed tool information with parameters
- `/tools-raw`: Show raw tool definitions in JSON
- `/resources`: List available resources
- `/chat`: Enter chat mode
- `/cls`: Clear the screen
- `/clear`: Clear the screen and show welcome message
- `/help`: Show help message
- `/exit` or `/quit`: Exit the program

## 📄 Using Command Mode

Command mode provides a Unix-friendly interface for automation and pipeline integration:

```bash
mcp-cli cmd --server sqlite [options]
```

This mode is designed for scripting, batch processing, and direct integration with other Unix tools.

### Command Mode Options

- `--input`: Input file path (use `-` for stdin)
- `--output`: Output file path (use `-` for stdout, default)
- `--prompt`: Prompt template (use `{{input}}` as placeholder for input)
- `--raw`: Output raw text without formatting
- `--tool`: Directly call a specific tool
- `--tool-args`: JSON arguments for tool call
- `--system-prompt`: Custom system prompt

### Command Mode Examples

Process content with LLM:

```bash
# Summarize a document
mcp-cli cmd --server sqlite --input document.md --prompt "Summarize this: {{input}}" --output summary.md

# Process stdin and output to stdout
cat document.md | mcp-cli cmd --server sqlite --input - --prompt "Extract key points: {{input}}"
```

Call tools directly:

```bash
# List database tables
mcp-cli cmd --server sqlite --tool list_tables --raw

# Run a SQL query
mcp-cli cmd --server sqlite --tool read_query --tool-args '{"query": "SELECT COUNT(*) FROM users"}'
```

Batch processing:

```bash
# Process multiple files with GNU Parallel
ls *.md | parallel mcp-cli cmd --server sqlite --input {} --output {}.summary.md --prompt "Summarize: {{input}}"
```

## 🔧 Direct Commands

Run individual commands without entering interactive mode:

```bash
# List available tools
mcp-cli tools list --server sqlite

# Call a specific tool
mcp-cli tools call --server sqlite

# List available prompts
mcp-cli prompts list --server sqlite

# Check server connectivity
mcp-cli ping --server sqlite

# List available resources
mcp-cli resources list --server sqlite
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

## 🏗️ Project Structure

```
src/
├── mcp_cli/
│   ├── chat/                  # Chat mode implementation
│   │   ├── commands/          # Chat slash commands
│   │   │   ├── __init__.py    # Command registration system
│   │   │   ├── conversation.py  # Conversation management
│   │   │   ├── conversation_history.py   
│   │   │   ├── exit.py              
│   │   │   ├── help.py              
│   │   │   ├── help_text.py         
│   │   │   ├── models.py            
│   │   │   ├── servers.py           
│   │   │   ├── tool_history.py      
│   │   │   └── tools.py             
│   │   ├── chat_context.py    # Chat session state management
│   │   ├── chat_handler.py    # Main chat loop handler
│   │   ├── command_completer.py  # Command completion
│   │   ├── conversation.py    # Conversation processor
│   │   ├── system_prompt.py   # System prompt generator
│   │   ├── tool_processor.py  # Tool handling
│   │   └── ui_manager.py      # User interface
│   ├── commands/              # CLI commands
│   │   ├── __init__.py
│   │   ├── chat.py            # Chat command
│   │   ├── cmd.py             # Command mode
│   │   ├── interactive.py     # Interactive mode
│   │   ├── ping.py            # Ping command
│   │   ├── prompts.py         # Prompts commands
│   │   ├── register_commands.py  # Command registration
│   │   ├── resources.py       # Resources commands
│   │   └── tools.py           # Tools commands
│   ├── llm/                   # LLM client implementations
│   │   ├── providers/         # Provider-specific clients
│   │   │   ├── __init__.py
│   │   │   ├── base.py        # Base LLM client
│   │   │   ├── openai_client.py  # OpenAI implementation
│   │   │   ├── ollama_client.py  # ollama implementation
│   │   │   └── watsonx_client.py # watsonx.ai implementation
│   │   ├── llm_client.py      # Client factory
│   │   ├── system_prompt_generator.py  # Prompt generator
│   │   └── tools_handler.py   # Tools handling
│   ├── ui/                    # User interface components
│   │   ├── colors.py          # Color definitions
│   │   └── ui_helpers.py      # UI utilities
│   ├── cli_options.py         # CLI options processing
│   ├── config.py              # Configuration loader
│   ├── main.py                # Main entry point
│   └── run_command.py         # Command execution
```

## 📈 Advanced Usage

### Tool Execution

The MCP CLI can automatically execute tools provided by the server. In chat mode, simply request information that requires tool usage, and the LLM will automatically select and call the appropriate tools.

Example conversation:

```
You: What tables are available in the database?
Assistant: Let me check for you.
[Tool Call: list_tables]
I found the following tables in the database:
- users
- products
- orders
- categories

You: How many users do we have?
Assistant: I'll query the database for that information.
[Tool Call: read_query]
There are 873 users in the database.
```

### Scripting with Command Mode

Command mode enables powerful automation through shell scripts:

```bash
#!/bin/bash
# Example script to analyze multiple documents

# Process all markdown files in the current directory
for file in *.md; do
  echo "Processing $file..."
  
  # Generate summary
  mcp-cli cmd --server sqlite --input "$file" \
    --prompt "Summarize this document: {{input}}" \
    --output "${file%.md}.summary.md"
  
  # Extract entities
  mcp-cli cmd --server sqlite --input "$file" \
    --prompt "Extract all company names, people, and locations from this text: {{input}}" \
    --output "${file%.md}.entities.txt" --raw
done

# Create a combined report
echo "Creating final report..."
cat *.entities.txt | mcp-cli cmd --server sqlite --input - \
  --prompt "Analyze these entities and identify the most frequently mentioned:" \
  --output report.md
```

### Conversation Management

Track and manage your conversation history:

```
> /conversation
Conversation History (12 messages)
# | Role      | Content
1 | system    | You are an intelligent assistant capable of using t...
2 | user      | What tables are available in the database?
3 | assistant | Let me check for you.
4 | assistant | [Tool call: list_tables]
...

> /conversation 4
Message #4 (Role: assistant)
[Tool call: list_tables]
Tool Calls:
  1. ID: call_list_tables_12345678, Type: function, Name: list_tables
     Arguments: {}

> /save conversation.json
Conversation saved to conversation.json

> /compact
Conversation history compacted with summary.
Summary:
The user asked about database tables, and I listed the available tables (users, products, orders, categories). The user then asked about the number of users, and I queried the database to find there are 873 users.
```

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

- [Anthropic Claude](https://www.anthropic.com/claude) for assistance with code development
- [Rich](https://github.com/Textualize/rich) for beautiful terminal formatting
- [Typer](https://typer.tiangolo.com/) for CLI argument parsing
- [Prompt Toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) for interactive input
- [CHUK-MCP](https://github.com/chrishayuk/chuk-mcp) for the core protocol implementation